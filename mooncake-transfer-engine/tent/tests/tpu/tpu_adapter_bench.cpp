// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Microbenchmark of Mooncake's built-in TPU adapter (tent/plugins/tpu) against
// the mock TPU buffer provider, whose "HBM" is host memory. It therefore
// measures Mooncake-side overhead only, not TPU DMA; for real HBM numbers see
// mooncake-transfer-engine/tent/benchmark/tpu_hbm_bench.py.
//
//   tent_tpu_adapter_bench [classify|copy|engine|all]
//
//   classify  TpuPjrtShim::isDevicePtr / deviceIndex latency (run per request
//             by TENT) as the number of registered regions grows.
//   copy      TpuPjrtShim::copyD2H / copyH2D bandwidth vs a plain memcpy.
//   engine    Two TransferEngines over loopback TCP: staged TPU->TPU transfers
//             vs host->host transfers of the same size.
//
// Not registered with ctest.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mock_tpu_env.h"
#include "tent/common/config.h"
#include "tent/platform/tpu_pjrt_shim.h"
#include "tent/transfer_engine.h"

#ifndef TPU_PJRT_ADAPTER_LIB
#error "TPU_PJRT_ADAPTER_LIB must be defined by the build"
#endif
#ifndef MOCK_TPU_BUFFER_PROVIDER_LIB
#error "MOCK_TPU_BUFFER_PROVIDER_LIB must be defined by the build"
#endif

using namespace mooncake::tent;
using Clock = std::chrono::steady_clock;

namespace {

void check(const Status& status) {
    if (!status.ok()) throw std::runtime_error(status.ToString());
}

double seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

double percentile(std::vector<double> samples, double ratio) {
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(ratio * (samples.size() - 1))];
}

std::string human(size_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB"};
    int unit = 0;
    double value = static_cast<double>(bytes);
    while (value >= 1024 && unit < 3) {
        value /= 1024;
        ++unit;
    }
    std::ostringstream out;
    out << std::setprecision(value < 10 ? 2 : 4) << value << ' ' << units[unit];
    return out.str();
}

// Iterations moving roughly `target` bytes, within [min_iters, max_iters].
size_t itersFor(size_t bytes, size_t target, size_t min_iters,
                size_t max_iters) {
    return std::clamp<size_t>(target / bytes, min_iters, max_iters);
}

MockTpuBuffer create(size_t size, int device = 0) {
    auto buffer = MockTpuEnv::instance().create(size, device);
    if (!buffer) throw std::runtime_error("adapter registration failed");
    return buffer;
}

// --- classify ----------------------------------------------------------------

void runClassify() {
    constexpr size_t kRegionSize = 64 * 1024;
    constexpr size_t kCallsPerSample = 1000;
    constexpr size_t kSamples = 300;
    auto& shim = TpuPjrtShim::instance();
    std::vector<uint8_t> host(4096);
    std::mt19937_64 random(12345);

    std::cout << "\n== classify: shim lookup latency, ns/call (p50 / p99 over "
              << kSamples << " samples of " << kCallsPerSample << " calls)\n";
    std::cout << std::left << std::setw(10) << "regions" << std::right
              << std::setw(22) << "isDevicePtr(hit)" << std::setw(22)
              << "isDevicePtr(miss)" << std::setw(22) << "deviceIndex" << '\n';

    std::vector<MockTpuBuffer> regions;
    for (size_t count : {1, 16, 256, 4096}) {
        while (regions.size() < count)
            regions.push_back(create(kRegionSize, regions.size() % 8));
        std::vector<const void*> hits(kCallsPerSample);
        for (auto& hit : hits) {
            auto& region = regions[random() % regions.size()];
            hit = region.at(random() % kRegionSize);
        }
        auto measure = [&](auto&& call) {
            std::vector<double> samples;
            volatile int sink = 0;
            for (size_t s = 0; s < kSamples; ++s) {
                int sum = 0;
                auto start = Clock::now();
                for (size_t i = 0; i < kCallsPerSample; ++i) sum += call(i);
                samples.push_back(seconds(start, Clock::now()) * 1e9 /
                                  kCallsPerSample);
                sink = sum;
            }
            (void)sink;
            std::ostringstream out;
            out << std::fixed << std::setprecision(1)
                << percentile(samples, 0.5) << " / "
                << percentile(samples, 0.99);
            return out.str();
        };
        auto hit = measure([&](size_t i) { return shim.isDevicePtr(hits[i]); });
        auto miss = measure(
            [&](size_t i) { return shim.isDevicePtr(host.data() + i % 4096); });
        auto index =
            measure([&](size_t i) { return shim.deviceIndex(hits[i]); });
        std::cout << std::left << std::setw(10) << count << std::right
                  << std::setw(22) << hit << std::setw(22) << miss
                  << std::setw(22) << index << '\n';
    }
    for (auto& region : regions) MockTpuEnv::instance().unregister(region);
}

// --- copy
// ----------------------------------------------------------------------

void runCopy() {
    constexpr size_t kMax = 64UL << 20;
    auto& shim = TpuPjrtShim::instance();
    MockTpuBuffer device = create(kMax);
    std::vector<uint8_t> host(kMax), host2(kMax, 1);

    std::cout << "\n== copy: shim D2H / H2D vs memcpy, GB/s (mock HBM is host "
                 "memory)\n";
    std::cout << std::left << std::setw(12) << "size" << std::right
              << std::setw(12) << "memcpy" << std::setw(12) << "D2H"
              << std::setw(12) << "H2D" << std::setw(16) << "D2H us/call"
              << '\n';
    for (size_t size = 4096; size <= kMax; size *= 4) {
        const size_t iters = itersFor(size, 4UL << 30, 10, 20000);
        auto bench = [&](auto&& copy) {
            for (size_t i = 0; i < std::min<size_t>(iters, 5); ++i) copy();
            auto start = Clock::now();
            for (size_t i = 0; i < iters; ++i) copy();
            return seconds(start, Clock::now());
        };
        double t_memcpy =
            bench([&] { std::memcpy(host.data(), host2.data(), size); });
        double t_d2h = bench(
            [&] { check(shim.copyD2H(host.data(), device.addr(), size)); });
        double t_h2d = bench(
            [&] { check(shim.copyH2D(device.addr(), host.data(), size)); });
        auto gbps = [&](double t) { return size * iters / t / 1e9; };
        std::cout << std::left << std::setw(12) << human(size) << std::right
                  << std::fixed << std::setprecision(2) << std::setw(12)
                  << gbps(t_memcpy) << std::setw(12) << gbps(t_d2h)
                  << std::setw(12) << gbps(t_h2d) << std::setw(16)
                  << t_d2h / iters * 1e6 << '\n';
    }
    MockTpuEnv::instance().unregister(device);
}

// --- engine
// --------------------------------------------------------------------

std::shared_ptr<Config> engineConfig() {
    auto c = std::make_shared<Config>();
    c->set("metadata_type", "p2p");
    c->set("metadata_servers", "P2PHANDSHAKE");
    c->set("rpc_server_hostname", "127.0.0.1");
    c->set("rpc_server_port", "0");
    c->set("log_level", "error");
    c->set("transports/tcp/enable", true);
    c->set("transports/rdma/enable", false);
    c->set("transports/shm/enable", false);
    c->set("transports/io_uring/enable", false);
    c->set("transports/tpu/enable", true);
    return c;
}

// Times `iters` single-request batches; returns per-transfer seconds.
std::vector<double> timeTransfers(TransferEngine& client, Request request,
                                  size_t iters) {
    std::vector<double> samples;
    for (size_t i = 0; i < iters + 2; ++i) {
        BatchID batch = client.allocateBatch(1);
        if (!batch) throw std::runtime_error("batch allocation failed");
        auto start = Clock::now();
        check(client.submitTransfer(batch, {request}));
        for (;;) {
            TransferStatus status{};
            check(client.getTransferStatus(batch, status));
            if (status.s == COMPLETED) break;
            if (status.s != PENDING && status.s != INITIAL)
                throw std::runtime_error("transfer failed");
            if (Clock::now() - start > std::chrono::seconds(60))
                throw std::runtime_error("transfer timed out");
            std::this_thread::yield();
        }
        auto end = Clock::now();
        check(client.freeBatch(batch));
        if (i >= 2) samples.push_back(seconds(start, end));  // 2 warmups
    }
    return samples;
}

void runEngine() {
    constexpr size_t kMax = 64UL << 20;
    // Device buffers are registered with the adapter before the engines exist.
    MockTpuBuffer client_tpu = create(kMax), server_tpu = create(kMax);
    TransferEngine server(engineConfig()), client(engineConfig());
    if (!server.available() || !client.available())
        throw std::runtime_error("TCP engine initialization failed");
    std::vector<uint8_t> client_host(kMax), server_host(kMax);
    check(client.registerLocalMemory(client_tpu.addr(), kMax));
    check(server.registerLocalMemory(server_tpu.addr(), kMax));
    check(client.registerLocalMemory(client_host.data(), kMax));
    check(server.registerLocalMemory(server_host.data(), kMax));
    SegmentID segment{};
    check(client.openSegment(segment, server.getSegmentName()));

    std::cout << "\n== engine: loopback TCP, single-request batches, GB/s "
                 "(p50 latency us)\n";
    std::cout << std::left << std::setw(12) << "size" << std::right
              << std::setw(24) << "host->host WRITE" << std::setw(24)
              << "tpu->tpu WRITE" << std::setw(24) << "tpu->tpu READ" << '\n';
    for (size_t size : {64UL << 10, 1UL << 20, 4UL << 20, 16UL << 20, kMax}) {
        const size_t iters = itersFor(size, 512UL << 20, 3, 200);
        auto cell = [&](void* local, void* remote, Request::OpCode op) {
            Request request{};
            request.opcode = op;
            request.source = local;
            request.target_id = segment;
            request.target_offset = reinterpret_cast<uint64_t>(remote);
            request.length = size;
            auto samples = timeTransfers(client, request, iters);
            double total = 0;
            for (double s : samples) total += s;
            std::ostringstream out;
            out << std::fixed << std::setprecision(2)
                << size * samples.size() / total / 1e9 << " ("
                << std::setprecision(0) << percentile(samples, 0.5) * 1e6
                << ")";
            return out.str();
        };
        std::cout << std::left << std::setw(12) << human(size) << std::right
                  << std::setw(24)
                  << cell(client_host.data(), server_host.data(),
                          Request::WRITE)
                  << std::setw(24)
                  << cell(client_tpu.addr(), server_tpu.addr(), Request::WRITE)
                  << std::setw(24)
                  << cell(client_tpu.addr(), server_tpu.addr(), Request::READ)
                  << std::endl;
    }
    // Spot-check that the staged path really moved the bytes.
    std::memset(server_tpu.data, 0, kMax);
    Request request{};
    request.opcode = Request::WRITE;
    request.source = client_tpu.addr();
    request.target_id = segment;
    request.target_offset = server_tpu.fake_addr;
    request.length = kMax;
    timeTransfers(client, request, 1);
    if (std::memcmp(client_tpu.data, server_tpu.data, kMax) != 0)
        throw std::runtime_error("TPU payload mismatch");

    check(client.closeSegment(segment));
    check(client.unregisterLocalMemory(client_host.data(), kMax));
    check(server.unregisterLocalMemory(server_host.data(), kMax));
    check(client.unregisterLocalMemory(client_tpu.addr(), kMax));
    check(server.unregisterLocalMemory(server_tpu.addr(), kMax));
    MockTpuEnv::instance().unregister(client_tpu);
    MockTpuEnv::instance().unregister(server_tpu);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string section = argc > 1 ? argv[1] : "all";
    if (section != "all" && section != "classify" && section != "copy" &&
        section != "engine") {
        std::cerr << "usage: " << argv[0] << " [classify|copy|engine|all]\n";
        return 2;
    }
    try {
        std::string error;
        if (!MockTpuEnv::instance().load(TPU_PJRT_ADAPTER_LIB,
                                         MOCK_TPU_BUFFER_PROVIDER_LIB, &error))
            throw std::runtime_error(error);
        // The shim reads the topology once; register all 8 devices first.
        std::vector<MockTpuBuffer> anchors;
        for (int device = 0; device < 8; ++device)
            anchors.push_back(create(4096, device));
        if (!TpuPjrtShim::instance().available())
            throw std::runtime_error("TpuPjrtShim did not load the adapter");
        std::cout << "TPU adapter microbenchmark (mock provider: host-memory "
                     "HBM; Mooncake overhead only)\n";
        if (section == "all" || section == "classify") runClassify();
        if (section == "all" || section == "copy") runCopy();
        if (section == "all" || section == "engine") runEngine();
        for (auto& anchor : anchors) MockTpuEnv::instance().unregister(anchor);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
