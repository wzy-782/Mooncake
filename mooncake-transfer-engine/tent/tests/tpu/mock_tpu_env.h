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

// Test/benchmark helper: loads Mooncake's built-in TPU adapter
// (libmooncake_tpu_pjrt.so) and the mock TPU buffer provider, and registers
// mock "HBM" buffers with the adapter, the way mooncake.tpu does for real
// framework buffers. The adapter state is process-wide, so this is too.

#ifndef TENT_TESTS_TPU_MOCK_TPU_ENV_H_
#define TENT_TESTS_TPU_MOCK_TPU_ENV_H_

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace mooncake {
namespace tent {

// One mock TPU buffer registered with the adapter.
struct MockTpuBuffer {
    uint64_t handle = 0;     // Provider handle, owned by the adapter.
    uint64_t fake_addr = 0;  // Address to register with the TransferEngine.
    uint64_t size = 0;
    int device = -1;
    uint8_t *data = nullptr;  // The mock provider's bytes (the "HBM").

    explicit operator bool() const { return fake_addr != 0; }
    void *addr() const { return reinterpret_cast<void *>(fake_addr); }
    uint8_t *at(size_t offset) const {
        return reinterpret_cast<uint8_t *>(fake_addr) + offset;
    }
};

class MockTpuEnv {
   public:
    static MockTpuEnv &instance() {
        static MockTpuEnv env;
        return env;
    }

    // Points TENT's TpuPjrtShim at `adapter_path` and binds the provider at
    // `provider_path`. Call it before the shim is first used, i.e. before the
    // TPU platform or any TransferEngine is created, and register at least one
    // buffer before that too: TENT reads the device topology at that point.
    bool load(const char *adapter_path, const char *provider_path,
              std::string *error) {
        if (adapter_) return true;
        ::setenv("MC_TPU_PJRT_LIB", adapter_path, /*overwrite=*/1);
        // RTLD_LOCAL: the shim's later dlopen() of the same path returns this
        // image, so the test, the shim and the adapter share one registry.
        void *adapter = dlopen(adapter_path, RTLD_NOW | RTLD_LOCAL);
        void *provider = dlopen(provider_path, RTLD_NOW | RTLD_LOCAL);
        if (!adapter || !provider) {
            *error = dlerror();
            return false;
        }
        load_provider_ =
            sym<int (*)(const char *)>(adapter, "mc_tpu_adapter_load_provider");
        register_ = sym<int (*)(uint64_t, uint64_t *, uint64_t *, int *)>(
            adapter, "mc_tpu_adapter_register");
        unregister_ =
            sym<int (*)(uint64_t)>(adapter, "mc_tpu_adapter_unregister");
        create_ = sym<uint64_t (*)(size_t, int, int)>(
            provider, "mock_provider_create_buffer");
        data_ =
            sym<uint8_t *(*)(uint64_t)>(provider, "mock_provider_buffer_data");
        live_count_ = sym<int (*)()>(provider, "mock_provider_live_count");
        set_fail_copies_ =
            sym<void (*)(int)>(provider, "mock_provider_set_fail_copies");
        set_device_alignment_ = sym<void (*)(size_t)>(
            provider, "mock_provider_set_device_alignment");
        rejected_count_ =
            sym<int (*)()>(provider, "mock_provider_rejected_count");
        if (!load_provider_ || !register_ || !unregister_ || !create_ ||
            !data_ || !live_count_ || !set_fail_copies_ ||
            !set_device_alignment_ || !rejected_count_) {
            *error = "adapter or mock provider is missing a symbol";
            return false;
        }
        if (load_provider_(provider_path) != 0) {
            *error = "mc_tpu_adapter_load_provider failed";
            return false;
        }
        adapter_ = adapter;
        return true;
    }

    // Creates a mock buffer on `device` (filled with i % 251) and registers it
    // with the adapter. Returns an empty MockTpuBuffer on failure.
    MockTpuBuffer create(size_t size, int device, int numa = 0) {
        MockTpuBuffer b;
        b.handle = create_(size, device, numa);
        b.data = data_(b.handle);
        uint64_t fake_addr = 0, registered_size = 0;
        int registered_device = -1;
        if (register_(b.handle, &fake_addr, &registered_size,
                      &registered_device) != 0) {
            return MockTpuBuffer{};
        }
        b.fake_addr = fake_addr;
        b.size = registered_size;
        b.device = registered_device;
        return b;
    }

    // Unregisters from the adapter (unregister from the engine first); the
    // adapter then releases the provider handle.
    bool unregister(const MockTpuBuffer &b) {
        return unregister_(b.fake_addr) == 0;
    }

    int liveCount() const { return live_count_(); }
    void setFailCopies(bool fail) { set_fail_copies_(fail ? 1 : 0); }
    // Copies at device offsets that are not multiples of `alignment` fail;
    // 0 accepts any offset. tpu-sync requires 32.
    void setDeviceAlignment(size_t alignment) {
        set_device_alignment_(alignment);
    }
    int rejectedCount() const { return rejected_count_(); }

   private:
    template <typename Fn>
    static Fn sym(void *lib, const char *name) {
        return reinterpret_cast<Fn>(dlsym(lib, name));
    }

    void *adapter_ = nullptr;  // Never dlclose()d, like the shim's handle.
    int (*load_provider_)(const char *) = nullptr;
    int (*register_)(uint64_t, uint64_t *, uint64_t *, int *) = nullptr;
    int (*unregister_)(uint64_t) = nullptr;
    uint64_t (*create_)(size_t, int, int) = nullptr;
    uint8_t *(*data_)(uint64_t) = nullptr;
    int (*live_count_)() = nullptr;
    void (*set_fail_copies_)(int) = nullptr;
    void (*set_device_alignment_)(size_t) = nullptr;
    int (*rejected_count_)() = nullptr;
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_TESTS_TPU_MOCK_TPU_ENV_H_
