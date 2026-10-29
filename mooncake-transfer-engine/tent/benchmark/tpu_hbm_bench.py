#!/usr/bin/env python3
# Copyright 2026 KVCache.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real-HBM benchmark of Mooncake TENT's TPU path on a TPU VM.

JAX arrays are exported through a TPU buffer provider (tpu-sync by default),
registered with ``mooncake.tpu`` and moved by two TENT engines in this process
over loopback TCP. Three sections:

* ``copy``:   the adapter's HBM<->host data path (``mc_tpu_pjrt_copy_d2h/h2d``,
  i.e. provider DMA plus fake-address lookup) into page-aligned host memory,
  as TENT's staging buffers are, and into DMA-mapped host memory (the ceiling
  if the staging pool were mapped), next to JAX's own
  ``device_get``/``device_put`` for the same bytes.
* ``engine``: end-to-end TENT transfers between two TPU chips (staged through
  host DRAM), TPU<->host, and a host->host baseline through the same engines.
* ``shards``: one batch moving every shard of an array sharded over several
  chips into an array sharded over other chips (the KV-cache layout).

Every transfer section verifies the destination bytes. For a mock-provider
microbenchmark that needs no TPU, see tent/tests/tpu/tpu_adapter_bench.cpp.

Requirements: Mooncake built with -DUSE_TENT=ON -DUSE_TPU=ON, JAX on TPU, and
tpu-sync's provider (tpu_sync.api.jax.mooncake_provider) importable.

  python tpu_hbm_bench.py --sizes 1M,16M,256M --iters 10
"""

import argparse
import ctypes
import json
import mmap
import os
import statistics
import sys
import time

# TENT must be selected before mooncake.engine creates its engine.
os.environ.setdefault("MC_USE_TENT", "1")
os.environ.setdefault(
    "MC_TENT_CONF",
    json.dumps(
        {
            "log_level": "warning",
            "transports": {
                "tcp": {"enable": True},
                "hp_tcp": {"enable": False},
                "rdma": {"enable": False},
                "shm": {"enable": False},
                "io_uring": {"enable": False},
                "tpu": {"enable": True},
            },
        }
    ),
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec  # noqa: E402

from mooncake import tpu  # noqa: E402

ROW_BYTES = 4096  # One float32 row of 1024 elements.


def parse_size(text):
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    text = text.strip().upper().rstrip("B").rstrip("I")
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def human(nbytes):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:g} {unit}" if unit == "B" else f"{nbytes:.4g} {unit}"
        nbytes /= 1024


def make_array(nbytes, device, seed):
    """A float32 (rows, 1024) array on `device`; rows are a multiple of 8 so
    the TPU's tiled layout has no padding."""
    rows = max(8, (nbytes // ROW_BYTES + 7) // 8 * 8)
    host = np.random.default_rng(seed).standard_normal((rows, 1024), np.float32)
    return jax.device_put(host, device).block_until_ready()


class Adapter:
    """Raw data-path ABI of the adapter mooncake.tpu loaded."""

    def __init__(self, path):
        lib = ctypes.CDLL(path)
        for name in ("mc_tpu_pjrt_copy_d2h", "mc_tpu_pjrt_copy_h2d"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_int
        self.lib = lib

    def d2h(self, host, fake_addr, nbytes):
        if self.lib.mc_tpu_pjrt_copy_d2h(host.ctypes.data, fake_addr, nbytes):
            raise RuntimeError("adapter D2H failed")

    def h2d(self, fake_addr, host, nbytes):
        if self.lib.mc_tpu_pjrt_copy_h2d(fake_addr, host.ctypes.data, nbytes):
            raise RuntimeError("adapter H2D failed")

    def read(self, fake_addr, nbytes):
        host = np.empty(nbytes, np.uint8)
        self.d2h(host, fake_addr, nbytes)
        return host


def timed(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def cell(nbytes, samples):
    gbps = nbytes * len(samples) / sum(samples) / 1e9
    return f"{gbps:8.2f} ({statistics.median(samples) * 1e6:9.0f})"


def print_header(title, columns):
    print(f"\n== {title}")
    print(f"{'size':<12}" + "".join(f"{c:>26}" for c in columns))


# --- copy ---------------------------------------------------------------------


HUGE_PAGE = 2 << 20


class HostRegion:
    """Prefaulted host memory for the copy section, 2 MiB aligned.

    The host side's page size dominates provider copy speed when the memory is
    not DMA-mapped (measured with tpu-sync on v6e: ~0.3 GB/s on 4 KiB pages vs
    ~14-20 GB/s on 2 MiB pages). TENT's staging pool comes from
    numa_alloc_onnode, i.e. anonymous mmap, so it gets huge pages only through
    transparent hugepages; `pages="huge"` models that and `pages="small"` a
    host with THP off. `dma_map` registers the region with the TPU's DMA
    engine (PjRtClient::DmaMap), the ceiling.
    """

    def __init__(self, nbytes, pages, dma_map_device=None):
        size = (nbytes + HUGE_PAGE - 1) // HUGE_PAGE * HUGE_PAGE
        self._mmap = mmap.mmap(
            -1, size + HUGE_PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        base = np.frombuffer(self._mmap, np.uint8)
        start = (-base.ctypes.data) % HUGE_PAGE
        offset = start  # mmap.madvise needs a page-aligned start
        self._mmap.madvise(
            mmap.MADV_HUGEPAGE if pages == "huge" else mmap.MADV_NOHUGEPAGE,
            offset,
            size,
        )
        self.array = base[start : start + nbytes]
        base[start : start + size] = 0  # prefault
        self._device = dma_map_device
        if dma_map_device is not None:
            dma_map_device.client.dma_map(base.ctypes.data + start, size)
            self._mapped = base.ctypes.data + start

    def close(self):
        if self._device is not None:
            self._device.client.dma_unmap(self._mapped)
        del self.array
        self._mmap.close()


def run_copy(args, adapter, devices):
    print_header(
        "copy: HBM<->host, GB/s (p50 us); host memory on 2 MiB pages (TENT's "
        "staging pool under THP), 4 KiB pages, or DMA-mapped",
        [
            "D2H 2MiB pages",
            "H2D 2MiB pages",
            "D2H 4KiB pages",
            "D2H dma-mapped",
            "H2D dma-mapped",
            "jax.device_get",
            "jax.device_put",
        ],
    )
    device = devices[args.src_device]
    for nbytes in args.sizes:
        array = make_array(nbytes, device, seed=1)
        shards = tpu.prepare([array])
        shard = shards[0][0]
        size = shard.size
        regions = [
            HostRegion(size, "huge"),
            HostRegion(size, "small"),
            HostRegion(size, "huge", dma_map_device=device),
        ]
        huge, small, mapped = (r.array for r in regions)
        addr = shard.fake_addr
        try:
            cols = [
                timed(lambda: adapter.d2h(huge, addr, size), args.iters),
                timed(lambda: adapter.h2d(addr, huge, size), args.iters),
                timed(lambda: adapter.d2h(small, addr, size), args.iters),
                timed(lambda: adapter.d2h(mapped, addr, size), args.iters),
                timed(lambda: adapter.h2d(addr, mapped, size), args.iters),
            ]
            # JAX caches an array's host value, so time fresh copies.
            fresh = [(array + 0).block_until_ready() for _ in range(args.iters + 2)]
            it = iter(fresh)
            cols.append(timed(lambda: jax.device_get(next(it)), args.iters))
            del fresh
            host_value = np.asarray(array)
            cols.append(
                timed(
                    lambda: jax.device_put(host_value, device).block_until_ready(),
                    args.iters,
                )
            )
        finally:
            huge = small = mapped = None  # Drop views so the mmaps can close.
            for region in regions:
                region.close()
            tpu.unregister(shards[0])
        print(f"{human(size):<12}" + "".join(f"{cell(size, c):>26}" for c in cols))


# --- engine -------------------------------------------------------------------


class Engines:
    """Two TENT engines in this process; `initiator` reaches `target`."""

    def __init__(self):
        from mooncake.engine import TransferEngine

        self.target = TransferEngine()
        self.initiator = TransferEngine()
        for engine in (self.target, self.initiator):
            if engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") != 0:
                raise RuntimeError("TransferEngine.initialize failed")
        self.target_name = f"127.0.0.1:{self.target.get_rpc_port()}"
        self.registered = []

    def register_host(self, engine, nbytes):
        host = np.zeros(nbytes, np.uint8)
        if engine.register_memory(host.ctypes.data, nbytes) != 0:
            raise RuntimeError("register_memory failed")
        self.registered.append((engine, host.ctypes.data, host))
        return host

    def register_tpu(self, engine, shards):
        tpu.register_with_engine(engine, [shards])
        for shard in shards:
            self.registered.append((engine, shard.fake_addr, None))

    def write(self, local, remote, nbytes):
        rc = self.initiator.transfer_sync_write(self.target_name, local, remote, nbytes)
        if rc != 0:
            raise RuntimeError(f"transfer_sync_write failed: {rc}")

    def read(self, local, remote, nbytes):
        rc = self.initiator.transfer_sync_read(self.target_name, local, remote, nbytes)
        if rc != 0:
            raise RuntimeError(f"transfer_sync_read failed: {rc}")

    def batch_write(self, locals_, remotes, lengths):
        rc = self.initiator.batch_transfer_sync_write(
            self.target_name, locals_, remotes, lengths
        )
        if rc != 0:
            raise RuntimeError(f"batch_transfer_sync_write failed: {rc}")

    def unregister_all(self):
        for engine, addr, _ in reversed(self.registered):
            engine.unregister_memory(addr)
        self.registered.clear()


def prepare_engine_arrays(args, devices, all_shards):
    src_arrays = {n: make_array(n, devices[args.src_device], 2) for n in args.sizes}
    dst_arrays = {n: make_array(n, devices[args.dst_device], 3) for n in args.sizes}
    src = {n: tpu.prepare([a])[0][0] for n, a in src_arrays.items()}
    dst = {n: tpu.prepare([a])[0][0] for n, a in dst_arrays.items()}
    all_shards.extend(list(src.values()) + list(dst.values()))
    return src_arrays, dst_arrays, src, dst


def bench_engine(args, adapter, engines, src_arrays, dst_arrays, src, dst):
    for n in args.sizes:
        engines.register_tpu(engines.initiator, [src[n]])
        engines.register_tpu(engines.target, [dst[n]])
    max_size = max(s.size for s in list(src.values()) + list(dst.values()))
    local_host = engines.register_host(engines.initiator, max_size)
    remote_host = engines.register_host(engines.target, max_size)
    local_host[:] = np.random.default_rng(4).integers(0, 255, max_size, np.uint8)

    print_header(
        f"engine: TENT over loopback TCP, tpu:{args.src_device} -> "
        f"tpu:{args.dst_device}, GB/s (p50 us)",
        [
            "host->host WRITE",
            "tpu->tpu WRITE",
            "tpu->tpu READ",
            "tpu->host WRITE",
            "host->tpu WRITE",
        ],
    )
    for n in args.sizes:
        s, d = src[n], dst[n]
        size = s.size
        lh, rh = local_host.ctypes.data, remote_host.ctypes.data
        cols = [
            timed(lambda: engines.write(lh, rh, size), args.iters),
            timed(lambda: engines.write(s.fake_addr, d.fake_addr, size), args.iters),
            timed(lambda: engines.read(s.fake_addr, d.fake_addr, size), args.iters),
            timed(lambda: engines.write(s.fake_addr, rh, size), args.iters),
            timed(lambda: engines.write(lh, d.fake_addr, size), args.iters),
        ]
        print(f"{human(size):<12}" + "".join(f"{cell(size, c):>26}" for c in cols))

    # Verify: host->tpu landed, then tpu->tpu landed and JAX sees it.
    for n in args.sizes:
        s, d = src[n], dst[n]
        engines.write(local_host.ctypes.data, d.fake_addr, d.size)
        if not np.array_equal(adapter.read(d.fake_addr, d.size), local_host[: d.size]):
            raise RuntimeError(f"host->tpu payload mismatch at {human(d.size)}")
        engines.write(s.fake_addr, d.fake_addr, s.size)
        if not np.array_equal(np.asarray(dst_arrays[n]), np.asarray(src_arrays[n])):
            raise RuntimeError(f"tpu->tpu payload mismatch at {human(s.size)}")
    print("verified: destination HBM matches the source (raw bytes and via JAX)")


def bench_shards(args, adapter, engines, all_shards, devices):
    k = args.shards
    if 2 * k > len(devices):
        print(f"\n== shards: skipped, needs {2 * k} TPU devices")
        return
    src_mesh = Mesh(np.array(devices[:k]), ("x",))
    dst_mesh = Mesh(np.array(devices[k : 2 * k]), ("x",))
    print_header(
        f"shards: one batch, {k} shards on tpu:0-{k - 1} -> tpu:{k}-{2 * k - 1}, "
        "aggregate GB/s (p50 us)",
        ["batch WRITE"],
    )
    for n in args.sizes:
        rows = max(8 * k, (n // ROW_BYTES + 8 * k - 1) // (8 * k) * (8 * k))
        host = np.random.default_rng(5).standard_normal((rows, 1024), np.float32)
        spec = PartitionSpec("x", None)
        src_array = jax.device_put(host, NamedSharding(src_mesh, spec))
        dst_array = jax.device_put(np.zeros_like(host), NamedSharding(dst_mesh, spec))
        jax.block_until_ready((src_array, dst_array))
        (src_shards,) = tpu.prepare([src_array])
        (dst_shards,) = tpu.prepare([dst_array])
        all_shards.extend(src_shards + dst_shards)
        engines.register_tpu(engines.initiator, src_shards)
        engines.register_tpu(engines.target, dst_shards)
        locals_ = [s.fake_addr for s in src_shards]
        remotes = [s.fake_addr for s in dst_shards]
        lengths = [s.size for s in src_shards]
        samples = timed(
            lambda: engines.batch_write(locals_, remotes, lengths), args.iters
        )
        total = sum(lengths)
        print(f"{human(total):<12}{cell(total, samples):>26}")
        if not np.array_equal(np.asarray(dst_array), host):
            raise RuntimeError(f"sharded payload mismatch at {human(total)}")
    print("verified: every destination shard matches its source shard (via JAX)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", default="1M,16M,64M,256M")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--src-device", type=int, default=0)
    parser.add_argument("--dst-device", type=int, default=1)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--sections", default="copy,engine,shards", help="comma-separated subset"
    )
    parser.add_argument("--adapter", default=None, help="libmooncake_tpu_pjrt.so")
    args = parser.parse_args()
    args.sizes = [parse_size(s) for s in args.sizes.split(",")]
    sections = set(args.sections.split(","))

    devices = jax.local_devices(backend="tpu")
    if not devices:
        sys.exit("no TPU devices visible to JAX")
    adapter_lib = tpu.get_adapter(args.adapter)
    adapter = Adapter(adapter_lib.path)
    print(
        f"TPU HBM benchmark: {len(devices)} x {devices[0].device_kind}, "
        f"jax {jax.__version__}, adapter {adapter_lib.path}"
    )

    if "copy" in sections:
        run_copy(args, adapter, devices)
    if not sections & {"engine", "shards"}:
        return

    # TENT reads the TPU topology when the engine is created, so register
    # buffers on every device the transfers will touch first.
    all_shards = []
    src_arrays, dst_arrays, src, dst = prepare_engine_arrays(args, devices, all_shards)
    anchors = [jax.device_put(jnp.zeros((8, 1024), jnp.float32), d) for d in devices]
    all_shards.extend(s for per_array in tpu.prepare(anchors) for s in per_array)
    engines = Engines()
    try:
        if "engine" in sections:
            bench_engine(args, adapter, engines, src_arrays, dst_arrays, src, dst)
        if "shards" in sections:
            bench_shards(args, adapter, engines, all_shards, devices)
    finally:
        engines.unregister_all()
        tpu.unregister(all_shards)


if __name__ == "__main__":
    main()
