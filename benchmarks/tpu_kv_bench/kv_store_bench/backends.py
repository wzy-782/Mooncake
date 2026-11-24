#!/usr/bin/env python3
"""Unified TPU KV-store backend abstractions for Mooncake Store and tpu-sync.

Important runtime rule: on TPU hosts, `jax` must be imported BEFORE `mooncake`
or `tpu_sync` C++ extensions so jaxlib initializes XLA first.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from workloads import KVGeometry


# Mooncake Store ErrorCodes (from mooncake-store/include/types.h).
MC_NO_AVAILABLE_HANDLE = -200
MC_OBJECT_NOT_FOUND = -704
MC_OBJECT_ALREADY_EXISTS = -705


def pick_free_port(bind_ip: str = "0.0.0.0") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((bind_ip, 0))
        return int(s.getsockname()[1])


def hash_to_stamp_value(hash_hex: str) -> float:
    """Deterministic finite float32 tag derived from a block hash for verification."""
    raw = bytes.fromhex(hash_hex)[:4]
    val = struct.unpack(">I", raw)[0]
    # Keep values well-represented in both bf16 and fp32 (integer in [1, 1024]).
    return float((val % 1024) + 1)


@dataclasses.dataclass
class LookupHit:
    hash_hex: str
    tier: str  # "local_dram", "remote_dram", "storage"
    raw_handle: Any = None  # Backend-specific metadata (e.g. RaidenBlockId).
    pinned: bool = False


@dataclasses.dataclass
class LookupResult:
    hits: List[LookupHit]  # Contiguous prefix hits (stops at first miss).
    raw_hit_count: int  # Total hits across all queried hashes.
    latency_ms: float


@dataclasses.dataclass
class TransferResult:
    ok_count: int
    failed_count: int
    rejected_count: int
    bytes_transferred: int
    remote_bytes: int
    latency_ms: float
    ok_indices: List[int] = dataclasses.field(default_factory=list)


class HBMBlockPool:
    """Fixed-capacity Tier-0 HBM prefix cache + scratch slot allocator.

    Shared identically by Mooncake and tpu-sync workers so Tier-0 HBM hit rates
    and slot eviction behavior are strictly apples-to-apples.
    """

    def __init__(self, total_slots: int, cache_slots: int):
        if cache_slots < 0 or cache_slots > total_slots:
            raise ValueError(f"cache_slots={cache_slots} out of [0, {total_slots}]")
        self.total_slots = total_slots
        self.cache_slots = cache_slots
        self._lock = threading.Lock()
        # LRU map: hash_hex -> slot_id (only populated when cache_slots > 0).
        self._lru: collections.OrderedDict[str, int] = collections.OrderedDict()
        self._slot_to_hash: Dict[int, str] = {}
        self._free_slots: collections.deque[int] = collections.deque(range(total_slots))
        self._pin_counts: Dict[int, int] = {}

    def clear(self) -> None:
        with self._lock:
            self._lru.clear()
            self._slot_to_hash.clear()
            self._free_slots = collections.deque(range(self.total_slots))
            self._pin_counts.clear()

    def lookup_prefix(self, hashes_hex: Sequence[str]) -> Tuple[List[int], int]:
        """Match contiguous prefix in HBM and pin matched slots.

        Returns (pinned_slots, prefix_hit_len). Caller must call release_slots().
        """
        if self.cache_slots <= 0:
            return [], 0
        pinned: List[int] = []
        with self._lock:
            for h in hashes_hex:
                slot = self._lru.get(h)
                if slot is None:
                    break
                self._lru.move_to_end(h)
                self._pin_counts[slot] = self._pin_counts.get(slot, 0) + 1
                pinned.append(slot)
        return pinned, len(pinned)

    def acquire_slots(self, count: int) -> List[int]:
        """Allocate `count` pinned HBM slots (evicting unpinned LRU entries if needed)."""
        if count <= 0:
            return []
        slots: List[int] = []
        with self._lock:
            for _ in range(count):
                slot = self._pop_slot_locked()
                if slot is None:
                    # Roll back partial acquisition.
                    for s in slots:
                        self._dec_pin_locked(s)
                    raise RuntimeError(
                        f"HBMBlockPool exhausted (total={self.total_slots}, "
                        f"requested={count}, pinned={len(self._pin_counts)})"
                    )
                self._pin_counts[slot] = self._pin_counts.get(slot, 0) + 1
                slots.append(slot)
        return slots

    def _pop_slot_locked(self) -> Optional[int]:
        if self._free_slots:
            return self._free_slots.popleft()
        for h, slot in list(self._lru.items()):
            if self._pin_counts.get(slot, 0) == 0:
                del self._lru[h]
                self._slot_to_hash.pop(slot, None)
                return slot
        return None

    def commit_slots(self, hashes_hex: Sequence[str], slots: Sequence[int]) -> None:
        """Record loaded/generated blocks in Tier-0 HBM LRU (if cache_slots > 0)."""
        if self.cache_slots <= 0:
            return
        with self._lock:
            for h, slot in zip(hashes_hex, slots):
                old_slot = self._lru.get(h)
                if old_slot is not None and old_slot != slot:
                    # Keep the newer slot; return old slot if unpinned.
                    self._slot_to_hash.pop(old_slot, None)
                    if self._pin_counts.get(old_slot, 0) == 0:
                        self._free_slots.append(old_slot)
                prev_hash = self._slot_to_hash.get(slot)
                if prev_hash is not None and prev_hash != h:
                    self._lru.pop(prev_hash, None)
                self._lru[h] = slot
                self._lru.move_to_end(h)
                self._slot_to_hash[slot] = h

            # Trim LRU down to cache_slots (only unpinned entries).
            while len(self._lru) > self.cache_slots:
                evicted = False
                for h, slot in list(self._lru.items()):
                    if self._pin_counts.get(slot, 0) == 0:
                        del self._lru[h]
                        self._slot_to_hash.pop(slot, None)
                        self._free_slots.append(slot)
                        evicted = True
                        break
                if not evicted:
                    break

    def release_slots(self, slots: Sequence[int]) -> None:
        with self._lock:
            for s in slots:
                self._dec_pin_locked(s)

    def _dec_pin_locked(self, slot: int) -> None:
        cnt = self._pin_counts.get(slot, 0)
        if cnt <= 1:
            self._pin_counts.pop(slot, None)
            if slot not in self._slot_to_hash and slot not in self._free_slots:
                self._free_slots.append(slot)
        else:
            self._pin_counts[slot] = cnt - 1


class TPUDeviceKVPool:
    """Real JAX sharded KV-cache arrays on local TPU devices + verification helpers."""

    def __init__(self, geometry: KVGeometry):
        import jax
        import jax.numpy as jnp
        import numpy as np

        self.geometry = geometry
        devices = jax.local_devices(backend="tpu")
        if not devices:
            raise RuntimeError("No local TPU devices visible to JAX")
        # Sort by NUMA node (matches tpu-sync convention for NUMA-local DMA).
        self.devices = sorted(
            devices, key=lambda d: getattr(d, "numa_node", -1)
        )
        self.num_devices = len(self.devices)

        dtype_map = {
            "bf16": jnp.bfloat16,
            "bfloat16": jnp.bfloat16,
            "fp32": jnp.float32,
            "float32": jnp.float32,
            "fp8": jnp.float8_e4m3fn,
            "float8_e4m3fn": jnp.float8_e4m3fn,
        }
        self.jnp_dtype = dtype_map[geometry.dtype]
        self.np_dtype = np.float32 if geometry.dtype in ("bf16", "bfloat16", "fp8", "float8_e4m3fn") else np.float32

        axis_shapes = (1, self.num_devices)
        mesh = jax.sharding.Mesh(
            np.array(self.devices).reshape(axis_shapes), ("data", "model")
        )
        self.sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, None, "model", None, None)
        )
        shape = geometry.layer_shape()

        # Allocate sharded arrays directly per device to avoid giant host allocations.
        shard_shape = list(shape)
        if shape[2] % self.num_devices != 0:
            raise ValueError(
                f"head_shards={shape[2]} not divisible by {self.num_devices} TPU devices"
            )
        shard_shape[2] = shape[2] // self.num_devices

        self.kv_caches = []
        for _ in range(geometry.num_layers):
            per_dev = []
            for dev in self.devices:
                zeros = jnp.zeros(tuple(shard_shape), dtype=self.jnp_dtype)
                per_dev.append(jax.device_put(zeros, jax.sharding.SingleDeviceSharding(dev)))
            arr = jax.make_array_from_single_device_arrays(shape, self.sharding, per_dev)
            self.kv_caches.append(arr)
        jax.block_until_ready(self.kv_caches)

        # Compiled helpers for fast device-side stamping and re-read verification.
        # Note: reading via jax.jit defeats jax.device_get host memoization so
        # DMA writes landed behind JAX's back are genuinely observed.
        @jax.jit
        def _read_slot_sample(arr, slot_idx):
            # Read first 8 elements across all head shards at token 0.
            return arr[slot_idx, 0, :, 0, 0].astype(jnp.float32)

        @jax.jit
        def _write_slot_tag(arr, slot_idx, tag_val):
            return arr.at[slot_idx, 0, :, 0, 0].set(tag_val.astype(arr.dtype))

        self._read_slot_sample = _read_slot_sample
        self._write_slot_tag = _write_slot_tag
        self._jax_lock = threading.Lock()
        self._jax = jax
        self._jnp = jnp

    def stamp_slots_for_verify(
        self, slots: Sequence[int], hashes_hex: Sequence[str]
    ) -> None:
        """Stamp layer 0 of each slot with a deterministic hash tag in HBM.

        NOTE: Only call this when buffers are NOT simultaneously registered as
        immutable handles in JAX (for verification, we stamp via adapter/DMA or
        before export). For in-place verification without reallocating JAX arrays,
        see `stamp_via_adapter` / `verify_slots`.
        """
        # Updating a jax.Array with .at[].set() produces a NEW buffer handle;
        # we NEVER replace self.kv_caches after registration! Instead, we keep
        # a single small host staging buffer for verification stamps when needed,
        # OR verify initial deterministic contents written BEFORE registration.
        raise NotImplementedError("Use initialize_deterministic_pattern() before registration")

    def initialize_deterministic_pattern(self) -> None:
        """Fill each HBM slot with a distinct non-zero pattern before export/registration.

        Slot `s` in layer `l` gets value `((s % 64) + 1) + 0.25 * (l % 4)` at its
        sentinel lane, allowing lightweight non-zero data checks without reallocating
        registered PJRT buffers.
        """
        jax = self._jax
        jnp = self._jnp
        import numpy as np

        shape = self.geometry.layer_shape()
        shard_shape = list(shape)
        shard_shape[2] = shape[2] // self.num_devices

        slot_vals = (np.arange(shape[0], dtype=np.float32) % 64 + 1.0).reshape(
            shape[0], 1, 1, 1, 1
        )
        new_caches = []
        for layer_idx in range(self.geometry.num_layers):
            per_dev = []
            for dev_idx, dev in enumerate(self.devices):
                base = np.broadcast_to(
                    slot_vals + float((layer_idx % 4) * 0.25 + (dev_idx % 4) * 0.0625),
                    tuple(shard_shape),
                ).copy()
                arr_dev = jax.device_put(
                    jnp.asarray(base, dtype=self.jnp_dtype),
                    jax.sharding.SingleDeviceSharding(dev),
                )
                per_dev.append(arr_dev)
            new_caches.append(
                jax.make_array_from_single_device_arrays(shape, self.sharding, per_dev)
            )
        jax.block_until_ready(new_caches)
        self.kv_caches = new_caches

    def read_slot_checksum(self, slot_idx: int) -> List[float]:
        """Force a true HBM read of slot_idx's sentinel lane across shards."""
        import numpy as np

        with self._jax_lock:
            vals = self._read_slot_sample(self.kv_caches[0], int(slot_idx))
            vals.block_until_ready()
            return [float(x) for x in np.asarray(vals)]


class ComputeOverlapDriver:
    """Runs steady XLA matmuls on TPU devices to measure transfer-induced jitter."""

    def __init__(self, devices: Sequence[Any], mat_dim: int = 2048):
        import jax
        import jax.numpy as jnp
        import numpy as np

        self._jax = jax
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline_ms: List[float] = []
        self.active_ms: List[float] = []

        dev = devices[0]
        rng = np.random.default_rng(7)
        a = jax.device_put(
            jnp.asarray(rng.standard_normal((mat_dim, mat_dim)), dtype=jnp.bfloat16),
            dev,
        )
        b = jax.device_put(
            jnp.asarray(rng.standard_normal((mat_dim, mat_dim)), dtype=jnp.bfloat16),
            dev,
        )
        jax.block_until_ready((a, b))
        self._a = a
        self._b = b

        @jax.jit
        def _step(x, y):
            return jnp.matmul(x, y)

        self._step = _step
        # Warm up XLA compilation.
        for _ in range(3):
            self._step(self._a, self._b).block_until_ready()

    def measure_baseline(self, steps: int = 30) -> List[float]:
        samples: List[float] = []
        for _ in range(steps):
            t0 = time.perf_counter()
            self._step(self._a, self._b).block_until_ready()
            samples.append((time.perf_counter() - t0) * 1000.0)
        self.baseline_ms = samples
        return samples

    def start(self) -> None:
        self._stop.clear()
        self.active_ms = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self._step(self._a, self._b).block_until_ready()
            self.active_ms.append((time.perf_counter() - t0) * 1000.0)
            # Small yield so Python transfer threads aren't GIL-starved.
            time.sleep(0.001)

    def stop(self) -> List[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        return self.active_ms


class KVStoreBackend:
    """Common interface implemented by Mooncake, tpu-sync, and Mock backends."""

    name: str = "base"

    def lookup(self, hashes_hex: Sequence[str]) -> LookupResult:
        raise NotImplementedError

    def load(self, hits: Sequence[LookupHit], hbm_slots: Sequence[int]) -> TransferResult:
        raise NotImplementedError

    def save(self, hashes_hex: Sequence[str], hbm_slots: Sequence[int]) -> TransferResult:
        raise NotImplementedError

    def release_hits(self, hits: Sequence[LookupHit]) -> None:
        """Release any backend pins held by lookup hits that were not loaded."""

    def verify_slot_nonzero(self, hbm_slot: int) -> bool:
        return True

    def memory_accounting(self) -> Tuple[int, int]:
        """Returns (allocated_host_bytes, nominal_host_bytes)."""
        return (0, 0)

    def close(self) -> None:
        pass


class MooncakeStoreBackend(KVStoreBackend):
    """Mooncake Distributed Store over TENT TPU path (`tpu-6-store` + `mooncake.tpu`)."""

    name = "mooncake"

    def __init__(
        self,
        *,
        geometry: KVGeometry,
        node_id: int,
        my_ip: str,
        master_addr: str,
        metadata_url: str,
        protocol: str = "hp_tcp",
        connections: Optional[int] = 8,
        replicas: int = 1,
        prefer_local: bool = True,
        segment_headroom: float = 1.15,
        local_buffer_mb: int = 64,
        adapter_path: Optional[str] = None,
    ):
        self.geometry = geometry
        self.node_id = node_id
        self.my_ip = my_ip
        self.replicas = replicas
        self.prefer_local = prefer_local

        # 1. Allocate real JAX sharded TPU KV-cache arrays FIRST (jax before mooncake!).
        self.tpu_pool = TPUDeviceKVPool(geometry)
        self.tpu_pool.initialize_deterministic_pattern()
        self.num_devices = self.tpu_pool.num_devices
        self.block_shard_bytes = geometry.block_shard_bytes(self.num_devices)
        self.block_total_bytes = geometry.block_total_bytes()

        # 2. Configure TENT before importing/initializing Mooncake Store.
        os.environ["MC_USE_TENT"] = "1"
        os.environ.setdefault("MC_STORE_MEMCPY", "0")
        hp_tcp: Dict[str, Any] = {"enable": protocol == "hp_tcp"}
        if protocol == "hp_tcp" and connections is not None:
            hp_tcp["connections_per_peer"] = int(connections)
        tent_conf = {
            "log_level": "warning",
            "rpc_server_hostname": my_ip,
            "transports": {
                "tcp": {"enable": protocol == "tcp"},
                "hp_tcp": hp_tcp,
                "rdma": {"enable": False},
                "shm": {"enable": False},
                "io_uring": {"enable": False},
                "tpu": {"enable": True},
            },
        }
        os.environ["MC_TENT_CONF"] = json.dumps(tent_conf)

        from mooncake import tpu as mc_tpu
        import mooncake.store as mc_store

        self._mc_tpu = mc_tpu
        self._mc_store = mc_store

        # 3. Export TPU shards to the TPU adapter BEFORE store.setup(), because
        #    TENT inspects TPU device topology once at engine construction.
        adapter = mc_tpu.get_adapter(adapter_path) if adapter_path else mc_tpu.get_adapter()
        self.shards = mc_tpu.prepare(self.tpu_pool.kv_caches, adapter=adapter)
        expected_shard_total = geometry.hbm_blocks * self.block_shard_bytes
        for layer_idx, layer_shards in enumerate(self.shards):
            if len(layer_shards) != self.num_devices:
                raise RuntimeError(
                    f"Layer {layer_idx}: expected {self.num_devices} TPU shards, "
                    f"got {len(layer_shards)}"
                )
            for dev_idx, sh in enumerate(layer_shards):
                if sh.size != expected_shard_total:
                    raise RuntimeError(
                        f"Layer {layer_idx} dev {dev_idx}: shard.size={sh.size} != "
                        f"expected {expected_shard_total} (XLA padding mismatch)"
                    )

        # 4. Initialize MooncakeDistributedStore.
        #    Note: RealClient::IsHostStoreSegmentProtocol accepts "tcp" (not "hp_tcp");
        #    TENT selects hp_tcp via MC_TENT_CONF above.
        self.client_port = pick_free_port(my_ip)
        self.local_hostname = f"{my_ip}:{self.client_port}"
        self.nominal_host_bytes = geometry.host_pool_bytes()
        # Round global segment size up to 64 MiB multiple with modest allocator headroom.
        raw_seg = int(self.nominal_host_bytes * segment_headroom)
        align_64m = 64 * 1024 * 1024
        self.allocated_host_bytes = max(
            align_64m, ((raw_seg + align_64m - 1) // align_64m) * align_64m
        )
        local_buf_bytes = max(16 * 1024 * 1024, local_buffer_mb * 1024 * 1024)

        self.store = mc_store.MooncakeDistributedStore()
        rc = self.store.setup(
            self.local_hostname,
            metadata_url,
            self.allocated_host_bytes,
            local_buf_bytes,
            "tcp",
            "",
            master_addr,
        )
        if rc != 0:
            mc_tpu.unregister([s for layer in self.shards for s in layer])
            raise RuntimeError(
                f"MooncakeDistributedStore.setup failed (rc={rc}) for "
                f"local={self.local_hostname} master={master_addr} meta={metadata_url}"
            )

        # 5. Register every TPU shard fake address with the Store's TENT engine.
        self._registered_addrs: List[int] = []
        for layer_shards in self.shards:
            for sh in layer_shards:
                rc = self.store.register_buffer(sh.fake_addr, sh.size)
                if rc != 0:
                    self.close()
                    raise RuntimeError(
                        f"store.register_buffer failed (rc={rc}) for fake_addr={sh.fake_addr:#x}"
                    )
                self._registered_addrs.append(sh.fake_addr)

        self._rep_config = mc_store.ReplicateConfig()
        self._rep_config.replica_num = int(replicas)
        self._rep_config.prefer_alloc_in_same_node = bool(prefer_local)
        # Exact segment endpoint (ip:rpc_port) so local vs remote DRAM classification
        # is accurate on both multi-host CDK and single-VM loopback tests.
        store_ep = ""
        try:
            store_ep = self.store.get_hostname() or ""
        except Exception:
            store_ep = ""
        self._local_endpoints = {ep for ep in (self.local_hostname, store_ep) if ep}

    def _slot_buffers(self, hbm_slot: int) -> Tuple[List[int], List[int]]:
        offset = int(hbm_slot) * self.block_shard_bytes
        ptrs: List[int] = []
        sizes: List[int] = []
        for layer_shards in self.shards:
            for sh in layer_shards:
                ptrs.append(sh.fake_addr + offset)
                sizes.append(self.block_shard_bytes)
        return ptrs, sizes

    def _classify_replicas(self, replicas: Sequence[Any]) -> Optional[str]:
        has_remote_mem = False
        has_storage = False
        for rep in replicas:
            if rep.is_memory_replica():
                ep = rep.get_memory_descriptor().buffer_descriptor.transport_endpoint
                if ep in self._local_endpoints:
                    return "local_dram"
                has_remote_mem = True
            elif rep.is_disk_replica() or rep.is_local_disk_replica():
                has_storage = True
        if has_remote_mem:
            return "remote_dram"
        if has_storage:
            return "storage"
        return None

    def lookup(self, hashes_hex: Sequence[str]) -> LookupResult:
        if not hashes_hex:
            return LookupResult(hits=[], raw_hit_count=0, latency_ms=0.0)
        keys = list(hashes_hex)
        t0 = time.perf_counter()
        desc_map = self.store.batch_get_replica_desc(keys)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hits: List[LookupHit] = []
        raw_hits = 0
        prefix_open = True
        for k in keys:
            reps = desc_map.get(k)
            tier = self._classify_replicas(reps) if reps else None
            if tier is not None:
                raw_hits += 1
                if prefix_open:
                    hits.append(LookupHit(hash_hex=k, tier=tier))
            else:
                prefix_open = False
        return LookupResult(hits=hits, raw_hit_count=raw_hits, latency_ms=elapsed_ms)

    def load(self, hits: Sequence[LookupHit], hbm_slots: Sequence[int]) -> TransferResult:
        if not hits:
            return TransferResult(0, 0, 0, 0, 0, 0.0)
        keys = [h.hash_hex for h in hits]
        all_ptrs: List[List[int]] = []
        all_sizes: List[List[int]] = []
        for slot in hbm_slots:
            ptrs, sizes = self._slot_buffers(slot)
            all_ptrs.append(ptrs)
            all_sizes.append(sizes)

        t0 = time.perf_counter()
        rcs = self.store.batch_get_into_multi_buffers(
            keys, all_ptrs, all_sizes, prefer_alloc_in_same_node=self.prefer_local
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ok_indices: List[int] = []
        failed = 0
        bytes_ok = 0
        remote_bytes = 0
        for idx, (rc, hit) in enumerate(zip(rcs, hits)):
            if rc >= 0:
                ok_indices.append(idx)
                bytes_ok += self.block_total_bytes
                if hit.tier == "remote_dram":
                    remote_bytes += self.block_total_bytes
            else:
                failed += 1
        return TransferResult(
            ok_count=len(ok_indices),
            failed_count=failed,
            rejected_count=0,
            bytes_transferred=bytes_ok,
            remote_bytes=remote_bytes,
            latency_ms=elapsed_ms,
            ok_indices=ok_indices,
        )

    def save(self, hashes_hex: Sequence[str], hbm_slots: Sequence[int]) -> TransferResult:
        if not hashes_hex:
            return TransferResult(0, 0, 0, 0, 0, 0.0)
        keys = list(hashes_hex)
        all_ptrs: List[List[int]] = []
        all_sizes: List[List[int]] = []
        for slot in hbm_slots:
            ptrs, sizes = self._slot_buffers(slot)
            all_ptrs.append(ptrs)
            all_sizes.append(sizes)

        t0 = time.perf_counter()
        rcs = self.store.batch_put_from_multi_buffers(
            keys, all_ptrs, all_sizes, self._rep_config
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ok_indices: List[int] = []
        rejected = 0
        failed = 0
        bytes_ok = 0
        for idx, rc in enumerate(rcs):
            if rc == 0:
                ok_indices.append(idx)
                bytes_ok += self.block_total_bytes
            elif rc == MC_OBJECT_ALREADY_EXISTS:
                # Content-addressed duplicate already present in cluster.
                ok_indices.append(idx)
            elif rc == MC_NO_AVAILABLE_HANDLE:
                rejected += 1
            else:
                failed += 1
        return TransferResult(
            ok_count=len(ok_indices),
            failed_count=failed,
            rejected_count=rejected,
            bytes_transferred=bytes_ok,
            remote_bytes=0,
            latency_ms=elapsed_ms,
            ok_indices=ok_indices,
        )

    def verify_slot_nonzero(self, hbm_slot: int) -> bool:
        vals = self.tpu_pool.read_slot_checksum(hbm_slot)
        return any(abs(v) > 1e-6 for v in vals)

    def memory_accounting(self) -> Tuple[int, int]:
        return (self.allocated_host_bytes, self.nominal_host_bytes)

    def close(self) -> None:
        if getattr(self, "store", None) is not None:
            for addr in reversed(self._registered_addrs):
                try:
                    self.store.unregister_buffer(addr)
                except Exception:
                    pass
            self._registered_addrs.clear()
            try:
                self.store.close()
            except Exception:
                pass
            self.store = None
        if getattr(self, "shards", None):
            try:
                self._mc_tpu.unregister([s for layer in self.shards for s in layer])
            except Exception:
                pass
            self.shards = []


class TpuSyncStoreBackend(KVStoreBackend):
    """tpu-sync (Raiden) KVCacheStore + KVCacheManager over real TPU HBM."""

    name = "tpu_sync"

    def __init__(
        self,
        *,
        geometry: KVGeometry,
        node_id: int,
        my_ip: str,
        global_registry_addr: str,
        parallelism: int = 8,
        enable_global: bool = True,
        enable_evict_sweep: bool = True,
        evict_low_watermark: float = 0.15,
        evict_high_watermark: float = 0.30,
        evict_sweep_period_s: int = 1,
        enable_shm: bool = False,
        shm_key: str = "",
        shm_model_uid: str = "kv_store_bench_v1",
        kv_pool_group: str = "kv_bench_pool",
        staging_blocks: int = 24,
        poll_timeout_s: float = 60.0,
    ):
        self.geometry = geometry
        self.node_id = node_id
        self.my_ip = my_ip
        self.enable_global = enable_global
        self.poll_timeout_s = poll_timeout_s

        # 1. Allocate JAX sharded TPU arrays BEFORE importing tpu_sync C++ extension.
        self.tpu_pool = TPUDeviceKVPool(geometry)
        self.tpu_pool.initialize_deterministic_pattern()
        self.num_devices = self.tpu_pool.num_devices
        self.block_shard_bytes = geometry.block_shard_bytes(self.num_devices)
        self.block_total_bytes = geometry.block_total_bytes()

        # 2. Configure tpu-sync environment knobs (read once at KVCacheStore init).
        os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"
        os.environ.setdefault("ENABLE_MULTI_NUMA", "0")
        if global_registry_addr and enable_evict_sweep:
            os.environ["RAIDEN_ENABLE_STORE_MONITOR"] = "true"
            os.environ["RAIDEN_ENABLE_EVICT_SWEEP"] = "true"
            os.environ["RAIDEN_STORE_MONITOR_HEARTBEAT_S"] = "1"
            os.environ["RAIDEN_EVICT_SWEEP_PERIOD_S"] = str(max(1, int(evict_sweep_period_s)))
            os.environ["RAIDEN_EVICT_LOW_WATERMARK"] = str(evict_low_watermark)
            os.environ["RAIDEN_EVICT_HIGH_WATERMARK"] = str(evict_high_watermark)
        else:
            os.environ["RAIDEN_ENABLE_STORE_MONITOR"] = "false"
            os.environ["RAIDEN_ENABLE_EVICT_SWEEP"] = "false"

        if enable_shm and shm_key:
            os.environ["RAIDEN_SHM_KEY"] = shm_key
            os.environ["RAIDEN_SHM_MODEL_UID"] = shm_model_uid
            os.environ["RAIDEN_SHM_SERVER_NAME"] = f"node_{node_id}"
        else:
            os.environ.pop("RAIDEN_SHM_KEY", None)
            os.environ.pop("RAIDEN_SHM_MODEL_UID", None)
            os.environ.pop("RAIDEN_SHM_SERVER_NAME", None)

        from tpu_sync.api.jax import kv_cache_manager
        from tpu_sync.api.jax import kv_cache_store

        self._kcm = kv_cache_manager
        self._kcs = kv_cache_store

        # 3. Create KVCacheStore (controller) + KVCacheManager (worker) with barrier.
        #    Note: worker node_id=0 on every single-host replica so cross-replica
        #    ReadRemote matches worker ranks 0 <-> 0 (strict node_id matching).
        self.controller_port = pick_free_port(my_ip)
        self.worker_port = pick_free_port(my_ip)
        self.rid = kv_cache_store.RaidenId(
            f"kvbench_node_{node_id}", "0", "kv_bench", 0
        )

        self.host_blocks = geometry.host_blocks
        self.total_host_blocks_alloc = geometry.host_blocks + max(0, staging_blocks)
        self.nominal_host_bytes = geometry.host_pool_bytes()
        self.allocated_host_bytes = self.total_host_blocks_alloc * self.block_total_bytes

        built: Dict[str, Any] = {}
        err_box: List[BaseException] = []

        def _start_worker():
            try:
                time.sleep(0.2)
                built["manager"] = kv_cache_manager.KVCacheManager(
                    kv_caches=self.tpu_pool.kv_caches,
                    local_control_port=0,
                    host_blocks_to_allocate=self.total_host_blocks_alloc,
                    unsafe_skip_buffer_lock=True,
                    parallelism=parallelism,
                    raiden_worker_port=self.worker_port,
                    raiden_controller_address=f"{my_ip}:{self.controller_port}",
                    worker_id=f"worker_{node_id}",
                    node_id=0,
                    enable_shm=enable_shm,
                )
            except BaseException as exc:  # noqa: BLE001
                err_box.append(exc)

        t = threading.Thread(target=_start_worker, daemon=True)
        t.start()
        try:
            self.store = kv_cache_store.KVCacheStore(
                capacity=self.host_blocks,
                global_registry_address=global_registry_addr,
                raiden_id=self.rid,
                num_shards=self.num_devices,
                shard_size_bytes=self.block_shard_bytes,
                store_server_ip=my_ip,
                raiden_controller_port=self.controller_port,
                expected_worker_count=1,
                kv_pool_group=kv_pool_group,
            )
        finally:
            t.join(timeout=180)
        if err_box:
            raise RuntimeError(f"KVCacheManager failed to start: {err_box[0]}") from err_box[0]
        if "manager" not in built:
            raise RuntimeError("KVCacheManager worker thread timed out")
        self.manager = built["manager"]
        # Serialize poll/submit calls cleanly if multiple Python threads share the store.
        self._op_lock = threading.Lock()

    def _poll_hashes(
        self,
        poll_fn,
        expected: set[bytes],
    ) -> Tuple[set[bytes], set[bytes]]:
        done_set: set[bytes] = set()
        failed_set: set[bytes] = set()
        deadline = time.perf_counter() + self.poll_timeout_s
        sleep_s = 0.0005
        while time.perf_counter() < deadline:
            res = poll_fn()
            done, failed = res[0], res[1]
            for h in done:
                if h in expected:
                    done_set.add(h)
            for h in failed:
                if h in expected:
                    failed_set.add(h)
            if len(done_set) + len(failed_set) >= len(expected):
                return done_set, failed_set
            time.sleep(sleep_s)
            sleep_s = min(0.002, sleep_s * 1.5)
        # Remaining hashes timed out -> mark failed.
        remaining = expected - done_set - failed_set
        failed_set.update(remaining)
        return done_set, failed_set

    def lookup(self, hashes_hex: Sequence[str]) -> LookupResult:
        if not hashes_hex:
            return LookupResult(hits=[], raw_hit_count=0, latency_ms=0.0)
        raw_hashes = [bytes.fromhex(h) for h in hashes_hex]
        status_enum = self._kcs.BlockStatus
        t0 = time.perf_counter()
        with self._op_lock:
            res = self.store.lookup(
                raw_hashes, enable_global=self.enable_global, pin_found=True
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hits: List[LookupHit] = []
        to_release: List[bytes] = []
        for raw_h, blk in res:
            h_hex = raw_h.hex()
            st = blk.status
            if st in (status_enum.HOST, status_enum.HOST_AND_HBM):
                hits.append(LookupHit(hash_hex=h_hex, tier="local_dram", raw_handle=blk, pinned=True))
            elif st == status_enum.REMOTE:
                hits.append(LookupHit(hash_hex=h_hex, tier="remote_dram", raw_handle=blk, pinned=False))
            elif st == status_enum.SHARED_STORAGE:
                hits.append(LookupHit(hash_hex=h_hex, tier="storage", raw_handle=blk, pinned=False))
            else:
                # HBM-only entry isn't host-resident; release its pin and halt prefix.
                to_release.append(raw_h)
                break
        if to_release:
            with self._op_lock:
                self.store.release(to_release)
        # Note: tpu-sync's lookup() always halts at first miss, so raw_hit_count == len(hits).
        return LookupResult(hits=hits, raw_hit_count=len(hits), latency_ms=elapsed_ms)

    def release_hits(self, hits: Sequence[LookupHit]) -> None:
        pinned = [bytes.fromhex(h.hash_hex) for h in hits if h.pinned]
        if pinned:
            with self._op_lock:
                self.store.release(pinned)

    def load(self, hits: Sequence[LookupHit], hbm_slots: Sequence[int]) -> TransferResult:
        if not hits:
            return TransferResult(0, 0, 0, 0, 0, 0.0)
        local_items: List[Tuple[int, bytes, int, Any]] = []
        remote_by_peer: Dict[Tuple[str, str], List[Tuple[int, bytes, int, Any]]] = {}
        storage_items: List[Tuple[int, bytes, int, Any]] = []

        for idx, (hit, slot) in enumerate(zip(hits, hbm_slots)):
            raw_h = bytes.fromhex(hit.hash_hex)
            if hit.tier == "local_dram":
                local_items.append((idx, raw_h, int(slot), hit.raw_handle))
            elif hit.tier == "remote_dram":
                rid = hit.raw_handle.raiden_id
                peer_key = (str(rid.job_name), str(rid.job_replica_id))
                remote_by_peer.setdefault(peer_key, []).append(
                    (idx, raw_h, int(slot), hit.raw_handle)
                )
            else:
                storage_items.append((idx, raw_h, int(slot), hit.raw_handle))

        t0 = time.perf_counter()
        launched_expected: set[bytes] = set()
        hash_to_idx: Dict[bytes, int] = {}
        hash_is_local: Dict[bytes, bool] = {}
        launch_failed_local: List[bytes] = []
        failed_count = 0

        with self._op_lock:
            if local_items:
                l_hashes = [x[1] for x in local_items]
                l_slots = [x[2] for x in local_items]
                ok = self.store.load(l_hashes, l_slots)
                if ok:
                    for idx, rh, _, _ in local_items:
                        launched_expected.add(rh)
                        hash_to_idx[rh] = idx
                        hash_is_local[rh] = True
                else:
                    launch_failed_local.extend(l_hashes)
                    failed_count += len(local_items)

            # read_remote requires one owning peer per call.
            for peer_items in remote_by_peer.values():
                r_hashes = [x[1] for x in peer_items]
                r_slices = [x[3] for x in peer_items]
                r_slots = [x[2] for x in peer_items]
                ok = self.store.read_remote(r_hashes, r_slices, r_slots)
                if ok:
                    for idx, rh, _, _ in peer_items:
                        launched_expected.add(rh)
                        hash_to_idx[rh] = idx
                        hash_is_local[rh] = False
                else:
                    failed_count += len(peer_items)

            if storage_items:
                s_hashes = [x[1] for x in storage_items]
                s_slices = [x[3] for x in storage_items]
                s_slots = [x[2] for x in storage_items]
                ok = self.store.read_remote(s_hashes, s_slices, s_slots)
                if ok:
                    for idx, rh, _, _ in storage_items:
                        launched_expected.add(rh)
                        hash_to_idx[rh] = idx
                        hash_is_local[rh] = False
                else:
                    failed_count += len(storage_items)

            done_set: set[bytes] = set()
            poll_failed: set[bytes] = set()
            if launched_expected:
                done_set, poll_failed = self._poll_hashes(
                    self.store.poll_load_status, launched_expected
                )

            # Per tpu-sync pin contract: failed local loads did not consume their pin.
            pins_to_release = list(launch_failed_local) + [
                rh for rh in poll_failed if hash_is_local.get(rh, False)
            ]
            if pins_to_release:
                self.store.release(pins_to_release)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        failed_count += len(poll_failed)

        ok_indices = sorted(hash_to_idx[rh] for rh in done_set)
        bytes_ok = len(ok_indices) * self.block_total_bytes
        remote_bytes = sum(
            self.block_total_bytes
            for idx in ok_indices
            if hits[idx].tier == "remote_dram"
        )
        return TransferResult(
            ok_count=len(ok_indices),
            failed_count=failed_count,
            rejected_count=0,
            bytes_transferred=bytes_ok,
            remote_bytes=remote_bytes,
            latency_ms=elapsed_ms,
            ok_indices=ok_indices,
        )

    def save(self, hashes_hex: Sequence[str], hbm_slots: Sequence[int]) -> TransferResult:
        if not hashes_hex:
            return TransferResult(0, 0, 0, 0, 0, 0.0)
        status_enum = self._kcs.BlockStatus
        raw_hashes = [bytes.fromhex(h) for h in hashes_hex]
        slices = [
            self._kcs.RaidenBlockId(
                self.rid,
                host_block_id=-1,
                device_block_id=int(slot),
                status=status_enum.HBM,
            )
            for slot in hbm_slots
        ]

        t0 = time.perf_counter()
        inserted_indices: List[int] = []
        rejected = 0
        failed = 0

        with self._op_lock:
            # First try full batch insert (tpu-sync's all-or-nothing contract).
            if self.store.insert(raw_hashes, slices, on_host=False):
                inserted_indices = list(range(len(raw_hashes)))
            else:
                # If the full batch was rejected (e.g. cache nearly full), insert as
                # long a contiguous prefix as fits so partial prefix progress is fair
                # while still counting every rejected block accurately.
                for idx in range(len(raw_hashes)):
                    if self.store.insert([raw_hashes[idx]], [slices[idx]], on_host=False):
                        inserted_indices.append(idx)
                    else:
                        rejected = len(raw_hashes) - len(inserted_indices)
                        break

            ok_indices: List[int] = []
            if inserted_indices:
                ins_hashes = [raw_hashes[i] for i in inserted_indices]
                launched = self.store.save(ins_hashes)
                if not launched:
                    self.store.release(ins_hashes)
                    failed += len(ins_hashes)
                else:
                    expected = set(ins_hashes)
                    done_set, failed_set = self._poll_hashes(
                        self.store.poll_save_status, expected
                    )
                    if failed_set:
                        self.store.release(list(failed_set))
                        failed += len(failed_set)
                    hash_to_idx = {raw_hashes[i]: i for i in inserted_indices}
                    ok_indices = sorted(hash_to_idx[h] for h in done_set)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return TransferResult(
            ok_count=len(ok_indices),
            failed_count=failed,
            rejected_count=rejected,
            bytes_transferred=len(ok_indices) * self.block_total_bytes,
            remote_bytes=0,
            latency_ms=elapsed_ms,
            ok_indices=ok_indices,
        )

    def verify_slot_nonzero(self, hbm_slot: int) -> bool:
        vals = self.tpu_pool.read_slot_checksum(hbm_slot)
        return any(abs(v) > 1e-6 for v in vals)

    def memory_accounting(self) -> Tuple[int, int]:
        return (self.allocated_host_bytes, self.nominal_host_bytes)

    def close(self) -> None:
        if getattr(self, "manager", None) is not None:
            del self.manager
            self.manager = None
        if getattr(self, "store", None) is not None:
            del self.store
            self.store = None


class MockClusterRegistry:
    """In-process multi-node mock registry for fast CPU-only tests of Tier 2."""

    def __init__(self):
        self.lock = threading.Lock()
        # node_id -> OrderedDict[hash_hex, int_size]
        self.node_caches: Dict[int, collections.OrderedDict[str, int]] = {}
        self.node_caps: Dict[int, int] = {}

    def register(self, node_id: int, capacity: int, preserve_existing: bool = False) -> None:
        with self.lock:
            self.node_caps[node_id] = capacity
            if not preserve_existing or node_id not in self.node_caches:
                self.node_caches[node_id] = collections.OrderedDict()

    def clear_node(self, node_id: int) -> None:
        with self.lock:
            if node_id in self.node_caches:
                self.node_caches[node_id].clear()


class MockStoreBackend(KVStoreBackend):
    """Simulated Mooncake or tpu-sync store for CPU-only verification of the harness."""

    def __init__(
        self,
        *,
        mode: str,  # "mooncake" or "tpu_sync"
        geometry: KVGeometry,
        node_id: int,
        registry: MockClusterRegistry,
        enable_shm: bool = False,
    ):
        self.name = mode
        self.mode = mode
        self.geometry = geometry
        self.node_id = node_id
        self.registry = registry
        self.block_total_bytes = geometry.block_total_bytes()
        self.nominal_host_bytes = geometry.host_pool_bytes()
        self.allocated_host_bytes = (
            int(self.nominal_host_bytes * 1.15)
            if mode == "mooncake"
            else (geometry.host_blocks + 24) * self.block_total_bytes
        )
        registry.register(
            node_id,
            geometry.host_blocks,
            preserve_existing=(mode == "tpu_sync" and enable_shm),
        )

    def lookup(self, hashes_hex: Sequence[str]) -> LookupResult:
        t0 = time.perf_counter()
        time.sleep(0.0002)
        hits: List[LookupHit] = []
        raw_hits = 0
        prefix_open = True
        with self.registry.lock:
            my_cache = self.registry.node_caches[self.node_id]
            for h in hashes_hex:
                tier = None
                if h in my_cache:
                    my_cache.move_to_end(h)
                    tier = "local_dram"
                else:
                    for peer_id, peer_cache in self.registry.node_caches.items():
                        if peer_id != self.node_id and h in peer_cache:
                            tier = "remote_dram"
                            break
                if tier is not None:
                    raw_hits += 1
                    if prefix_open:
                        hits.append(LookupHit(hash_hex=h, tier=tier))
                else:
                    prefix_open = False
        return LookupResult(
            hits=hits,
            raw_hit_count=raw_hits,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def load(self, hits: Sequence[LookupHit], hbm_slots: Sequence[int]) -> TransferResult:
        t0 = time.perf_counter()
        remote = sum(1 for h in hits if h.tier == "remote_dram")
        local = len(hits) - remote
        # Simulate local DMA faster than remote NIC hop.
        time.sleep(local * 0.0002 + remote * 0.0006)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return TransferResult(
            ok_count=len(hits),
            failed_count=0,
            rejected_count=0,
            bytes_transferred=len(hits) * self.block_total_bytes,
            remote_bytes=remote * self.block_total_bytes,
            latency_ms=elapsed_ms,
            ok_indices=list(range(len(hits))),
        )

    def save(self, hashes_hex: Sequence[str], hbm_slots: Sequence[int]) -> TransferResult:
        t0 = time.perf_counter()
        time.sleep(len(hashes_hex) * 0.0002)
        ok_indices: List[int] = []
        rejected = 0
        with self.registry.lock:
            my_cache = self.registry.node_caches[self.node_id]
            cap = self.registry.node_caps[self.node_id]
            for idx, h in enumerate(hashes_hex):
                if h in my_cache:
                    my_cache.move_to_end(h)
                    ok_indices.append(idx)
                    continue
                if self.mode == "mooncake":
                    # Cluster pooled DRAM: spill to a peer with free space, else inline LRU evict.
                    if len(my_cache) < cap:
                        my_cache[h] = self.block_total_bytes
                    else:
                        spilled = False
                        for peer_id, peer_cache in self.registry.node_caches.items():
                            if len(peer_cache) < self.registry.node_caps[peer_id]:
                                peer_cache[h] = self.block_total_bytes
                                spilled = True
                                break
                        if not spilled:
                            my_cache.popitem(last=False)
                            my_cache[h] = self.block_total_bytes
                    ok_indices.append(idx)
                else:
                    # tpu-sync: local-only LRU; evicts periodically rather than inline when full.
                    if len(my_cache) < cap:
                        my_cache[h] = self.block_total_bytes
                        ok_indices.append(idx)
                    else:
                        # Simulate periodic background sweep freeing 15% every ~10 misses.
                        if (idx + len(my_cache)) % 7 == 0:
                            for _ in range(max(1, cap // 8)):
                                if my_cache:
                                    my_cache.popitem(last=False)
                            my_cache[h] = self.block_total_bytes
                            ok_indices.append(idx)
                        else:
                            rejected += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return TransferResult(
            ok_count=len(ok_indices),
            failed_count=0,
            rejected_count=rejected,
            bytes_transferred=len(ok_indices) * self.block_total_bytes,
            remote_bytes=0,
            latency_ms=elapsed_ms,
            ok_indices=ok_indices,
        )

    def memory_accounting(self) -> Tuple[int, int]:
        return (self.allocated_host_bytes, self.nominal_host_bytes)
