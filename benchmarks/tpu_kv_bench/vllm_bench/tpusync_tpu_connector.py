#!/usr/bin/env python3
"""tpu-sync (`KVCacheStore`) KV connectors for vLLM V1 on JAX/TPU (`tpu-inference`).

Provides two drop-in vLLM V1 `KVConnectorBase_V1` connectors that mirror
`mooncake_tpu_connector.py` so Tier 3 multi-host CDK serving benchmarks compare
`MooncakeDistributedStore` vs `tpu-sync KVCacheStore` under identical vLLM
connector hooks and host DRAM budgets on any stock `vllm/vllm-tpu` image:

1. `TpuSyncTPUOffloadConnector` (`kv_role="kv_both"`):
   Cluster-wide shared prefix-cache KV offloading across 1..N vLLM-TPU serving
   pods backed by `tpu_sync.api.jax.kv_cache_store.KVCacheStore` +
   `global_registry_server`.

2. `TpuSyncTPUDisaggConnector` (`kv_role="kv_producer"` / `"kv_consumer"`):
   Multi-host disaggregated 1P1D KV transfer compatible with
   `toy_proxy_server.py`, publishing Prefill blocks to `KVCacheStore` and
   pulling them directly into Decode HBM via `read_remote`.

Environment variables:
  TPUSYNC_REGISTRY_ADDR      e.g. "10.0.0.5:50051" (required)
  TPUSYNC_HOST_IP            reachable IP of this pod (auto-detected if unset)
  TPUSYNC_NODE_ID            unique pod index (default: "0")
  TPUSYNC_HOST_CACHE_GB      host DRAM pool per pod in GiB (default: 32)
  TPUSYNC_PARALLELISM        transfer socket parallelism (default: 8)
  TPUSYNC_STAGING_BLOCKS     extra host staging blocks (default: 32)
  TPUSYNC_ENABLE_EVICT_SWEEP "1" (default) or "0"
  TPUSYNC_POOL_GROUP         kv_pool_group name (default: "vllm_tpu_pool")
  TPUSYNC_MODEL_TAG          key namespace prefix (default: model name)
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import os
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

# Import JAX before any tpu_sync C++ extension to avoid XLA runtime abort.
import jax
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

ReqId = str


def _detect_host_ip() -> str:
    env_ip = os.environ.get("TPUSYNC_HOST_IP", "").strip()
    if env_ip:
        return env_ip
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _pick_free_port(bind_ip: str = "0.0.0.0") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((bind_ip, 0))
        return int(s.getsockname()[1])


def _prefix_block_hashes(
    namespace: str, token_ids: Sequence[int], block_size: int
) -> List[bytes]:
    """Deterministic 16-byte prefix-chained hashes for full `block_size` chunks."""
    num_full = len(token_ids) // block_size
    if num_full <= 0:
        return []
    hashes: List[bytes] = []
    acc = namespace.encode("utf-8")
    for b in range(num_full):
        chunk = token_ids[b * block_size : (b + 1) * block_size]
        packed = struct.pack(f">{len(chunk)}I", *(int(t) & 0xFFFFFFFF for t in chunk))
        acc = hashlib.blake2b(acc + packed, digest_size=16).digest()
        hashes.append(acc)
    return hashes


def _disagg_block_hash(uid: int, idx: int) -> bytes:
    return hashlib.blake2b(f"disagg:{uid}:{idx}".encode("utf-8"), digest_size=16).digest()


def _get_uuid_50bit() -> int:
    return uuid4().int >> 78


class _TpuSyncStoreBridge:
    """Shared worker-side bridge owning `KVCacheStore` + `KVCacheManager`."""

    def __init__(self, kv_caches: Sequence[jax.Array], host_cache_gb: float):
        self.my_ip = _detect_host_ip()
        self.node_tag = os.environ.get("TPUSYNC_NODE_ID", "0")
        self.registry_addr = os.environ.get("TPUSYNC_REGISTRY_ADDR", "127.0.0.1:50051")
        parallelism = int(os.environ.get("TPUSYNC_PARALLELISM", "8"))
        staging_blocks = int(os.environ.get("TPUSYNC_STAGING_BLOCKS", "32"))
        pool_group = os.environ.get("TPUSYNC_POOL_GROUP", "vllm_tpu_pool")
        enable_evict = os.environ.get("TPUSYNC_ENABLE_EVICT_SWEEP", "1").lower() in (
            "1",
            "true",
            "yes",
        )

        self.kv_caches = list(kv_caches)
        if not self.kv_caches:
            raise RuntimeError("runner.kv_caches is empty")

        num_blocks = int(self.kv_caches[0].shape[0])
        shards0 = self.kv_caches[0].addressable_shards
        num_shards = len(shards0)
        shard_bytes_per_layer = int(shards0[0].data.nbytes) // num_blocks
        total_block_bytes = sum(
            int(s.data.nbytes) // num_blocks
            for kc in self.kv_caches
            for s in kc.addressable_shards
        )
        host_blocks = max(64, int((host_cache_gb * (1024**3)) // max(1, total_block_bytes)))
        total_alloc_blocks = host_blocks + max(0, staging_blocks)

        os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"
        os.environ.setdefault("ENABLE_MULTI_NUMA", "0")
        if self.registry_addr and enable_evict:
            os.environ["RAIDEN_ENABLE_STORE_MONITOR"] = "true"
            os.environ["RAIDEN_ENABLE_EVICT_SWEEP"] = "true"
            os.environ.setdefault("RAIDEN_STORE_MONITOR_HEARTBEAT_S", "1")
            os.environ.setdefault("RAIDEN_EVICT_SWEEP_PERIOD_S", "1")
            os.environ.setdefault("RAIDEN_EVICT_LOW_WATERMARK", "0.15")
            os.environ.setdefault("RAIDEN_EVICT_HIGH_WATERMARK", "0.30")
        else:
            os.environ["RAIDEN_ENABLE_STORE_MONITOR"] = "false"
            os.environ["RAIDEN_ENABLE_EVICT_SWEEP"] = "false"

        from tpu_sync.api.jax import kv_cache_manager
        from tpu_sync.api.jax import kv_cache_store

        self._kcs = kv_cache_store
        self.controller_port = _pick_free_port(self.my_ip)
        self.worker_port = _pick_free_port(self.my_ip)
        self.rid = kv_cache_store.RaidenId(
            f"vllm_node_{self.node_tag}", "0", "vllm_bench", 0
        )

        built: Dict[str, Any] = {}
        err_box: List[BaseException] = []

        def _start_worker() -> None:
            try:
                time.sleep(0.2)
                # Every single-host replica passes node_id=0 so cross-replica
                # ReadRemote passes strict worker rank matching (0 <-> 0).
                built["manager"] = kv_cache_manager.KVCacheManager(
                    kv_caches=self.kv_caches,
                    local_control_port=0,
                    host_blocks_to_allocate=total_alloc_blocks,
                    unsafe_skip_buffer_lock=True,
                    parallelism=parallelism,
                    raiden_worker_port=self.worker_port,
                    raiden_controller_address=f"{self.my_ip}:{self.controller_port}",
                    worker_id=f"vllm_worker_{self.node_tag}",
                    node_id=0,
                    enable_shm=False,
                )
            except BaseException as exc:  # noqa: BLE001
                err_box.append(exc)

        t = threading.Thread(target=_start_worker, daemon=True)
        t.start()
        try:
            self.store = kv_cache_store.KVCacheStore(
                capacity=host_blocks,
                global_registry_address=self.registry_addr,
                raiden_id=self.rid,
                num_shards=num_shards,
                shard_size_bytes=shard_bytes_per_layer,
                store_server_ip=self.my_ip,
                raiden_controller_port=self.controller_port,
                expected_worker_count=1,
                kv_pool_group=pool_group,
            )
        finally:
            t.join(timeout=180)

        if err_box:
            raise RuntimeError(f"tpu-sync KVCacheManager failed: {err_box[0]}") from err_box[0]
        if "manager" not in built:
            raise RuntimeError("tpu-sync KVCacheManager startup timed out")
        self.manager = built["manager"]
        self._op_lock = threading.Lock()
        self.poll_timeout_s = 60.0

        logger.info(
            "TpuSyncStoreBridge ready: ip=%s node=%s registry=%s layers=%d "
            "shards=%d block_bytes=%d host_blocks=%d (%.1f GiB)",
            self.my_ip,
            self.node_tag,
            self.registry_addr,
            len(self.kv_caches),
            num_shards,
            total_block_bytes,
            host_blocks,
            (host_blocks * total_block_bytes) / (1024**3),
        )

    def _poll_hashes(
        self, poll_fn, expected: Set[bytes]
    ) -> Tuple[Set[bytes], Set[bytes]]:
        done_set: Set[bytes] = set()
        failed_set: Set[bytes] = set()
        deadline = time.perf_counter() + self.poll_timeout_s
        sleep_s = 0.0005
        while time.perf_counter() < deadline:
            done, failed = poll_fn()
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
        failed_set.update(expected - done_set - failed_set)
        return done_set, failed_set

    def lookup_prefix_len(self, raw_hashes: Sequence[bytes]) -> int:
        """Non-pinning prefix lookup used by the Scheduler."""
        if not raw_hashes:
            return 0
        status_enum = self._kcs.BlockStatus
        with self._op_lock:
            res = self.store.lookup(
                list(raw_hashes), enable_global=True, pin_found=False
            )
        matched = 0
        for _, blk in res:
            if blk.status in (
                status_enum.HOST,
                status_enum.HOST_AND_HBM,
                status_enum.REMOTE,
                status_enum.SHARED_STORAGE,
            ):
                matched += 1
            else:
                break
        return matched

    def load_blocks(self, raw_hashes: Sequence[bytes], hbm_slots: Sequence[int]) -> bool:
        if not raw_hashes:
            return True
        status_enum = self._kcs.BlockStatus
        with self._op_lock:
            res = self.store.lookup(
                list(raw_hashes), enable_global=True, pin_found=True
            )
            if len(res) < len(raw_hashes):
                # Release any local pins acquired on a partial race.
                to_rel = [
                    rh
                    for rh, blk in res
                    if blk.status in (status_enum.HOST, status_enum.HOST_AND_HBM)
                ]
                if to_rel:
                    self.store.release(to_rel)
                return False

            local_hashes: List[bytes] = []
            local_slots: List[int] = []
            remote_by_peer: Dict[Tuple[str, str], Tuple[List[bytes], List[Any], List[int]]] = {}

            for (rh, blk), slot in zip(res, hbm_slots):
                if blk.status in (status_enum.HOST, status_enum.HOST_AND_HBM):
                    local_hashes.append(rh)
                    local_slots.append(int(slot))
                elif blk.status in (status_enum.REMOTE, status_enum.SHARED_STORAGE):
                    rid = blk.raiden_id
                    pkey = (str(rid.job_name), str(rid.job_replica_id))
                    grp = remote_by_peer.setdefault(pkey, ([], [], []))
                    grp[0].append(rh)
                    grp[1].append(blk)
                    grp[2].append(int(slot))
                else:
                    # HBM-only entry not offload-readable.
                    if local_hashes:
                        self.store.release(local_hashes)
                    return False

            launched: Set[bytes] = set()
            local_set = set(local_hashes)
            launch_failed_local: List[bytes] = []
            ok_all = True

            if local_hashes:
                if self.store.load(local_hashes, local_slots):
                    launched.update(local_hashes)
                else:
                    launch_failed_local.extend(local_hashes)
                    ok_all = False

            for r_hashes, r_slices, r_slots in remote_by_peer.values():
                if self.store.read_remote(r_hashes, r_slices, r_slots):
                    launched.update(r_hashes)
                else:
                    ok_all = False

            done_set: Set[bytes] = set()
            poll_failed: Set[bytes] = set()
            if launched:
                done_set, poll_failed = self._poll_hashes(
                    self.store.poll_load_status, launched
                )

            pins_to_release = launch_failed_local + [
                rh for rh in poll_failed if rh in local_set
            ]
            if pins_to_release:
                self.store.release(pins_to_release)

            return ok_all and len(poll_failed) == 0 and len(done_set) == len(raw_hashes)

    def save_blocks(self, raw_hashes: Sequence[bytes], hbm_slots: Sequence[int]) -> bool:
        if not raw_hashes:
            return True
        status_enum = self._kcs.BlockStatus
        slices = [
            self._kcs.RaidenBlockId(
                self.rid,
                host_block_id=-1,
                device_block_id=int(slot),
                status=status_enum.HBM,
            )
            for slot in hbm_slots
        ]
        with self._op_lock:
            ins_hashes: List[bytes] = []
            if self.store.insert(list(raw_hashes), slices, on_host=False):
                ins_hashes = list(raw_hashes)
            else:
                for rh, sl in zip(raw_hashes, slices):
                    if self.store.insert([rh], [sl], on_host=False):
                        ins_hashes.append(rh)
                    else:
                        break
            if not ins_hashes:
                return False
            if not self.store.save(ins_hashes):
                self.store.release(ins_hashes)
                return False
            done_set, failed_set = self._poll_hashes(
                self.store.poll_save_status, set(ins_hashes)
            )
            if failed_set:
                self.store.release(list(failed_set))
            return len(done_set) == len(raw_hashes)

    def close(self) -> None:
        try:
            self.manager = None
            self.store = None
        except Exception:
            pass


# In single-host JAX SPMD (`tpu-inference`), Scheduler and Worker live in the
# same EngineCore process; share the initialized KVCacheStore bridge cleanly.
_SHARED_BRIDGE_LOCK = threading.Lock()
_SHARED_BRIDGE: Optional[_TpuSyncStoreBridge] = None


def _get_shared_bridge() -> Optional[_TpuSyncStoreBridge]:
    with _SHARED_BRIDGE_LOCK:
        return _SHARED_BRIDGE


def _set_shared_bridge(bridge: Optional[_TpuSyncStoreBridge]) -> None:
    global _SHARED_BRIDGE
    with _SHARED_BRIDGE_LOCK:
        _SHARED_BRIDGE = bridge


# ===========================================================================
# 1. TpuSyncTPUOffloadConnector (kv_role="kv_both", shared cluster prefix cache)
# ===========================================================================


@dataclass
class _TSOffloadLoadReq:
    hashes: List[bytes]
    block_ids: List[int]


@dataclass
class _TSOffloadSaveReq:
    hashes: List[bytes]
    block_ids: List[int]


@dataclass
class TpuSyncOffloadMetadata(KVConnectorMetadata):
    reqs_to_load: Dict[ReqId, _TSOffloadLoadReq] = field(default_factory=dict)
    reqs_to_save: Dict[ReqId, _TSOffloadSaveReq] = field(default_factory=dict)


class _TpuSyncOffloadScheduler:
    def __init__(self, vllm_config: VllmConfig):
        self.block_size = vllm_config.cache_config.block_size
        model_name = os.environ.get(
            "TPUSYNC_MODEL_TAG", vllm_config.model_config.model
        ).replace("/", "_")
        self.namespace = f"ts_tpu_offload:{model_name}:bs{self.block_size}"

        self._pending_load_hashes: Dict[ReqId, Tuple[int, List[bytes]]] = {}
        self._reqs_to_load: Dict[ReqId, _TSOffloadLoadReq] = {}
        self._reqs_to_save: Dict[ReqId, _TSOffloadSaveReq] = {}
        self._saved_watermark_blocks: Dict[ReqId, int] = {}

        self._total_queries = 0
        self._total_hits = 0
        self._total_requests = 0
        self._last_log_t = time.monotonic()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> Tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        start_blk = num_computed_tokens // self.block_size
        all_hashes = _prefix_block_hashes(
            self.namespace, request.prompt_token_ids, self.block_size
        )
        candidate = all_hashes[start_blk:]
        if not candidate:
            return 0, False

        bridge = _get_shared_bridge()
        matched = bridge.lookup_prefix_len(candidate) if bridge is not None else 0

        self._total_requests += 1
        self._total_queries += len(candidate)
        self._total_hits += matched
        now = time.monotonic()
        if now - self._last_log_t >= 5.0:
            rate = (self._total_hits / self._total_queries * 100.0) if self._total_queries else 0.0
            logger.info(
                "External prefix cache hit rate: %.2f%% (requests=%d, hits=%d, queries=%d)",
                rate,
                self._total_requests,
                self._total_hits,
                self._total_queries,
            )
            self._last_log_t = now

        if matched <= 0:
            self._saved_watermark_blocks[request.request_id] = start_blk
            return 0, False

        self._pending_load_hashes[request.request_id] = (start_blk, candidate[:matched])
        self._saved_watermark_blocks[request.request_id] = start_blk + matched
        return matched * self.block_size, True

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        info = self._pending_load_hashes.pop(request.request_id, None)
        if info is None or num_external_tokens <= 0:
            return
        start_blk, matched_hashes = info
        all_block_ids = blocks.get_block_ids()[0]
        dst_ids = all_block_ids[start_blk : start_blk + len(matched_hashes)]
        if len(dst_ids) == len(matched_hashes):
            self._reqs_to_load[request.request_id] = _TSOffloadLoadReq(
                hashes=matched_hashes, block_ids=list(dst_ids)
            )

    def build_connector_meta(self) -> TpuSyncOffloadMetadata:
        meta = TpuSyncOffloadMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=self._reqs_to_save,
        )
        self._reqs_to_load = {}
        self._reqs_to_save = {}
        return meta

    def request_finished(
        self, request: "Request", block_ids: List[int]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        watermark = self._saved_watermark_blocks.pop(request.request_id, 0)
        all_hashes = _prefix_block_hashes(
            self.namespace, request.prompt_token_ids, self.block_size
        )
        num_full = min(len(all_hashes), len(block_ids))
        if num_full <= watermark:
            return False, None
        save_hashes = all_hashes[watermark:num_full]
        save_blocks = list(block_ids[watermark:num_full])
        if not save_hashes:
            return False, None
        self._reqs_to_save[request.request_id] = _TSOffloadSaveReq(
            hashes=save_hashes, block_ids=save_blocks
        )
        return True, None


class _TpuSyncOffloadWorker:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.bridge: Optional[_TpuSyncStoreBridge] = None
        self.runner = None
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        self._done_sending: Set[ReqId] = set()
        self._done_recving: Set[ReqId] = set()
        self._pending_futures: List[Future] = []

    def register_runner(self, runner: Any) -> None:
        self.runner = runner
        host_gb = float(os.environ.get("TPUSYNC_HOST_CACHE_GB", "32"))
        self.bridge = _TpuSyncStoreBridge(runner.kv_caches, host_cache_gb=host_gb)
        _set_shared_bridge(self.bridge)

    def process_meta(self, meta: TpuSyncOffloadMetadata) -> None:
        if self.bridge is None:
            return
        if meta.reqs_to_save and self.runner is not None:
            jax.block_until_ready(self.runner.kv_caches)

        for req_id, load_req in meta.reqs_to_load.items():
            def _do_load(rid=req_id, lr=load_req):
                try:
                    ok = self.bridge.load_blocks(lr.hashes, lr.block_ids)
                    if not ok:
                        logger.warning("tpu-sync offload load failed for req %s", rid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("tpu-sync offload load exception for req %s: %s", rid, exc)
                finally:
                    with self._lock:
                        self._done_recving.add(rid)

            self._pending_futures.append(self._pool.submit(_do_load))

        for req_id, save_req in meta.reqs_to_save.items():
            def _do_save(rid=req_id, sr=save_req):
                try:
                    ok = self.bridge.save_blocks(sr.hashes, sr.block_ids)
                    if not ok:
                        logger.warning("tpu-sync offload save partial/rejected for req %s", rid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("tpu-sync offload save exception for req %s: %s", rid, exc)
                finally:
                    with self._lock:
                        self._done_sending.add(rid)

            self._pending_futures.append(self._pool.submit(_do_save))

    def get_finished(self) -> Tuple[Set[ReqId], Set[ReqId]]:
        with self._lock:
            done_s = set(self._done_sending)
            done_r = set(self._done_recving)
            self._done_sending.clear()
            self._done_recving.clear()
        self._pending_futures = [f for f in self._pending_futures if not f.done()]
        return done_s, done_r

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        if self.bridge is not None:
            _set_shared_bridge(None)
            self.bridge.close()
            self.bridge = None


class TpuSyncTPUOffloadConnector(KVConnectorBase_V1):
    """vLLM V1 shared cluster prefix-cache offload connector for JAX TPU + tpu-sync."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._connector_metadata = None
        if role == KVConnectorRole.SCHEDULER:
            self._sched = _TpuSyncOffloadScheduler(vllm_config)
            self._worker = None
        else:
            self._sched = None
            self._worker = _TpuSyncOffloadWorker(vllm_config)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> Tuple[int, bool]:
        assert self._sched is not None
        return self._sched.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self._sched is not None
        self._sched.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self._sched is not None
        return self._sched.build_connector_meta()

    def request_finished(self, request: "Request", block_ids: List[int]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        assert self._sched is not None
        return self._sched.request_finished(request, block_ids)

    def register_kv_caches(self, kv_caches: Any) -> None:
        pass

    def register_runner(self, runner: Any) -> None:
        assert self._worker is not None
        self._worker.register_runner(runner)

    def start_load_kv(self, _, **kwargs) -> None:
        assert self._worker is not None
        if isinstance(self._connector_metadata, TpuSyncOffloadMetadata):
            self._worker.process_meta(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(self, finished_req_ids: Set[str]) -> Tuple[Set[str], Set[str]]:
        assert self._worker is not None
        return self._worker.get_finished()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()


# ===========================================================================
# 2. TpuSyncTPUDisaggConnector (1P1D disaggregated serving via KVCacheStore)
# ===========================================================================


@dataclass
class _TSDisaggSendMeta:
    uuid: int
    local_block_ids: List[int]


@dataclass
class _TSDisaggLoadMeta:
    uuid: int
    local_block_ids: Optional[List[int]]
    remote_block_ids: Optional[List[int]]


@dataclass
class TpuSyncDisaggMetadata(KVConnectorMetadata):
    reqs_to_send: Dict[ReqId, _TSDisaggSendMeta] = field(default_factory=dict)
    reqs_to_load: Dict[ReqId, _TSDisaggLoadMeta] = field(default_factory=dict)


class TpuSyncTPUDisaggConnector(KVConnectorBase_V1):
    """Disaggregated 1P1D connector backed by tpu-sync `KVCacheStore`.

    Compatible with `toy_proxy_server.py`. Prefill saves finished blocks into
    its host `KVCacheStore` under `disagg:<uuid>:<i>` and frees HBM; Decode
    pulls them into Decode HBM via `read_remote`.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.is_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.block_size = vllm_config.cache_config.block_size
        self._connector_metadata = None
        self._reqs_to_send: Dict[ReqId, _TSDisaggSendMeta] = {}
        self._reqs_to_load: Dict[ReqId, _TSDisaggLoadMeta] = {}
        self._host_ip = _detect_host_ip()

        self._bridge: Optional[_TpuSyncStoreBridge] = None
        self._runner = None
        self._pool = ThreadPoolExecutor(max_workers=8) if role == KVConnectorRole.WORKER else None
        self._lock = threading.Lock()
        self._done_sending: Set[ReqId] = set()
        self._done_recving: Set[ReqId] = set()
        self._submitted: Set[ReqId] = set()

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> Tuple[int, bool]:
        if self.is_producer or not request.kv_transfer_params:
            return 0, False
        if request.kv_transfer_params.get("do_remote_prefill", True) is False:
            return 0, False
        rounded = round_down(len(request.prompt_token_ids), self.block_size)
        count = max(rounded - num_computed_tokens, 0)
        return (count, True) if count > 0 else (0, False)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return
        params = request.kv_transfer_params
        if num_external_tokens > 0:
            self._reqs_to_load[request.request_id] = _TSDisaggLoadMeta(
                uuid=int(params["uuid"]),
                local_block_ids=list(blocks.get_block_ids()[0]),
                remote_block_ids=list(params["remote_block_ids"]),
            )
        else:
            self._reqs_to_load[request.request_id] = _TSDisaggLoadMeta(
                uuid=int(params["uuid"]),
                local_block_ids=None,
                remote_block_ids=None,
            )
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> TpuSyncDisaggMetadata:
        meta = TpuSyncDisaggMetadata()
        if self.is_producer:
            meta.reqs_to_send = self._reqs_to_send
            self._reqs_to_send = {}
        else:
            meta.reqs_to_load = self._reqs_to_load
            self._reqs_to_load = {}
        return meta

    def request_finished(self, request: "Request", block_ids: List[int]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        if not self.is_producer:
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_ids = list(block_ids if all_full else block_ids[:-1])
        if not computed_ids:
            return False, {}
        uid = _get_uuid_50bit()
        self._reqs_to_send[request.request_id] = _TSDisaggSendMeta(
            uuid=uid, local_block_ids=computed_ids
        )
        return True, {
            "uuid": uid,
            "remote_block_ids": computed_ids,
            "remote_host": self._host_ip,
            "remote_port": 0,
        }

    def get_finished_count(self) -> int:
        return 1

    def register_kv_caches(self, kv_caches: Any) -> None:
        pass

    def register_runner(self, runner: Any) -> None:
        self._runner = runner
        default_gb = "32" if self.is_producer else os.environ.get("TPUSYNC_DECODE_CACHE_GB", "4")
        host_gb = float(os.environ.get("TPUSYNC_HOST_CACHE_GB", default_gb))
        self._bridge = _TpuSyncStoreBridge(runner.kv_caches, host_cache_gb=host_gb)

    def start_load_kv(self, _, **kwargs) -> None:
        if self._bridge is None or not isinstance(self._connector_metadata, TpuSyncDisaggMetadata):
            return
        meta = self._connector_metadata

        if meta.reqs_to_send and self._runner is not None:
            jax.block_until_ready(self._runner.kv_caches)

        for req_id, sm in meta.reqs_to_send.items():
            hashes = [_disagg_block_hash(sm.uuid, i) for i in range(len(sm.local_block_ids))]

            def _send(rid=req_id, hs=hashes, bids=sm.local_block_ids):
                try:
                    ok = self._bridge.save_blocks(hs, bids)
                    if not ok:
                        logger.error("tpu-sync disagg send failed for req %s", rid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("tpu-sync disagg send exception for req %s: %s", rid, exc)
                finally:
                    with self._lock:
                        self._done_sending.add(rid)

            self._pool.submit(_send)

        for req_id, lm in meta.reqs_to_load.items():
            if lm.remote_block_ids is None or lm.local_block_ids is None:
                self._submitted.discard(req_id)
                continue
            if req_id in self._submitted:
                continue
            self._submitted.add(req_id)
            n = min(len(lm.remote_block_ids), len(lm.local_block_ids))
            hashes = [_disagg_block_hash(lm.uuid, i) for i in range(n)]
            dst_bids = lm.local_block_ids[:n]

            def _recv(rid=req_id, hs=hashes, bids=dst_bids):
                deadline = time.monotonic() + 60.0
                loaded = False
                while time.monotonic() < deadline:
                    try:
                        if self._bridge.lookup_prefix_len(hs) == len(hs):
                            loaded = self._bridge.load_blocks(hs, bids)
                            if loaded:
                                break
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("tpu-sync disagg recv retry for %s: %s", rid, exc)
                    time.sleep(0.005)
                if loaded:
                    with self._lock:
                        self._done_recving.add(rid)
                else:
                    logger.error("tpu-sync disagg recv timed out / failed for req %s", rid)

            self._pool.submit(_recv)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, *args, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(self, finished_req_ids: Set[str]) -> Tuple[Set[str], Set[str]]:
        with self._lock:
            ds = set(self._done_sending)
            dr = set(self._done_recving)
            self._done_sending.clear()
            self._done_recving.clear()
        return ds, dr

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
