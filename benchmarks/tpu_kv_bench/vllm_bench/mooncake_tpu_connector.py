#!/usr/bin/env python3
"""Mooncake Store KV connectors for vLLM V1 on JAX/TPU (`tpu-inference`).

Provides two drop-in vLLM V1 `KVConnectorBase_V1` connectors for Tier 3
end-to-end serving benchmarks on multi-host CDK:

1. `MooncakeTPUOffloadConnector` (`kv_role="kv_both"`):
   Cluster-wide shared prefix-cache KV offloading across 1..N vLLM-TPU serving
   pods backed by `MooncakeDistributedStore` over TENT TPU. Drop-in comparison
   against `RaidenOffloadConnector`.

   --kv-transfer-config '{
     "kv_connector": "MooncakeTPUOffloadConnector",
     "kv_connector_module_path": "mooncake_tpu_connector",
     "kv_role": "kv_both"
   }'

2. `MooncakeTPUDisaggConnector` (`kv_role="kv_producer"` / `"kv_consumer"`):
   Multi-host disaggregated 1P1D KV transfer using the same `toy_proxy_server`
   `kv_transfer_params` handshake as `tpu_inference.distributed.tpu_raiden_connector`,
   moving KV blocks from Prefill HBM through Mooncake Store into Decode HBM.

   Prefill:
   --kv-transfer-config '{
     "kv_connector": "MooncakeTPUDisaggConnector",
     "kv_connector_module_path": "mooncake_tpu_connector",
     "kv_role": "kv_producer"
   }'
   Decode:
   --kv-transfer-config '{
     "kv_connector": "MooncakeTPUDisaggConnector",
     "kv_connector_module_path": "mooncake_tpu_connector",
     "kv_role": "kv_consumer"
   }'

Environment variables:
  MOONCAKE_MASTER_ADDR        e.g. "10.0.0.5:50051" (required)
  MOONCAKE_METADATA_URL       e.g. "http://10.0.0.5:8080/metadata" (required)
  MOONCAKE_HOST_IP            reachable IP of this pod (auto-detected if unset)
  MOONCAKE_TENT_PROTOCOL      "hp_tcp" (default) or "tcp"
  MOONCAKE_TENT_CONNECTIONS   hp_tcp connections per peer (default: 8)
  MOONCAKE_GLOBAL_SEGMENT_GB  host DRAM pool per pod in GiB (default: 32)
  MOONCAKE_LOCAL_BUFFER_MB    client local buffer size in MiB (default: 128)
  MOONCAKE_REPLICAS           memory replica count (default: 1)
  MOONCAKE_PREFER_LOCAL       "1" (default) or "0"
  MOONCAKE_MODEL_TAG          key namespace prefix (default: model name)
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
import socket
import struct
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

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
MC_OBJECT_ALREADY_EXISTS = -705


def _detect_host_ip() -> str:
    env_ip = os.environ.get("MOONCAKE_HOST_IP", "").strip()
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


def _configure_tent_env(my_ip: str) -> None:
    os.environ["MC_USE_TENT"] = "1"
    os.environ.setdefault("MC_STORE_MEMCPY", "0")
    if "MC_TENT_CONF" in os.environ:
        return
    protocol = os.environ.get("MOONCAKE_TENT_PROTOCOL", "hp_tcp")
    connections = int(os.environ.get("MOONCAKE_TENT_CONNECTIONS", "8"))
    hp_tcp: Dict[str, Any] = {"enable": protocol == "hp_tcp"}
    if protocol == "hp_tcp":
        hp_tcp["connections_per_peer"] = connections
    conf = {
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
    os.environ["MC_TENT_CONF"] = json.dumps(conf)


def _store_endpoints_from_env() -> Tuple[str, str]:
    master = os.environ.get("MOONCAKE_MASTER_ADDR", "127.0.0.1:50051")
    meta = os.environ.get("MOONCAKE_METADATA_URL", "http://127.0.0.1:8080/metadata")
    return master, meta


def _prefix_block_keys(
    namespace: str, token_ids: Sequence[int], block_size: int
) -> List[str]:
    """Deterministic prefix-chained store keys for full `block_size` chunks."""
    num_full = len(token_ids) // block_size
    if num_full <= 0:
        return []
    keys: List[str] = []
    acc = namespace.encode("utf-8")
    for b in range(num_full):
        chunk = token_ids[b * block_size : (b + 1) * block_size]
        packed = struct.pack(f">{len(chunk)}I", *(int(t) & 0xFFFFFFFF for t in chunk))
        acc = hashlib.blake2b(acc + packed, digest_size=16).digest()
        keys.append(f"{namespace}:{acc.hex()}")
    return keys


def _get_uuid_50bit() -> int:
    # Matches tpu_raiden_connector: < 64-bit safe JSON integer.
    return uuid4().int >> 78


class _MooncakeTPUStoreBridge:
    """Shared worker-side bridge exporting JAX `runner.kv_caches` into Mooncake Store."""

    def __init__(self, kv_caches: Sequence[jax.Array], global_segment_bytes: int):
        from mooncake import tpu as mc_tpu
        import mooncake.store as mc_store

        self._mc_tpu = mc_tpu
        self.my_ip = _detect_host_ip()
        _configure_tent_env(self.my_ip)
        master_addr, metadata_url = _store_endpoints_from_env()

        self.kv_caches = list(kv_caches)
        if not self.kv_caches:
            raise RuntimeError("runner.kv_caches is empty")

        # Export all TPU shards BEFORE creating MooncakeDistributedStore / TENT.
        self.shards = mc_tpu.prepare(self.kv_caches)
        num_blocks = int(self.kv_caches[0].shape[0])
        self.num_blocks = num_blocks
        self.block_shard_bytes: List[List[int]] = []
        for layer_idx, layer_shards in enumerate(self.shards):
            per_dev: List[int] = []
            for sh in layer_shards:
                if sh.size % num_blocks != 0:
                    raise RuntimeError(
                        f"Layer {layer_idx}: shard.size={sh.size} not divisible "
                        f"by num_blocks={num_blocks}"
                    )
                per_dev.append(sh.size // num_blocks)
            self.block_shard_bytes.append(per_dev)

        port = _pick_free_port(self.my_ip)
        self.local_hostname = f"{self.my_ip}:{port}"
        local_buf_mb = int(os.environ.get("MOONCAKE_LOCAL_BUFFER_MB", "128"))
        local_buf_bytes = max(16 * 1024 * 1024, local_buf_mb * 1024 * 1024)

        self.store = mc_store.MooncakeDistributedStore()
        rc = self.store.setup(
            self.local_hostname,
            metadata_url,
            int(global_segment_bytes),
            local_buf_bytes,
            "tcp",
            "",
            master_addr,
        )
        if rc != 0:
            mc_tpu.unregister([s for layer in self.shards for s in layer])
            raise RuntimeError(
                f"MooncakeDistributedStore.setup failed rc={rc} "
                f"(local={self.local_hostname}, master={master_addr}, meta={metadata_url})"
            )

        self._registered: List[int] = []
        for layer_shards in self.shards:
            for sh in layer_shards:
                rc = self.store.register_buffer(sh.fake_addr, sh.size)
                if rc != 0:
                    self.close()
                    raise RuntimeError(f"register_buffer failed rc={rc}")
                self._registered.append(sh.fake_addr)

        self.rep_config = mc_store.ReplicateConfig()
        self.rep_config.replica_num = int(os.environ.get("MOONCAKE_REPLICAS", "1"))
        self.rep_config.prefer_alloc_in_same_node = (
            os.environ.get("MOONCAKE_PREFER_LOCAL", "1").lower() in ("1", "true", "yes")
        )
        self.prefer_local = self.rep_config.prefer_alloc_in_same_node
        logger.info(
            "MooncakeTPUStoreBridge ready: local=%s master=%s layers=%d blocks=%d seg_gib=%.1f",
            self.local_hostname,
            master_addr,
            len(self.shards),
            self.num_blocks,
            global_segment_bytes / (1024**3),
        )

    def _buffers_for_block(self, block_id: int) -> Tuple[List[int], List[int]]:
        ptrs: List[int] = []
        sizes: List[int] = []
        for layer_shards, dev_bytes in zip(self.shards, self.block_shard_bytes):
            for sh, bsz in zip(layer_shards, dev_bytes):
                ptrs.append(sh.fake_addr + int(block_id) * bsz)
                sizes.append(bsz)
        return ptrs, sizes

    def put_blocks(self, keys: Sequence[str], block_ids: Sequence[int]) -> bool:
        if not keys:
            return True
        all_ptrs: List[List[int]] = []
        all_sizes: List[List[int]] = []
        for bid in block_ids:
            p, s = self._buffers_for_block(bid)
            all_ptrs.append(p)
            all_sizes.append(s)
        rcs = self.store.batch_put_from_multi_buffers(
            list(keys), all_ptrs, all_sizes, self.rep_config
        )
        return all(rc == 0 or rc == MC_OBJECT_ALREADY_EXISTS for rc in rcs)

    def get_blocks(self, keys: Sequence[str], block_ids: Sequence[int]) -> bool:
        if not keys:
            return True
        all_ptrs: List[List[int]] = []
        all_sizes: List[List[int]] = []
        for bid in block_ids:
            p, s = self._buffers_for_block(bid)
            all_ptrs.append(p)
            all_sizes.append(s)
        rcs = self.store.batch_get_into_multi_buffers(
            list(keys), all_ptrs, all_sizes, prefer_alloc_in_same_node=self.prefer_local
        )
        return all(rc >= 0 for rc in rcs)

    def close(self) -> None:
        if getattr(self, "store", None) is not None:
            for addr in reversed(self._registered):
                try:
                    self.store.unregister_buffer(addr)
                except Exception:
                    pass
            self._registered.clear()
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


# ===========================================================================
# 1. MooncakeTPUOffloadConnector (kv_role="kv_both", shared cluster prefix cache)
# ===========================================================================


@dataclass
class _OffloadLoadReq:
    keys: List[str]
    block_ids: List[int]


@dataclass
class _OffloadSaveReq:
    keys: List[str]
    block_ids: List[int]


@dataclass
class MooncakeOffloadMetadata(KVConnectorMetadata):
    reqs_to_load: Dict[ReqId, _OffloadLoadReq] = field(default_factory=dict)
    reqs_to_save: Dict[ReqId, _OffloadSaveReq] = field(default_factory=dict)


class _OffloadScheduler:
    """Scheduler-side prefix lookup and work-order builder backed by Mooncake master."""

    def __init__(self, vllm_config: VllmConfig):
        import mooncake.store as mc_store

        self.block_size = vllm_config.cache_config.block_size
        model_name = os.environ.get(
            "MOONCAKE_MODEL_TAG", vllm_config.model_config.model
        ).replace("/", "_")
        self.namespace = f"mc_tpu_offload:{model_name}:bs{self.block_size}"

        my_ip = _detect_host_ip()
        _configure_tent_env(my_ip)
        master_addr, metadata_url = _store_endpoints_from_env()
        port = _pick_free_port(my_ip)

        # Lightweight metadata-only client (0 global segment, 16 MiB local buffer).
        self.meta_store = mc_store.MooncakeDistributedStore()
        rc = self.meta_store.setup(
            f"{my_ip}:{port}",
            metadata_url,
            0,
            16 * 1024 * 1024,
            "tcp",
            "",
            master_addr,
        )
        if rc != 0:
            raise RuntimeError(f"Mooncake offload scheduler client setup failed: {rc}")

        self._pending_load_keys: Dict[ReqId, Tuple[int, List[str]]] = {}
        self._reqs_to_load: Dict[ReqId, _OffloadLoadReq] = {}
        self._reqs_to_save: Dict[ReqId, _OffloadSaveReq] = {}
        self._saved_watermark_blocks: Dict[ReqId, int] = {}

        # External prefix cache hit accounting (logged periodically in the same
        # format as RaidenOffloadConnector so benchmark parsers treat both identically).
        self._total_queries = 0
        self._total_hits = 0
        self._total_requests = 0
        self._last_log_t = time.monotonic()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> Tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        start_blk = num_computed_tokens // self.block_size
        all_keys = _prefix_block_keys(self.namespace, request.prompt_token_ids, self.block_size)
        candidate_keys = all_keys[start_blk:]
        if not candidate_keys:
            return 0, False

        exists = self.meta_store.batch_is_exist(candidate_keys)
        matched = 0
        for flag in exists:
            if flag == 1:
                matched += 1
            else:
                break

        self._total_requests += 1
        self._total_queries += len(candidate_keys)
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

        self._pending_load_keys[request.request_id] = (start_blk, candidate_keys[:matched])
        self._saved_watermark_blocks[request.request_id] = start_blk + matched
        return matched * self.block_size, True

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        info = self._pending_load_keys.pop(request.request_id, None)
        if info is None or num_external_tokens <= 0:
            return
        start_blk, matched_keys = info
        all_block_ids = blocks.get_block_ids()[0]
        dst_ids = all_block_ids[start_blk : start_blk + len(matched_keys)]
        if len(dst_ids) == len(matched_keys):
            self._reqs_to_load[request.request_id] = _OffloadLoadReq(
                keys=matched_keys, block_ids=list(dst_ids)
            )

    def build_connector_meta(self) -> MooncakeOffloadMetadata:
        meta = MooncakeOffloadMetadata(
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
        all_keys = _prefix_block_keys(self.namespace, request.prompt_token_ids, self.block_size)
        num_full = min(len(all_keys), len(block_ids))
        if num_full <= watermark:
            return False, None
        save_keys = all_keys[watermark:num_full]
        save_blocks = list(block_ids[watermark:num_full])
        if not save_keys:
            return False, None
        self._reqs_to_save[request.request_id] = _OffloadSaveReq(
            keys=save_keys, block_ids=save_blocks
        )
        # Delay freeing HBM blocks until the worker confirms async save completion.
        return True, None


class _OffloadWorker:
    """Worker-side async HBM <-> Mooncake Store loader and saver."""

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.bridge: Optional[_MooncakeTPUStoreBridge] = None
        self.runner = None
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        self._done_sending: Set[ReqId] = set()
        self._done_recving: Set[ReqId] = set()
        self._pending_futures: List[Future] = []

    def register_runner(self, runner: Any) -> None:
        self.runner = runner
        seg_gb = float(os.environ.get("MOONCAKE_GLOBAL_SEGMENT_GB", "32"))
        seg_bytes = int(seg_gb * (1024**3))
        self.bridge = _MooncakeTPUStoreBridge(runner.kv_caches, global_segment_bytes=seg_bytes)

    def process_meta(self, meta: MooncakeOffloadMetadata) -> None:
        if self.bridge is None:
            return

        if meta.reqs_to_save and self.runner is not None:
            # Ensure TPU forward writes to kv_caches are complete before D2H DMA reads.
            jax.block_until_ready(self.runner.kv_caches)

        for req_id, load_req in meta.reqs_to_load.items():
            def _do_load(rid=req_id, lr=load_req):
                try:
                    ok = self.bridge.get_blocks(lr.keys, lr.block_ids)
                    if not ok:
                        logger.warning("Mooncake offload load failed for req %s", rid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Mooncake offload load exception for req %s: %s", rid, exc)
                finally:
                    with self._lock:
                        self._done_recving.add(rid)

            self._pending_futures.append(self._pool.submit(_do_load))

        for req_id, save_req in meta.reqs_to_save.items():
            def _do_save(rid=req_id, sr=save_req):
                try:
                    ok = self.bridge.put_blocks(sr.keys, sr.block_ids)
                    if not ok:
                        logger.warning("Mooncake offload save partial/rejected for req %s", rid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Mooncake offload save exception for req %s: %s", rid, exc)
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
            self.bridge.close()
            self.bridge = None


class MooncakeTPUOffloadConnector(KVConnectorBase_V1):
    """vLLM V1 shared cluster prefix-cache offload connector for JAX TPU + Mooncake Store."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._connector_metadata = None
        if role == KVConnectorRole.SCHEDULER:
            self._sched = _OffloadScheduler(vllm_config)
            self._worker = None
        else:
            self._sched = None
            self._worker = _OffloadWorker(vllm_config)

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
        if isinstance(self._connector_metadata, MooncakeOffloadMetadata):
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
# 2. MooncakeTPUDisaggConnector (1P1D disaggregated serving via Mooncake Store)
# ===========================================================================


@dataclass
class _DisaggSendMeta:
    uuid: int
    local_block_ids: List[int]


@dataclass
class _DisaggLoadMeta:
    uuid: int
    local_block_ids: Optional[List[int]]
    remote_block_ids: Optional[List[int]]


@dataclass
class MooncakeDisaggMetadata(KVConnectorMetadata):
    reqs_to_send: Dict[ReqId, _DisaggSendMeta] = field(default_factory=dict)
    reqs_to_load: Dict[ReqId, _DisaggLoadMeta] = field(default_factory=dict)


class MooncakeTPUDisaggConnector(KVConnectorBase_V1):
    """Disaggregated 1P1D connector backed by Mooncake Store over TENT TPU.

    Compatible with `tpu-inference/examples/disagg/toy_proxy_server.py`.
    Prefill publishes finished blocks to Mooncake Store under `disagg:<uuid>:<i>`
    and immediately frees HBM; Decode pulls them directly into Decode HBM.
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
        self._reqs_to_send: Dict[ReqId, _DisaggSendMeta] = {}
        self._reqs_to_load: Dict[ReqId, _DisaggLoadMeta] = {}
        self._host_ip = _detect_host_ip()

        # Worker state
        self._bridge: Optional[_MooncakeTPUStoreBridge] = None
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
            self._reqs_to_load[request.request_id] = _DisaggLoadMeta(
                uuid=int(params["uuid"]),
                local_block_ids=list(blocks.get_block_ids()[0]),
                remote_block_ids=list(params["remote_block_ids"]),
            )
        else:
            self._reqs_to_load[request.request_id] = _DisaggLoadMeta(
                uuid=int(params["uuid"]),
                local_block_ids=None,
                remote_block_ids=None,
            )
        params["do_remote_prefill"] = False

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeDisaggMetadata:
        meta = MooncakeDisaggMetadata()
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
        self._reqs_to_send[request.request_id] = _DisaggSendMeta(
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
        # Prefill hosts the DRAM pool; Decode can also contribute or set 0.
        default_gb = "32" if self.is_producer else os.environ.get("MOONCAKE_DECODE_SEGMENT_GB", "4")
        seg_gb = float(os.environ.get("MOONCAKE_GLOBAL_SEGMENT_GB", default_gb))
        self._bridge = _MooncakeTPUStoreBridge(
            runner.kv_caches, global_segment_bytes=int(seg_gb * (1024**3))
        )

    def start_load_kv(self, _, **kwargs) -> None:
        if self._bridge is None or not isinstance(self._connector_metadata, MooncakeDisaggMetadata):
            return
        meta = self._connector_metadata

        if meta.reqs_to_send and self._runner is not None:
            jax.block_until_ready(self._runner.kv_caches)

        for req_id, sm in meta.reqs_to_send.items():
            keys = [f"disagg:{sm.uuid}:{i}" for i in range(len(sm.local_block_ids))]

            def _send(rid=req_id, ks=keys, bids=sm.local_block_ids):
                try:
                    ok = self._bridge.put_blocks(ks, bids)
                    if not ok:
                        logger.error("Mooncake disagg send failed for req %s", rid)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Mooncake disagg send exception for req %s: %s", rid, exc)
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
            keys = [f"disagg:{lm.uuid}:{i}" for i in range(n)]
            dst_bids = lm.local_block_ids[:n]

            def _recv(rid=req_id, ks=keys, bids=dst_bids):
                # Prefill's async put may still be finishing right after response 1 token;
                # poll briefly until keys are published.
                deadline = time.monotonic() + 60.0
                loaded = False
                while time.monotonic() < deadline:
                    try:
                        ex = self._bridge.store.batch_is_exist(ks)
                        if all(v == 1 for v in ex):
                            loaded = self._bridge.get_blocks(ks, bids)
                            break
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Mooncake disagg recv retry for %s: %s", rid, exc)
                    time.sleep(0.005)
                if loaded:
                    try:
                        self._bridge.store.batch_remove(ks, True)
                    except Exception:
                        pass
                    with self._lock:
                        self._done_recving.add(rid)
                else:
                    logger.error("Mooncake disagg recv timed out / failed for req %s", rid)

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


# Aliases accepted by tpu_platform.py's kv_connector allowlist when loaded via
# kv_connector_module_path="mooncake_tpu_connector".
TPUOffloadConnector = MooncakeTPUOffloadConnector
RaidenOffloadConnector = MooncakeTPUOffloadConnector
TPUConnector = MooncakeTPUDisaggConnector
