#!/usr/bin/env python3
"""Trace loaders and synthetic KV-cache workload generators for Tier 2 benchmarks.

Every workload produces a deterministic list of KVRequest records distributed across
`num_nodes` serving hosts, using 16-byte content hashes for prefix blocks so both
Mooncake Store and tpu-sync KVCacheStore run the exact same block sequence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class KVGeometry:
    """KV-cache tensor geometry shared by both backends on each TPU host."""

    name: str
    num_layers: int
    tokens_per_block: int
    head_shards: int
    heads_per_shard: int
    head_dim: int
    dtype: str  # "bf16" or "fp32"
    hbm_blocks: int
    host_blocks: int

    @property
    def dtype_bytes(self) -> int:
        if self.dtype in ("bf16", "bfloat16", "fp16", "float16"):
            return 2
        if self.dtype in ("fp32", "float32"):
            return 4
        if self.dtype in ("fp8", "float8_e4m3fn"):
            return 1
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    def layer_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.hbm_blocks,
            self.tokens_per_block,
            self.head_shards,
            self.heads_per_shard,
            self.head_dim,
        )

    def block_bytes_per_layer(self) -> int:
        return (
            self.tokens_per_block
            * self.head_shards
            * self.heads_per_shard
            * self.head_dim
            * self.dtype_bytes
        )

    def block_shard_bytes(self, num_devices: int) -> int:
        if self.head_shards % num_devices != 0:
            raise ValueError(
                f"head_shards={self.head_shards} must be divisible by "
                f"num_devices={num_devices}"
            )
        return self.block_bytes_per_layer() // num_devices

    def block_total_bytes(self) -> int:
        return self.block_bytes_per_layer() * self.num_layers

    def host_pool_bytes(self) -> int:
        return self.block_total_bytes() * self.host_blocks

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["block_bytes_per_layer"] = self.block_bytes_per_layer()
        d["block_total_bytes"] = self.block_total_bytes()
        d["host_pool_bytes"] = self.host_pool_bytes()
        return d


GEOMETRY_PRESETS: Dict[str, KVGeometry] = {
    # Fast local smoke preset: 1 MiB/block (2 layers x 512 KiB/layer in bf16).
    "small": KVGeometry(
        name="small",
        num_layers=2,
        tokens_per_block=128,
        head_shards=8,
        heads_per_shard=2,
        head_dim=128,
        dtype="bf16",
        hbm_blocks=32,
        host_blocks=96,
    ),
    # Default multi-host CDK preset: 8 MiB/block (8 layers x 1 MiB/layer in bf16).
    # 512 host blocks/node = 4 GiB/node host KV pool -> 8 GiB across 2 nodes.
    "medium": KVGeometry(
        name="medium",
        num_layers=8,
        tokens_per_block=128,
        head_shards=8,
        heads_per_shard=4,
        head_dim=128,
        dtype="bf16",
        hbm_blocks=64,
        host_blocks=256,
    ),
    # Production-scale Qwen3-32B-like geometry: 32 MiB/block (64 layers x 512 KiB).
    "qwen3_32b": KVGeometry(
        name="qwen3_32b",
        num_layers=64,
        tokens_per_block=128,
        head_shards=8,
        heads_per_shard=2,
        head_dim=128,
        dtype="bf16",
        hbm_blocks=64,
        host_blocks=256,
    ),
}


def make_geometry(
    preset: str = "medium",
    *,
    num_layers: Optional[int] = None,
    hbm_blocks: Optional[int] = None,
    host_blocks: Optional[int] = None,
    tokens_per_block: Optional[int] = None,
    head_shards: Optional[int] = None,
    heads_per_shard: Optional[int] = None,
    dtype: Optional[str] = None,
) -> KVGeometry:
    if preset not in GEOMETRY_PRESETS:
        raise ValueError(
            f"Unknown geometry preset {preset!r}; choose from {sorted(GEOMETRY_PRESETS)}"
        )
    base = GEOMETRY_PRESETS[preset]
    return KVGeometry(
        name=preset,
        num_layers=num_layers if num_layers is not None else base.num_layers,
        tokens_per_block=(
            tokens_per_block if tokens_per_block is not None else base.tokens_per_block
        ),
        head_shards=head_shards if head_shards is not None else base.head_shards,
        heads_per_shard=(
            heads_per_shard if heads_per_shard is not None else base.heads_per_shard
        ),
        head_dim=base.head_dim,
        dtype=dtype if dtype is not None else base.dtype,
        hbm_blocks=hbm_blocks if hbm_blocks is not None else base.hbm_blocks,
        host_blocks=host_blocks if host_blocks is not None else base.host_blocks,
    )


def block_hash(namespace: str, block_id: int) -> bytes:
    """Deterministic 16-byte hash for a logical KV block."""
    return hashlib.blake2b(
        f"{namespace}:{block_id}".encode("utf-8"), digest_size=16
    ).digest()


def prefix_chain_hashes(namespace: str, raw_ids: Sequence[int]) -> List[str]:
    """Hex-encoded 16-byte prefix-chained hashes.

    Chaining each block hash with its parent mirrors vLLM/SGLang prefix hashing
    so identical block IDs at different prefix positions never collide.
    """
    out: List[str] = []
    acc = namespace.encode("utf-8")
    for bid in raw_ids:
        acc = hashlib.blake2b(acc + b":" + str(bid).encode("utf-8"), digest_size=16).digest()
        out.append(acc.hex())
    return out


@dataclasses.dataclass
class KVRequest:
    """One logical LLM request arriving at a serving host."""

    req_id: int
    timestamp_ms: float
    node_id: int
    hashes_hex: List[str]  # Ordered prefix block hashes (16 bytes each, hex).
    input_tokens: int
    output_tokens: int = 128
    phase: str = "main"  # "warm", "main", "burst", "recover", "p2d_prefill", "p2d_decode"
    save_misses: bool = True
    read_only: bool = False  # If True, do not save misses (e.g. decode pull or probe).

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KVRequest":
        return cls(**data)


@dataclasses.dataclass
class WorkloadPlan:
    """Complete multi-phase plan for one benchmark workload."""

    name: str
    description: str
    phases: List[Dict]  # Ordered phases: {"name": str, "requests": [KVRequest], ...}
    overlap_compute: bool = False
    verify_sample_rate: float = 0.05


def _pick_node(
    routing: str,
    idx: int,
    session_key: int,
    num_nodes: int,
    rng: random.Random,
) -> int:
    if num_nodes <= 1:
        return 0
    if routing == "round_robin":
        return idx % num_nodes
    if routing == "session_sticky":
        return session_key % num_nodes
    if routing == "random":
        return rng.randrange(num_nodes)
    if routing == "node0_only":
        return 0
    raise ValueError(f"Unknown routing policy: {routing!r}")


def _zipf_weights(n: int, alpha: float) -> List[float]:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(weights)
    return [w / total for w in weights]


def gen_multiturn_tree(
    *,
    num_nodes: int,
    host_blocks: int,
    tokens_per_block: int,
    num_requests: int = 240,
    working_set_ratio: float = 2.0,
    routing: str = "round_robin",
    seed: int = 42,
    run_tag: str = "mt",
) -> WorkloadPlan:
    """Multi-turn chat and agent prefix tree exceeding single-node DRAM.

    Shared system prompts branch into multi-turn conversations whose total unique
    KV blocks equal `working_set_ratio * host_blocks` (so >1.0 overflows one node's
    DRAM, rewarding cluster-wide pooling and good eviction ordering).
    """
    rng = random.Random(seed)
    target_unique_blocks = max(16, int(host_blocks * working_set_ratio))
    system_blocks = max(2, min(6, host_blocks // 16))
    turn_blocks = max(1, min(3, max(1, host_blocks // 48)))
    turns_per_session = 4

    # Size number of system prompts and sessions to hit target_unique_blocks.
    blocks_per_session = turns_per_session * turn_blocks
    num_systems = max(2, min(8, target_unique_blocks // (system_blocks + 3 * blocks_per_session)))
    remaining = max(blocks_per_session, target_unique_blocks - num_systems * system_blocks)
    num_sessions = max(num_systems * 2, remaining // blocks_per_session)

    next_block_id = 1
    systems: List[List[int]] = []
    for _ in range(num_systems):
        ids = list(range(next_block_id, next_block_id + system_blocks))
        next_block_id += system_blocks
        systems.append(ids)

    # Pre-build session turn trees so repeated visits reuse exact prefixes.
    session_turns: List[List[List[str]]] = []
    for s in range(num_sessions):
        sys_ids = systems[s % num_systems]
        chain = list(sys_ids)
        turns: List[List[str]] = []
        for _ in range(turns_per_session):
            new_ids = list(range(next_block_id, next_block_id + turn_blocks))
            next_block_id += turn_blocks
            chain = chain + new_ids
            turns.append(prefix_chain_hashes(f"{run_tag}:mt", chain))
        session_turns.append(turns)

    # Schedule multi-turn progression interleaved across sessions with Zipf session skew.
    session_progress = [0] * num_sessions
    weights = _zipf_weights(num_sessions, 0.9)
    requests: List[KVRequest] = []
    for i in range(num_requests):
        s = rng.choices(range(num_sessions), weights=weights, k=1)[0]
        turn_idx = session_progress[s] % turns_per_session
        session_progress[s] += 1
        hashes = session_turns[s][turn_idx]
        node = _pick_node(routing, i, s, num_nodes, rng)
        requests.append(
            KVRequest(
                req_id=i,
                timestamp_ms=float(i * 25),
                node_id=node,
                hashes_hex=hashes,
                input_tokens=len(hashes) * tokens_per_block,
                output_tokens=128,
                phase="main",
            )
        )

    return WorkloadPlan(
        name="multiturn_tree",
        description=(
            f"Multi-turn prefix tree (~{next_block_id - 1} unique blocks = "
            f"{(next_block_id - 1) / max(1, host_blocks):.2f}x single-node host DRAM, "
            f"routing={routing})"
        ),
        phases=[{"name": "main", "requests": requests}],
    )


def gen_hot_prefix_fanout(
    *,
    num_nodes: int,
    host_blocks: int,
    tokens_per_block: int,
    num_requests: int = 240,
    seed: int = 43,
    run_tag: str = "hf",
) -> WorkloadPlan:
    """Hot-prefix Zipf fan-out across all hosts.

    Phase 1 seeds popular prefixes on Node 0 (simulating initial prefill workers).
    Phase 2 sends skewed Zipf(1.2) traffic across ALL nodes hitting those prefixes
    plus short unique suffixes, exposing cross-node traffic amplification and hot
    peer contention.
    """
    rng = random.Random(seed)
    prefix_blocks = max(4, min(12, host_blocks // 8))
    num_hot = max(4, min(16, max(4, host_blocks // (prefix_blocks * 2))))
    suffix_blocks = max(1, min(2, prefix_blocks // 4))

    warm_requests: List[KVRequest] = []
    hot_prefixes: List[List[int]] = []
    next_id = 1
    for p in range(num_hot):
        ids = list(range(next_id, next_id + prefix_blocks))
        next_id += prefix_blocks
        hot_prefixes.append(ids)
        hashes = prefix_chain_hashes(f"{run_tag}:hf", ids)
        warm_requests.append(
            KVRequest(
                req_id=p,
                timestamp_ms=0.0,
                node_id=0,
                hashes_hex=hashes,
                input_tokens=len(hashes) * tokens_per_block,
                output_tokens=64,
                phase="warm",
            )
        )

    weights = _zipf_weights(num_hot, 1.2)
    main_requests: List[KVRequest] = []
    for i in range(num_requests):
        p = rng.choices(range(num_hot), weights=weights, k=1)[0]
        # Every request shares the hot prefix and adds a small unique suffix.
        suffix = list(range(next_id, next_id + suffix_blocks))
        next_id += suffix_blocks
        hashes = prefix_chain_hashes(f"{run_tag}:hf", hot_prefixes[p] + suffix)
        # Prefer non-origin nodes when multi-node so cross-node fan-out is measured directly.
        if num_nodes > 1:
            node = (i % (num_nodes - 1)) + 1 if (i % 5 != 0) else 0
        else:
            node = 0
        main_requests.append(
            KVRequest(
                req_id=num_hot + i,
                timestamp_ms=float(i * 15),
                node_id=node,
                hashes_hex=hashes,
                input_tokens=len(hashes) * tokens_per_block,
                output_tokens=128,
                phase="main",
            )
        )

    return WorkloadPlan(
        name="hot_prefix_fanout",
        description=(
            f"Zipf(1.2) hot-prefix fan-out ({num_hot} prefixes x {prefix_blocks} blocks "
            f"seeded on node 0, read across {num_nodes} nodes)"
        ),
        phases=[
            {"name": "warm", "requests": warm_requests, "barrier_after": True},
            {"name": "main", "requests": main_requests},
        ],
    )


def gen_burst_write_flood(
    *,
    num_nodes: int,
    host_blocks: int,
    tokens_per_block: int,
    num_requests: int = 240,
    working_set_ratio: float = 2.5,
    seed: int = 44,
    run_tag: str = "bf",
) -> WorkloadPlan:
    """Steady hot reads interrupted by burst waves of brand-new writes on a full cache.

    Tests insert rejection rate, inline vs background eviction stalls, and p99 read
    latency spikes under sustained write pressure.
    """
    rng = random.Random(seed)
    block_len = max(4, min(8, host_blocks // 10))
    # Phase 1: fill ~85% of per-node host DRAM with reusable prompts across all nodes.
    warm_prompts_per_node = max(2, int((host_blocks * 0.85) // block_len))
    next_id = 1
    warm_requests: List[KVRequest] = []
    reusable_by_node: List[List[List[str]]] = [[] for _ in range(num_nodes)]

    req_id = 0
    for node in range(num_nodes):
        for _ in range(warm_prompts_per_node):
            ids = list(range(next_id, next_id + block_len))
            next_id += block_len
            hashes = prefix_chain_hashes(f"{run_tag}:bf", ids)
            reusable_by_node[node].append(hashes)
            warm_requests.append(
                KVRequest(
                    req_id=req_id,
                    timestamp_ms=0.0,
                    node_id=node,
                    hashes_hex=hashes,
                    input_tokens=len(hashes) * tokens_per_block,
                    phase="warm",
                )
            )
            req_id += 1

    # Phase 2: interleave 40% reads of warm reusable prompts with 60% fresh burst writes.
    burst_blocks_target = max(host_blocks, int(host_blocks * num_nodes * working_set_ratio))
    burst_count = max(num_requests, burst_blocks_target // block_len)
    burst_count = min(burst_count, max(num_requests, 400))

    main_requests: List[KVRequest] = []
    for i in range(burst_count):
        node = i % num_nodes
        # Burst waves: 30-request write storm, then 20-request read probe, repeating.
        in_storm = (i % 50) < 30
        if in_storm and rng.random() < 0.75:
            ids = list(range(next_id, next_id + block_len))
            next_id += block_len
            hashes = prefix_chain_hashes(f"{run_tag}:bf", ids)
            phase = "burst_write"
        else:
            pool = reusable_by_node[node] if rng.random() < 0.7 else reusable_by_node[(node + 1) % num_nodes]
            hashes = rng.choice(pool)
            phase = "burst_read"
        main_requests.append(
            KVRequest(
                req_id=req_id,
                timestamp_ms=float(i * 10),
                node_id=node,
                hashes_hex=hashes,
                input_tokens=len(hashes) * tokens_per_block,
                phase=phase,
            )
        )
        req_id += 1

    return WorkloadPlan(
        name="burst_write_flood",
        description=(
            f"Full-cache burst write flood (warm ~85% DRAM, then {burst_count} "
            f"interleaved write-storm/read-probe requests)"
        ),
        phases=[
            {"name": "warm", "requests": warm_requests, "barrier_after": True},
            {"name": "burst", "requests": main_requests},
        ],
    )


def gen_p2d_overlap(
    *,
    num_nodes: int,
    host_blocks: int,
    tokens_per_block: int,
    num_requests: int = 120,
    seed: int = 45,
    run_tag: str = "p2d",
) -> WorkloadPlan:
    """Disaggregated Prefill-to-Decode KV handoff with concurrent TPU compute.

    Node 0 produces prefill KV blocks and saves them to the store. After each wave
    barrier, peer nodes (node 1..N-1, or node 0 if single-node) pull those exact
    blocks into HBM while running continuous XLA matmul compute on the TPU chips.
    """
    rng = random.Random(seed)
    wave_size = max(4, min(16, host_blocks // 8))
    block_len = max(4, min(8, max(4, host_blocks // 16)))
    num_waves = max(2, min(8, math.ceil(num_requests / max(1, wave_size * 2))))

    phases: List[Dict] = []
    next_id = 1
    req_id = 0
    for w in range(num_waves):
        prefill_reqs: List[KVRequest] = []
        decode_reqs: List[KVRequest] = []
        for j in range(wave_size):
            # Include a small shared prefix across wave items plus unique context.
            shared = [100000 + (j % 4), 100100 + (j % 4)]
            unique = list(range(next_id, next_id + max(2, block_len - 2)))
            next_id += len(unique)
            hashes = prefix_chain_hashes(f"{run_tag}:p2d", shared + unique)
            prefill_node = 0
            decode_node = (1 + (j % (num_nodes - 1))) if num_nodes > 1 else 0
            prefill_reqs.append(
                KVRequest(
                    req_id=req_id,
                    timestamp_ms=float(j * 10),
                    node_id=prefill_node,
                    hashes_hex=hashes,
                    input_tokens=len(hashes) * tokens_per_block,
                    phase="p2d_prefill",
                    save_misses=True,
                )
            )
            req_id += 1
            decode_reqs.append(
                KVRequest(
                    req_id=req_id,
                    timestamp_ms=float(j * 10),
                    node_id=decode_node,
                    hashes_hex=hashes,
                    input_tokens=len(hashes) * tokens_per_block,
                    phase="p2d_decode",
                    save_misses=False,
                    read_only=True,
                )
            )
            req_id += 1
        phases.append({"name": f"wave{w}_prefill", "requests": prefill_reqs, "barrier_after": True})
        phases.append({"name": f"wave{w}_decode", "requests": decode_reqs, "barrier_after": True})

    return WorkloadPlan(
        name="p2d_overlap",
        description=(
            f"Disaggregated P->D handoff ({num_waves} waves x {wave_size} reqs, "
            f"{block_len} blocks/req) with concurrent XLA compute overlap"
        ),
        phases=phases,
        overlap_compute=True,
    )


def gen_worker_restart(
    *,
    num_nodes: int,
    host_blocks: int,
    tokens_per_block: int,
    num_requests: int = 120,
    seed: int = 46,
    run_tag: str = "wr",
) -> WorkloadPlan:
    """Warm-cache crash and recovery benchmark.

    Phase 1 warms reusable prefixes on the target restart node (node 1 if multi-node,
    else node 0) and peer node 0.
    Phase 2 SIGKILLs and respawns the target worker subprocess (testing tpu-sync
    /dev/shm warm-boot persistence vs Mooncake surviving peer/cluster DRAM).
    Phase 3 replays those exact prefixes on the restarted node and tracks recovery
    hit rate and latency.
    """
    rng = random.Random(seed)
    target_node = 1 if num_nodes > 1 else 0
    block_len = max(4, min(8, host_blocks // 8))
    num_prefixes = max(4, min(int((host_blocks * 0.6) // block_len), 24))

    prefixes: List[List[str]] = []
    next_id = 1
    for _ in range(num_prefixes):
        ids = list(range(next_id, next_id + block_len))
        next_id += block_len
        prefixes.append(prefix_chain_hashes(f"{run_tag}:wr", ids))

    warm_reqs: List[KVRequest] = []
    req_id = 0
    # Warm half on target_node, half on node 0 (when multi-node) so both local
    # persistence (/dev/shm) and surviving peer DRAM (cluster pool) are exercised!
    for idx, hashes in enumerate(prefixes):
        owner = target_node if (num_nodes == 1 or idx % 2 == 0) else 0
        warm_reqs.append(
            KVRequest(
                req_id=req_id,
                timestamp_ms=0.0,
                node_id=owner,
                hashes_hex=hashes,
                input_tokens=len(hashes) * tokens_per_block,
                phase="warm",
            )
        )
        req_id += 1

    recover_count = max(num_prefixes * 2, min(num_requests, 120))
    recover_reqs: List[KVRequest] = []
    for i in range(recover_count):
        hashes = prefixes[i % num_prefixes]
        recover_reqs.append(
            KVRequest(
                req_id=req_id,
                timestamp_ms=float(i * 15),
                node_id=target_node,
                hashes_hex=hashes,
                input_tokens=len(hashes) * tokens_per_block,
                phase="recover",
            )
        )
        req_id += 1

    return WorkloadPlan(
        name="worker_restart",
        description=(
            f"Worker SIGKILL + restart on node {target_node} ({num_prefixes} warm "
            f"prefixes x {block_len} blocks, {recover_count} post-restart requests)"
        ),
        phases=[
            {"name": "warm", "requests": warm_reqs, "barrier_after": True},
            {"name": "restart", "restart_node": target_node, "requests": []},
            {"name": "recover", "requests": recover_reqs},
        ],
    )


def load_fast25_trace(
    trace_path: str,
    *,
    name: str,
    num_nodes: int,
    tokens_per_block: int,
    max_requests: int = 300,
    max_blocks_per_req: int = 32,
    routing: str = "hash_prefix",
    replay_scale: float = 0.0,
    seed: int = 47,
    run_tag: str = "f25",
) -> WorkloadPlan:
    """Load a Mooncake FAST'25 JSONL trace (conversation/toolagent/synthetic)."""
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(f"FAST25 trace not found: {trace_path}")

    rng = random.Random(seed)
    requests: List[KVRequest] = []
    base_ts: Optional[float] = None

    with open(trace_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_requests > 0 and len(requests) >= max_requests:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_ids = obj.get("hash_ids") or []
            if not raw_ids:
                continue
            if max_blocks_per_req > 0 and len(raw_ids) > max_blocks_per_req:
                raw_ids = raw_ids[:max_blocks_per_req]
            ts = float(obj.get("timestamp", 0.0))
            if base_ts is None:
                base_ts = ts
            rel_ts = (ts - base_ts) / replay_scale if replay_scale > 0 else 0.0
            hashes = prefix_chain_hashes(f"{run_tag}:{name}", raw_ids)
            if routing == "hash_prefix":
                # Route by first block hash (models prefix-aware gateway routing).
                node = int(raw_ids[0]) % num_nodes
            else:
                node = _pick_node(routing, idx, int(raw_ids[0]), num_nodes, rng)
            requests.append(
                KVRequest(
                    req_id=len(requests),
                    timestamp_ms=rel_ts,
                    node_id=node,
                    hashes_hex=hashes,
                    input_tokens=int(obj.get("input_length", len(hashes) * tokens_per_block)),
                    output_tokens=int(obj.get("output_length", 128)),
                    phase="main",
                )
            )

    return WorkloadPlan(
        name=name,
        description=(
            f"FAST'25 trace {os.path.basename(trace_path)} ({len(requests)} reqs, "
            f"max {max_blocks_per_req} blocks/req, routing={routing})"
        ),
        phases=[{"name": "main", "requests": requests}],
    )


ALL_WORKLOADS = (
    "multiturn_tree",
    "hot_prefix_fanout",
    "burst_write_flood",
    "p2d_overlap",
    "worker_restart",
    "fast25_conversation",
    "fast25_toolagent",
    "fast25_synthetic",
)

DEFAULT_WORKLOADS = (
    "multiturn_tree",
    "hot_prefix_fanout",
    "burst_write_flood",
    "p2d_overlap",
    "worker_restart",
    "fast25_conversation",
)


def build_workload(
    name: str,
    *,
    num_nodes: int,
    geometry: KVGeometry,
    trace_dir: str,
    num_requests: int = 200,
    working_set_ratio: float = 2.0,
    routing: str = "round_robin",
    replay_scale: float = 0.0,
    max_blocks_per_req: int = 24,
    seed: int = 42,
    run_tag: str = "run",
) -> WorkloadPlan:
    if name == "multiturn_tree":
        return gen_multiturn_tree(
            num_nodes=num_nodes,
            host_blocks=geometry.host_blocks,
            tokens_per_block=geometry.tokens_per_block,
            num_requests=num_requests,
            working_set_ratio=working_set_ratio,
            routing=routing,
            seed=seed,
            run_tag=run_tag,
        )
    if name == "hot_prefix_fanout":
        return gen_hot_prefix_fanout(
            num_nodes=num_nodes,
            host_blocks=geometry.host_blocks,
            tokens_per_block=geometry.tokens_per_block,
            num_requests=num_requests,
            seed=seed,
            run_tag=run_tag,
        )
    if name == "burst_write_flood":
        return gen_burst_write_flood(
            num_nodes=num_nodes,
            host_blocks=geometry.host_blocks,
            tokens_per_block=geometry.tokens_per_block,
            num_requests=num_requests,
            working_set_ratio=max(working_set_ratio, 2.2),
            seed=seed,
            run_tag=run_tag,
        )
    if name == "p2d_overlap":
        return gen_p2d_overlap(
            num_nodes=num_nodes,
            host_blocks=geometry.host_blocks,
            tokens_per_block=geometry.tokens_per_block,
            num_requests=num_requests,
            seed=seed,
            run_tag=run_tag,
        )
    if name == "worker_restart":
        return gen_worker_restart(
            num_nodes=num_nodes,
            host_blocks=geometry.host_blocks,
            tokens_per_block=geometry.tokens_per_block,
            num_requests=num_requests,
            seed=seed,
            run_tag=run_tag,
        )
    fast25_files = {
        "fast25_conversation": "conversation_trace.jsonl",
        "fast25_toolagent": "toolagent_trace.jsonl",
        "fast25_synthetic": "synthetic_trace.jsonl",
    }
    if name in fast25_files:
        path = os.path.join(trace_dir, fast25_files[name])
        return load_fast25_trace(
            path,
            name=name,
            num_nodes=num_nodes,
            tokens_per_block=geometry.tokens_per_block,
            max_requests=num_requests,
            max_blocks_per_req=max_blocks_per_req,
            routing=routing if routing != "round_robin" else "hash_prefix",
            replay_scale=replay_scale,
            seed=seed,
            run_tag=run_tag,
        )
    raise ValueError(f"Unknown workload {name!r}; choose from {ALL_WORKLOADS}")
