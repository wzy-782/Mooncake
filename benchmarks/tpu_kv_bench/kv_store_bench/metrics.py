#!/usr/bin/env python3
"""Production metrics collection, aggregation, and comparison tables for Tier 2."""

from __future__ import annotations

import dataclasses
import math
import os
import resource
import statistics
from typing import Any, Dict, List, Optional, Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(values, 50.0), 3),
        "p95": round(percentile(values, 95.0), 3),
        "p99": round(percentile(values, 99.0), 3),
        "p999": round(percentile(values, 99.9), 3),
    }


def read_proc_rss_bytes() -> tuple[int, int]:
    """Returns (current_rss_bytes, peak_rss_bytes) from /proc/self/status."""
    rss = 0
    hwm = 0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) * 1024
    except OSError:
        pass
    return rss, hwm


def snapshot_resources() -> Dict[str, float]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss, hwm = read_proc_rss_bytes()
    return {
        "cpu_user_s": float(ru.ru_utime),
        "cpu_sys_s": float(ru.ru_stime),
        "rss_bytes": float(rss),
        "hwm_bytes": float(hwm),
    }


@dataclasses.dataclass
class RequestMetric:
    req_id: int
    node_id: int
    phase: str
    seq_idx: int
    total_blocks: int
    hbm_hit_blocks: int
    local_dram_hit_blocks: int
    remote_dram_hit_blocks: int
    storage_hit_blocks: int
    miss_blocks: int
    raw_store_hit_blocks: int
    lookup_ms: float
    load_ms: float
    save_ms: float
    wall_ms: float
    load_bytes: int
    remote_load_bytes: int
    save_bytes: int
    insert_rejected_blocks: int
    save_failed_blocks: int
    load_failed_blocks: int
    modeled_prefill_ms: float
    net_ttft_saved_ms: float
    effective_hit_blocks: int
    verified_blocks: int = 0
    verify_failures: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RequestMetric":
        return cls(**data)


@dataclasses.dataclass
class ComputeOverlapSummary:
    baseline_step_ms: Dict[str, float]
    active_step_ms: Dict[str, float]
    p50_inflation_pct: float
    p99_inflation_pct: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def summarize_compute_overlap(
    baseline_ms: Sequence[float], active_ms: Sequence[float]
) -> Optional[Dict[str, Any]]:
    if not baseline_ms or not active_ms:
        return None
    base = latency_summary(baseline_ms)
    act = latency_summary(active_ms)
    p50_inf = ((act["p50"] - base["p50"]) / base["p50"] * 100.0) if base["p50"] > 0 else 0.0
    p99_inf = ((act["p99"] - base["p99"]) / base["p99"] * 100.0) if base["p99"] > 0 else 0.0
    return ComputeOverlapSummary(
        baseline_step_ms=base,
        active_step_ms=act,
        p50_inflation_pct=round(p50_inf, 2),
        p99_inflation_pct=round(p99_inf, 2),
    ).to_dict()


def build_recovery_curve(
    records: Sequence[RequestMetric], window_size: int = 10
) -> List[Dict[str, Any]]:
    """Time-ordered bucketed hit-rate & latency curve for restart or burst phases."""
    ordered = sorted(records, key=lambda r: (r.seq_idx, r.req_id))
    buckets: List[Dict[str, Any]] = []
    if not ordered:
        return buckets
    step = max(1, window_size)
    for start in range(0, len(ordered), step):
        chunk = ordered[start : start + step]
        total_blks = sum(r.total_blocks for r in chunk)
        hit_blks = sum(
            r.hbm_hit_blocks
            + r.local_dram_hit_blocks
            + r.remote_dram_hit_blocks
            + r.storage_hit_blocks
            for r in chunk
        )
        local_blks = sum(r.local_dram_hit_blocks for r in chunk)
        remote_blks = sum(r.remote_dram_hit_blocks for r in chunk)
        ext_hits = local_blks + remote_blks + sum(r.storage_hit_blocks for r in chunk)
        rejected = sum(r.insert_rejected_blocks for r in chunk)
        misses = sum(r.miss_blocks for r in chunk)
        loads = [r.load_ms for r in chunk if r.load_bytes > 0]
        buckets.append(
            {
                "window_start": start,
                "window_end": start + len(chunk) - 1,
                "requests": len(chunk),
                "prefix_hit_rate": round(hit_blks / total_blks, 4) if total_blks else 0.0,
                "external_hit_rate": round(ext_hits / total_blks, 4) if total_blks else 0.0,
                "local_dram_hit_rate": round(local_blks / total_blks, 4) if total_blks else 0.0,
                "remote_dram_hit_rate": round(remote_blks / total_blks, 4) if total_blks else 0.0,
                "insert_reject_rate": round(rejected / misses, 4) if misses else 0.0,
                "load_p99_ms": round(percentile(loads, 99.0), 3) if loads else 0.0,
            }
        )
    return buckets


def aggregate_workload_metrics(
    *,
    backend: str,
    workload: str,
    records: Sequence[RequestMetric],
    wall_elapsed_s: float,
    node_stats: Sequence[Dict[str, Any]],
    block_total_bytes: int,
    include_warm_in_headline: bool = False,
) -> Dict[str, Any]:
    """Aggregate cluster-wide metrics across all worker nodes for one workload."""
    if include_warm_in_headline:
        measured = list(records)
    else:
        non_warm = [r for r in records if r.phase != "warm"]
        measured = non_warm if non_warm else list(records)

    total_reqs = len(measured)
    total_blocks = sum(r.total_blocks for r in measured)
    hbm_hits = sum(r.hbm_hit_blocks for r in measured)
    local_dram_hits = sum(r.local_dram_hit_blocks for r in measured)
    remote_dram_hits = sum(r.remote_dram_hit_blocks for r in measured)
    storage_hits = sum(r.storage_hit_blocks for r in measured)
    misses = sum(r.miss_blocks for r in measured)
    raw_store_hits = sum(r.raw_store_hit_blocks for r in measured)
    prefix_hits = hbm_hits + local_dram_hits + remote_dram_hits + storage_hits
    external_hits = local_dram_hits + remote_dram_hits + storage_hits
    non_hbm_blocks = max(0, total_blocks - hbm_hits)
    effective_hits = sum(r.effective_hit_blocks for r in measured)

    load_bytes = sum(r.load_bytes for r in measured)
    remote_load_bytes = sum(r.remote_load_bytes for r in measured)
    save_bytes = sum(r.save_bytes for r in measured)
    total_bytes_all = sum(r.load_bytes + r.save_bytes for r in records)

    insert_rejected = sum(r.insert_rejected_blocks for r in measured)
    save_failed = sum(r.save_failed_blocks for r in measured)
    load_failed = sum(r.load_failed_blocks for r in measured)
    verified_blocks = sum(r.verified_blocks for r in records)
    verify_failures = sum(r.verify_failures for r in records)

    lookup_lat = latency_summary([r.lookup_ms for r in measured])
    load_lat = latency_summary([r.load_ms for r in measured if r.load_bytes > 0])
    save_lat = latency_summary([r.save_ms for r in measured if r.save_bytes > 0])
    wall_lat = latency_summary([r.wall_ms for r in measured])

    total_load_s = sum(r.load_ms for r in measured) / 1000.0
    total_save_s = sum(r.save_ms for r in measured) / 1000.0
    load_gbps = (load_bytes / 1e9) / total_load_s if total_load_s > 0 else 0.0
    save_gbps = (save_bytes / 1e9) / total_save_s if total_save_s > 0 else 0.0
    cluster_load_gbps = (load_bytes / 1e9) / wall_elapsed_s if wall_elapsed_s > 0 else 0.0
    cluster_save_gbps = (save_bytes / 1e9) / wall_elapsed_s if wall_elapsed_s > 0 else 0.0

    total_modeled_prefill_ms = sum(r.modeled_prefill_ms for r in measured)
    total_net_ttft_saved_ms = sum(r.net_ttft_saved_ms for r in measured)
    net_ttft_saving_ratio = (
        total_net_ttft_saved_ms / total_modeled_prefill_ms
        if total_modeled_prefill_ms > 0
        else 0.0
    )

    total_cpu_s = sum(float(n.get("cpu_s", 0.0)) for n in node_stats)
    total_alloc_host_bytes = sum(int(n.get("allocated_host_bytes", 0)) for n in node_stats)
    total_nominal_host_bytes = sum(int(n.get("nominal_host_bytes", 0)) for n in node_stats)
    peak_rss_bytes = max((int(n.get("peak_rss_bytes", 0)) for n in node_stats), default=0)
    mem_amplification = (
        total_alloc_host_bytes / total_nominal_host_bytes
        if total_nominal_host_bytes > 0
        else 0.0
    )
    cpu_cores_per_gbps = (
        total_cpu_s / (total_bytes_all / 1e9) if total_bytes_all > 0 else 0.0
    )
    net_amplification = (
        remote_load_bytes / load_bytes if load_bytes > 0 else 0.0
    )

    # Merge compute overlap samples across nodes if present.
    base_steps: List[float] = []
    active_steps: List[float] = []
    for n in node_stats:
        base_steps.extend(n.get("compute_baseline_ms", []))
        active_steps.extend(n.get("compute_active_ms", []))
    overlap = summarize_compute_overlap(base_steps, active_steps)

    # Phase-specific timelines for restart / burst workloads.
    recover_records = [r for r in records if r.phase == "recover"]
    burst_records = [r for r in records if r.phase in ("burst_read", "burst_write")]
    timeline: List[Dict[str, Any]] = []
    if recover_records:
        timeline = build_recovery_curve(recover_records, window_size=max(4, len(recover_records) // 6))
    elif burst_records:
        timeline = build_recovery_curve(burst_records, window_size=max(10, len(burst_records) // 6))

    # Per-phase breakdown.
    phases_seen = sorted({r.phase for r in records})
    per_phase: Dict[str, Any] = {}
    for ph in phases_seen:
        ph_recs = [r for r in records if r.phase == ph]
        ph_blks = sum(r.total_blocks for r in ph_recs)
        ph_hits = sum(
            r.hbm_hit_blocks + r.local_dram_hit_blocks + r.remote_dram_hit_blocks + r.storage_hit_blocks
            for r in ph_recs
        )
        per_phase[ph] = {
            "requests": len(ph_recs),
            "prefix_hit_rate": round(ph_hits / ph_blks, 4) if ph_blks else 0.0,
            "local_dram_hits": sum(r.local_dram_hit_blocks for r in ph_recs),
            "remote_dram_hits": sum(r.remote_dram_hit_blocks for r in ph_recs),
            "misses": sum(r.miss_blocks for r in ph_recs),
            "insert_rejected": sum(r.insert_rejected_blocks for r in ph_recs),
            "load_p99_ms": round(percentile([r.load_ms for r in ph_recs if r.load_bytes > 0], 99.0), 3),
            "save_p99_ms": round(percentile([r.save_ms for r in ph_recs if r.save_bytes > 0], 99.0), 3),
        }

    return {
        "backend": backend,
        "workload": workload,
        "requests": total_reqs,
        "wall_elapsed_s": round(wall_elapsed_s, 3),
        "qps": round(total_reqs / wall_elapsed_s, 2) if wall_elapsed_s > 0 else 0.0,
        "block_total_bytes": block_total_bytes,
        "cache": {
            "total_blocks": total_blocks,
            "prefix_hit_rate": round(prefix_hits / total_blocks, 4) if total_blocks else 0.0,
            "external_hit_rate": round(external_hits / non_hbm_blocks, 4) if non_hbm_blocks else 0.0,
            "raw_store_hit_rate": round(raw_store_hits / non_hbm_blocks, 4) if non_hbm_blocks else 0.0,
            "effective_hit_rate": round(effective_hits / total_blocks, 4) if total_blocks else 0.0,
            "hbm_hit_rate": round(hbm_hits / total_blocks, 4) if total_blocks else 0.0,
            "local_dram_hit_rate": round(local_dram_hits / total_blocks, 4) if total_blocks else 0.0,
            "remote_dram_hit_rate": round(remote_dram_hits / total_blocks, 4) if total_blocks else 0.0,
            "storage_hit_rate": round(storage_hits / total_blocks, 4) if total_blocks else 0.0,
            "miss_rate": round(misses / total_blocks, 4) if total_blocks else 0.0,
            "net_ttft_saved_ms_avg": round(total_net_ttft_saved_ms / total_reqs, 3) if total_reqs else 0.0,
            "net_ttft_saving_ratio": round(net_ttft_saving_ratio, 4),
            "insert_rejected_blocks": insert_rejected,
            "insert_reject_rate": round(insert_rejected / misses, 4) if misses else 0.0,
            "save_failed_blocks": save_failed,
            "load_failed_blocks": load_failed,
        },
        "latency_ms": {
            "lookup": lookup_lat,
            "load": load_lat,
            "save": save_lat,
            "wall": wall_lat,
        },
        "throughput": {
            "load_bytes": load_bytes,
            "remote_load_bytes": remote_load_bytes,
            "save_bytes": save_bytes,
            "per_op_load_gbps": round(load_gbps, 3),
            "per_op_save_gbps": round(save_gbps, 3),
            "cluster_load_gbps": round(cluster_load_gbps, 3),
            "cluster_save_gbps": round(cluster_save_gbps, 3),
        },
        "efficiency": {
            "allocated_host_bytes": total_alloc_host_bytes,
            "nominal_host_bytes": total_nominal_host_bytes,
            "memory_amplification": round(mem_amplification, 3),
            "peak_rss_gib": round(peak_rss_bytes / (1024**3), 3),
            "cpu_seconds": round(total_cpu_s, 3),
            "cpu_s_per_gb": round(cpu_cores_per_gbps, 4),
            "remote_traffic_ratio": round(net_amplification, 4),
        },
        "integrity": {
            "verified_blocks": verified_blocks,
            "verify_failures": verify_failures,
        },
        "compute_overlap": overlap,
        "timeline": timeline,
        "per_phase": per_phase,
    }


def format_comparison_table(all_results: Sequence[Dict[str, Any]]) -> str:
    """Format a scannable side-by-side comparison table grouped by workload."""
    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for res in all_results:
        by_workload.setdefault(res["workload"], []).append(res)

    lines: List[str] = []
    for workload, runs in by_workload.items():
        lines.append(f"\n=== Workload: {workload} ===")
        backends = [r["backend"] for r in runs]
        header = f"{'Metric':<34}" + "".join(f"{b:>18}" for b in backends)
        lines.append(header)
        lines.append("-" * len(header))

        def row(label: str, getter):
            vals = "".join(f"{getter(r):>18}" for r in runs)
            lines.append(f"{label:<34}{vals}")

        row("Prefix hit rate (all tiers)", lambda r: f"{r['cache']['prefix_hit_rate']*100:6.2f}%")
        row("  Local DRAM hit rate", lambda r: f"{r['cache']['local_dram_hit_rate']*100:6.2f}%")
        row("  Remote DRAM hit rate", lambda r: f"{r['cache']['remote_dram_hit_rate']*100:6.2f}%")
        row("  External hit rate (non-HBM)", lambda r: f"{r['cache']['external_hit_rate']*100:6.2f}%")
        row("  Effective hit rate (beats prefill)", lambda r: f"{r['cache']['effective_hit_rate']*100:6.2f}%")
        row("Net TTFT saved / req (ms)", lambda r: f"{r['cache']['net_ttft_saved_ms_avg']:8.2f}")
        row("Insert reject rate", lambda r: f"{r['cache']['insert_reject_rate']*100:6.2f}%")
        row("Load / Save fail blocks", lambda r: f"{r['cache']['load_failed_blocks']}/{r['cache']['save_failed_blocks']}")
        row("Lookup latency p50 / p99 (ms)", lambda r: f"{r['latency_ms']['lookup']['p50']:.2f}/{r['latency_ms']['lookup']['p99']:.2f}")
        row("Load latency p50 / p99 (ms)", lambda r: f"{r['latency_ms']['load']['p50']:.2f}/{r['latency_ms']['load']['p99']:.2f}")
        row("Save latency p50 / p99 (ms)", lambda r: f"{r['latency_ms']['save']['p50']:.2f}/{r['latency_ms']['save']['p99']:.2f}")
        row("Load / Save per-op BW (GB/s)", lambda r: f"{r['throughput']['per_op_load_gbps']:.2f}/{r['throughput']['per_op_save_gbps']:.2f}")
        row("Cluster QPS", lambda r: f"{r['qps']:8.2f}")
        row("Remote traffic ratio (net amp)", lambda r: f"{r['efficiency']['remote_traffic_ratio']*100:6.2f}%")
        row("Memory amplification", lambda r: f"{r['efficiency']['memory_amplification']:7.2f}x")
        row("Host CPU cost (cpu-s / GB)", lambda r: f"{r['efficiency']['cpu_s_per_gb']:8.3f}")
        row("Integrity verified / failures", lambda r: f"{r['integrity']['verified_blocks']}/{r['integrity']['verify_failures']}")

        if any(r.get("compute_overlap") for r in runs):
            row(
                "XLA compute p99 jitter",
                lambda r: (
                    f"{r['compute_overlap']['p99_inflation_pct']:+6.1f}%"
                    if r.get("compute_overlap")
                    else "n/a"
                ),
            )

        for r in runs:
            if r.get("timeline"):
                pts = " ".join(
                    f"w{b['window_start']}:{b['external_hit_rate']*100:.0f}%"
                    for b in r["timeline"]
                )
                lines.append(f"  [{r['backend']} timeline ext-hit] {pts}")

    return "\n".join(lines) + "\n"
