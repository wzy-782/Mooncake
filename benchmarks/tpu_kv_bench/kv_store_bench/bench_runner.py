#!/usr/bin/env python3
"""Multi-host trace-driven TPU KV-store benchmark runner (Tier 2).

Compares Mooncake Store (TENT TPU path) against tpu-sync KVCacheStore on identical
multi-host TPU topologies, KV geometries, and LLM prefix traces.

Three modes:
  1. Multi-host CDK (default in tpu_kv_store_bench.yml):
     - Node 0:   bench_runner.py --role coordinator --num-nodes 2 --listen-port 5566 ...
     - Node 1+:  bench_runner.py --role agent --node-id 1 --coordinator HOST0:5566
  2. Single-VM local 2-node TPU test (8-chip VM split into chips 0-3 and 4-7):
     - bench_runner.py --role local --num-nodes 2 --backends mooncake,tpu_sync ...
  3. Fast CPU-only mock test (verifies workloads, restart, metrics, tables):
     - bench_runner.py --role mock --num-nodes 2 --backends mooncake,tpu_sync ...
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure sibling modules are importable regardless of working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics import (
    RequestMetric,
    aggregate_workload_metrics,
    format_comparison_table,
    snapshot_resources,
)
from workloads import (
    ALL_WORKLOADS,
    DEFAULT_WORKLOADS,
    GEOMETRY_PRESETS,
    KVGeometry,
    KVRequest,
    build_workload,
    make_geometry,
)


# ---------------------------------------------------------------------------
# Framing helpers (newline-delimited JSON over TCP)
# ---------------------------------------------------------------------------


class JsonConn:
    """Line-delimited JSON channel over a TCP socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._r = sock.makefile("r", encoding="utf-8")
        self._w = sock.makefile("w", encoding="utf-8")
        self._lock = threading.Lock()

    def send(self, msg: Dict[str, Any]) -> None:
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        with self._lock:
            self._w.write(line)
            self._w.flush()

    def recv(self) -> Dict[str, Any]:
        line = self._r.readline()
        if not line:
            raise ConnectionError("peer closed control connection")
        return json.loads(line)

    def close(self) -> None:
        for obj in (self._w, self._r, self.sock):
            try:
                obj.close()
            except Exception:
                pass


def connect_with_retry(host: str, port: int, timeout_s: float = 600.0) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(None)
            return sock
        except OSError as exc:
            last_err = exc
            time.sleep(1.0)
    raise TimeoutError(f"Could not connect to {host}:{port} within {timeout_s}s ({last_err})")


def wait_port_open(host: str, port: int, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Service on {host}:{port} did not become ready within {timeout_s}s")


def detect_reachable_ip(peer_host: Optional[str] = None) -> str:
    if peer_host and peer_host not in ("127.0.0.1", "localhost"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((peer_host, 80))
            return str(s.getsockname()[0])
        except OSError:
            pass
        finally:
            s.close()
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


def pick_free_port(bind_ip: str = "0.0.0.0") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((bind_ip, 0))
        return int(s.getsockname()[1])


def find_binary(name: str, candidates: Sequence[str]) -> str:
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        f"Could not find executable {name!r}; searched: {list(candidates)}"
    )


def default_trace_dir() -> str:
    for cand in (
        os.path.join(os.path.dirname(_HERE), "traces"),
        os.path.join(_HERE, "traces"),
        os.path.abspath(os.path.join(_HERE, "..", "..", "..", "FAST25-release", "traces")),
        "/opt/mooncake-bench/traces",
    ):
        if os.path.isdir(cand):
            return cand
    return os.path.abspath(os.path.join(_HERE, "..", "..", "..", "FAST25-release", "traces"))


def cleanup_shm_prefix(shm_key: str) -> None:
    if not shm_key:
        return
    for path in glob.glob(f"/dev/shm/{shm_key}*"):
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Worker process (runs on real TPU chips or in mock mode)
# ---------------------------------------------------------------------------


def execute_request(
    req: KVRequest,
    *,
    seq_idx: int,
    backend,
    hbm_pool,
    geometry: KVGeometry,
    prefill_tok_per_s: float,
    verify_rate: float,
    rng: random.Random,
) -> RequestMetric:
    """Execute one prefix-caching LLM request against Tier-0 HBM + external store."""
    t_wall0 = time.perf_counter()
    hashes = req.hashes_hex
    total_blocks = len(hashes)

    # Tier 0: on-device HBM prefix lookup.
    pinned_hbm_slots, hbm_hits = hbm_pool.lookup_prefix(hashes)
    remaining = hashes[hbm_hits:]

    lookup_ms = 0.0
    load_ms = 0.0
    save_ms = 0.0
    local_dram_hits = 0
    remote_dram_hits = 0
    storage_hits = 0
    raw_store_hits = 0
    load_bytes = 0
    remote_load_bytes = 0
    save_bytes = 0
    insert_rejected = 0
    save_failed = 0
    load_failed = 0
    verified_blocks = 0
    verify_failures = 0

    acquired_load_slots: List[int] = []
    acquired_miss_slots: List[int] = []
    try:
        # Tier 1+: external store prefix lookup for blocks not in HBM.
        lookup_hits = []
        if remaining:
            lres = backend.lookup(remaining)
            lookup_ms = lres.latency_ms
            lookup_hits = lres.hits
            raw_store_hits = lres.raw_hit_count

        # Load matched prefix blocks from external store into HBM slots.
        loaded_hashes: List[str] = []
        loaded_slots: List[int] = []
        if lookup_hits:
            try:
                acquired_load_slots = hbm_pool.acquire_slots(len(lookup_hits))
            except Exception:
                backend.release_hits(lookup_hits)
                raise
            tr = backend.load(lookup_hits, acquired_load_slots)
            load_ms = tr.latency_ms
            load_bytes = tr.bytes_transferred
            remote_load_bytes = tr.remote_bytes
            load_failed = tr.failed_count

            for ok_i in tr.ok_indices:
                hit = lookup_hits[ok_i]
                slot = acquired_load_slots[ok_i]
                loaded_hashes.append(hit.hash_hex)
                loaded_slots.append(slot)
                if hit.tier == "local_dram":
                    local_dram_hits += 1
                elif hit.tier == "remote_dram":
                    remote_dram_hits += 1
                else:
                    storage_hits += 1

            # Sampled HBM integrity check on loaded blocks (always verify the
            # first loaded request in a phase when verify_rate > 0).
            if loaded_slots and verify_rate > 0 and (seq_idx == 0 or rng.random() < verify_rate):
                check_slot = loaded_slots[0]
                verified_blocks += 1
                if not backend.verify_slot_nonzero(check_slot):
                    verify_failures += 1

            if loaded_hashes:
                hbm_pool.commit_slots(loaded_hashes, loaded_slots)

        # Contiguous external hits = number of prefix hits that loaded cleanly.
        ext_hits_ok = local_dram_hits + remote_dram_hits + storage_hits
        misses = remaining[len(lookup_hits) :]

        # Save missed blocks (newly prefilled KV) back to external store.
        if misses and req.save_misses and not req.read_only:
            acquired_miss_slots = hbm_pool.acquire_slots(len(misses))
            sr = backend.save(misses, acquired_miss_slots)
            save_ms = sr.latency_ms
            save_bytes = sr.bytes_transferred
            insert_rejected = sr.rejected_count
            save_failed = sr.failed_count
            hbm_pool.commit_slots(misses, acquired_miss_slots)

    finally:
        if acquired_miss_slots:
            hbm_pool.release_slots(acquired_miss_slots)
        if acquired_load_slots:
            hbm_pool.release_slots(acquired_load_slots)
        if pinned_hbm_slots:
            hbm_pool.release_slots(pinned_hbm_slots)

    wall_ms = (time.perf_counter() - t_wall0) * 1000.0
    prefix_hits_total = hbm_hits + local_dram_hits + remote_dram_hits + storage_hits
    ext_hit_tokens = (local_dram_hits + remote_dram_hits + storage_hits) * geometry.tokens_per_block
    all_hit_tokens = prefix_hits_total * geometry.tokens_per_block

    modeled_prefill_ms = (
        (req.input_tokens / prefill_tok_per_s) * 1000.0 if prefill_tok_per_s > 0 else 0.0
    )
    hit_prefill_saved_ms = (
        (all_hit_tokens / prefill_tok_per_s) * 1000.0 if prefill_tok_per_s > 0 else 0.0
    )
    ext_prefill_saved_ms = (
        (ext_hit_tokens / prefill_tok_per_s) * 1000.0 if prefill_tok_per_s > 0 else 0.0
    )
    net_ttft_saved_ms = hit_prefill_saved_ms - (lookup_ms + load_ms)

    # Effective hits: HBM hits always save prefill; external hits only count if
    # lookup + load beat modeled prefill time for those tokens.
    effective_hits = hbm_hits
    if ext_hit_tokens > 0 and (lookup_ms + load_ms) < ext_prefill_saved_ms:
        effective_hits += local_dram_hits + remote_dram_hits + storage_hits

    return RequestMetric(
        req_id=req.req_id,
        node_id=req.node_id,
        phase=req.phase,
        seq_idx=seq_idx,
        total_blocks=total_blocks,
        hbm_hit_blocks=hbm_hits,
        local_dram_hit_blocks=local_dram_hits,
        remote_dram_hit_blocks=remote_dram_hits,
        storage_hit_blocks=storage_hits,
        miss_blocks=len(misses),
        raw_store_hit_blocks=raw_store_hits,
        lookup_ms=round(lookup_ms, 4),
        load_ms=round(load_ms, 4),
        save_ms=round(save_ms, 4),
        wall_ms=round(wall_ms, 4),
        load_bytes=load_bytes,
        remote_load_bytes=remote_load_bytes,
        save_bytes=save_bytes,
        insert_rejected_blocks=insert_rejected,
        save_failed_blocks=save_failed,
        load_failed_blocks=load_failed,
        modeled_prefill_ms=round(modeled_prefill_ms, 4),
        net_ttft_saved_ms=round(net_ttft_saved_ms, 4),
        effective_hit_blocks=effective_hits,
        verified_blocks=verified_blocks,
        verify_failures=verify_failures,
    )


def worker_main(args: argparse.Namespace) -> None:
    cfg = json.loads(os.environ["KV_BENCH_WORKER_CFG"])
    coord_host, coord_port_s = args.coordinator.rsplit(":", 1)
    coord_port = int(coord_port_s)

    # IMPORTANT: import jax BEFORE mooncake / tpu_sync extensions.
    if cfg["backend"] in ("mooncake", "tpu_sync"):
        import jax  # noqa: F401

    from backends import (
        ComputeOverlapDriver,
        HBMBlockPool,
        MooncakeStoreBackend,
        TpuSyncStoreBackend,
    )

    geom = KVGeometry(**{k: v for k, v in cfg["geometry"].items() if k in KVGeometry.__dataclass_fields__})
    node_id = int(cfg["node_id"])
    my_ip = cfg["my_ip"]
    generation = int(cfg["generation"])

    t_init0 = time.perf_counter()
    if cfg["backend"] == "mooncake":
        backend = MooncakeStoreBackend(
            geometry=geom,
            node_id=node_id,
            my_ip=my_ip,
            master_addr=cfg["master_addr"],
            metadata_url=cfg["metadata_url"],
            protocol=cfg["mc_protocol"],
            connections=cfg.get("mc_connections"),
            replicas=cfg["mc_replicas"],
            prefer_local=cfg["mc_prefer_local"],
            adapter_path=cfg.get("mc_adapter"),
        )
    elif cfg["backend"] == "tpu_sync":
        backend = TpuSyncStoreBackend(
            geometry=geom,
            node_id=node_id,
            my_ip=my_ip,
            global_registry_addr=cfg["registry_addr"],
            parallelism=cfg["ts_parallelism"],
            enable_global=cfg["ts_enable_global"],
            enable_evict_sweep=cfg["ts_enable_evict_sweep"],
            evict_low_watermark=cfg["ts_evict_low_watermark"],
            evict_high_watermark=cfg["ts_evict_high_watermark"],
            evict_sweep_period_s=cfg["ts_evict_sweep_period_s"],
            enable_shm=cfg["ts_enable_shm"],
            shm_key=cfg.get("ts_shm_key", ""),
            shm_model_uid=cfg.get("ts_shm_model_uid", "kv_store_bench_v1"),
            kv_pool_group=cfg.get("ts_pool_group", "kv_bench_pool"),
            staging_blocks=cfg["ts_staging_blocks"],
        )
    else:
        raise ValueError(f"Unsupported worker backend: {cfg['backend']}")

    init_s = time.perf_counter() - t_init0
    hbm_cache_slots = int(cfg["hbm_cache_blocks"])
    hbm_pool = HBMBlockPool(total_slots=geom.hbm_blocks, cache_slots=hbm_cache_slots)

    overlap_driver: Optional[ComputeOverlapDriver] = None
    if cfg.get("prepare_compute_overlap"):
        overlap_driver = ComputeOverlapDriver(backend.tpu_pool.devices)
        overlap_driver.measure_baseline(steps=25)

    alloc_bytes, nom_bytes = backend.memory_accounting()
    conn = JsonConn(connect_with_retry(coord_host, coord_port, timeout_s=120.0))
    try:
        conn.send(
            {
                "type": "worker_ready",
                "node_id": node_id,
                "generation": generation,
                "init_s": round(init_s, 3),
                "allocated_host_bytes": alloc_bytes,
                "nominal_host_bytes": nom_bytes,
            }
        )

        while True:
            msg = conn.recv()
            op = msg["op"]
            if op == "shutdown":
                conn.send({"type": "worker_bye", "node_id": node_id})
                break
            if op == "clear_hbm":
                hbm_pool.clear()
                conn.send({"type": "ack", "node_id": node_id})
                continue
            if op == "run_phase":
                requests = [KVRequest.from_dict(r) for r in msg["requests"]]
                concurrency = max(1, int(msg.get("concurrency", 1)))
                verify_rate = float(msg.get("verify_rate", 0.05))
                prefill_tok_s = float(msg.get("prefill_tok_per_s", 16000.0))
                replay_paced = bool(msg.get("replay_paced", False))
                run_overlap = bool(msg.get("overlap_compute", False))

                if run_overlap and overlap_driver is None:
                    overlap_driver = ComputeOverlapDriver(backend.tpu_pool.devices)
                    overlap_driver.measure_baseline(steps=25)

                res_before = snapshot_resources()
                if run_overlap and overlap_driver is not None:
                    overlap_driver.start()

                phase_t0 = time.perf_counter()
                records: List[RequestMetric] = []
                if concurrency <= 1 or len(requests) <= 1:
                    rng = random.Random(1000 + node_id)
                    for idx, req in enumerate(requests):
                        if replay_paced and req.timestamp_ms > 0:
                            wait_s = (req.timestamp_ms / 1000.0) - (time.perf_counter() - phase_t0)
                            if wait_s > 0:
                                time.sleep(wait_s)
                        records.append(
                            execute_request(
                                req,
                                seq_idx=idx,
                                backend=backend,
                                hbm_pool=hbm_pool,
                                geometry=geom,
                                prefill_tok_per_s=prefill_tok_s,
                                verify_rate=verify_rate,
                                rng=rng,
                            )
                        )
                else:
                    def _run_one(item: Tuple[int, KVRequest]) -> RequestMetric:
                        idx, req = item
                        if replay_paced and req.timestamp_ms > 0:
                            wait_s = (req.timestamp_ms / 1000.0) - (time.perf_counter() - phase_t0)
                            if wait_s > 0:
                                time.sleep(wait_s)
                        local_rng = random.Random(1000 + node_id * 10000 + idx)
                        return execute_request(
                            req,
                            seq_idx=idx,
                            backend=backend,
                            hbm_pool=hbm_pool,
                            geometry=geom,
                            prefill_tok_per_s=prefill_tok_s,
                            verify_rate=verify_rate,
                            rng=local_rng,
                        )

                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        records = list(pool.map(_run_one, enumerate(requests)))

                phase_elapsed = time.perf_counter() - phase_t0
                active_steps: List[float] = []
                base_steps: List[float] = []
                if run_overlap and overlap_driver is not None:
                    active_steps = overlap_driver.stop()
                    base_steps = list(overlap_driver.baseline_ms)

                res_after = snapshot_resources()
                cpu_s = (res_after["cpu_user_s"] - res_before["cpu_user_s"]) + (
                    res_after["cpu_sys_s"] - res_before["cpu_sys_s"]
                )
                conn.send(
                    {
                        "type": "phase_done",
                        "node_id": node_id,
                        "phase_elapsed_s": phase_elapsed,
                        "records": [r.to_dict() for r in records],
                        "node_stats": {
                            "node_id": node_id,
                            "cpu_s": cpu_s,
                            "allocated_host_bytes": alloc_bytes,
                            "nominal_host_bytes": nom_bytes,
                            "peak_rss_bytes": int(res_after["hwm_bytes"]),
                            "compute_baseline_ms": base_steps,
                            "compute_active_ms": active_steps,
                        },
                    }
                )
            else:
                raise ValueError(f"Unknown worker op: {op!r}")
    finally:
        conn.close()
        backend.close()


# ---------------------------------------------------------------------------
# NodeAgent (one per physical host or simulated local node; manages worker proc)
# ---------------------------------------------------------------------------


def agent_main(args: argparse.Namespace) -> None:
    coord_host, coord_port_s = args.coordinator.rsplit(":", 1)
    coord_port = int(coord_port_s)
    sock = connect_with_retry(coord_host, coord_port, timeout_s=600.0)
    my_ip = args.my_ip or sock.getsockname()[0]
    conn = JsonConn(sock)

    child: Optional[subprocess.Popen] = None
    last_shm_key = ""

    def _stop_child(sigkill: bool = False) -> None:
        nonlocal child
        if child is None:
            return
        if child.poll() is None:
            if sigkill:
                child.kill()
            else:
                child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        child = None

    try:
        conn.send(
            {
                "type": "agent_hello",
                "node_id": int(args.node_id),
                "ip": my_ip,
                "hostname": socket.gethostname(),
            }
        )
        while True:
            msg = conn.recv()
            op = msg["op"]
            if op == "spawn_worker":
                _stop_child(sigkill=False)
                wcfg = dict(msg["worker_cfg"])
                wcfg["my_ip"] = my_ip
                last_shm_key = wcfg.get("ts_shm_key", "") or last_shm_key
                env = dict(os.environ)
                env["KV_BENCH_WORKER_CFG"] = json.dumps(wcfg)
                if not wcfg.get("verbose", False):
                    # Suppress expected OBJECT_NOT_FOUND glog error spam on cache misses.
                    env.setdefault("GLOG_minloglevel", "3")
                    env.setdefault("GLOG_stderrthreshold", "3")
                if args.tpu_chips:
                    env["TPU_VISIBLE_CHIPS"] = args.tpu_chips
                    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = args.tpu_bounds
                    env["TPU_PROCESS_BOUNDS"] = "1,1,1"
                    env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "true"
                    # Avoid libtpu coordinator port collisions between two local
                    # processes sharing one TPU VM.
                    env.setdefault("TPU_MEGASCALE_GRPC_ENABLE_DEBUG_LOGGING", "0")
                cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--role",
                    "worker",
                    "--coordinator",
                    f"{coord_host}:{coord_port}",
                ]
                child = subprocess.Popen(cmd, env=env)
                conn.send({"type": "agent_ack", "node_id": int(args.node_id), "pid": child.pid})
            elif op == "kill_worker":
                _stop_child(sigkill=bool(msg.get("sigkill", True)))
                conn.send({"type": "agent_ack", "node_id": int(args.node_id)})
            elif op == "wait_worker_exit":
                if child is not None:
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    child = None
                conn.send({"type": "agent_ack", "node_id": int(args.node_id)})
            elif op == "cleanup_shm":
                cleanup_shm_prefix(msg.get("shm_key", last_shm_key))
                conn.send({"type": "agent_ack", "node_id": int(args.node_id)})
            elif op == "shutdown":
                _stop_child(sigkill=False)
                if last_shm_key:
                    cleanup_shm_prefix(last_shm_key)
                conn.send({"type": "agent_bye", "node_id": int(args.node_id)})
                break
            else:
                raise ValueError(f"Unknown agent op: {op!r}")
    finally:
        _stop_child(sigkill=True)
        conn.close()


# ---------------------------------------------------------------------------
# Coordinator (runs on Node 0; launches services, orchestrates agents/workers)
# ---------------------------------------------------------------------------


class BackendServiceHandle:
    """Manages mooncake_master or tpu-sync global_registry_server (+ optional store node)."""

    def __init__(self, procs: List[subprocess.Popen], log_files: List[Any], info: Dict[str, Any]):
        self.procs = procs
        self.log_files = log_files
        self.info = info

    def stop(self) -> None:
        for p in reversed(self.procs):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        for f in self.log_files:
            try:
                f.close()
            except Exception:
                pass


def start_backend_services(
    backend: str,
    coord_ip: str,
    args: argparse.Namespace,
    geometry: KVGeometry,
    log_dir: str,
) -> BackendServiceHandle:
    os.makedirs(log_dir, exist_ok=True)
    procs: List[subprocess.Popen] = []
    log_files: List[Any] = []

    if backend == "mooncake":
        master_bin = find_binary(
            "mooncake_master",
            [
                args.mooncake_master_bin,
                os.path.join(os.path.dirname(sys.executable), "mooncake_master"),
                "/opt/mooncake-bench/bin/mooncake_master",
                "/opt/mooncake-bench/site-packages/mooncake/mooncake_master",
                os.path.expanduser("~/.venvs/mooncake/bin/mooncake_master"),
                os.path.expanduser(
                    "~/.venvs/mooncake/lib/python3.12/site-packages/mooncake/mooncake_master"
                ),
            ],
        )
        rpc_port = pick_free_port("0.0.0.0")
        http_port = pick_free_port("0.0.0.0")
        metrics_port = pick_free_port("0.0.0.0")
        log_f = open(os.path.join(log_dir, "mooncake_master.log"), "w", encoding="utf-8")
        log_files.append(log_f)
        cmd = [
            master_bin,
            "--rpc_address=0.0.0.0",
            f"--rpc_port={rpc_port}",
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            f"--http_metadata_server_port={http_port}",
            f"--metrics_port={metrics_port}",
            f"--eviction_ratio={args.mc_eviction_ratio}",
            f"--eviction_high_watermark_ratio={args.mc_eviction_hwm}",
            f"--default_kv_lease_ttl={args.mc_lease_ttl_ms}",
            f"--client_active_ttl_sec={args.mc_client_ttl_s}",
            f"--client_suspicion_ttl_sec={args.mc_client_ttl_s}",
            "--logtostderr=true",
        ]
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append(p)
        try:
            wait_port_open("127.0.0.1", rpc_port, timeout_s=30.0)
            wait_port_open("127.0.0.1", http_port, timeout_s=30.0)
        except Exception:
            try:
                log_f.flush()
                with open(log_path, "r", encoding="utf-8", errors="replace") as rf:
                    tail = rf.read()[-2000:]
                print(f"[coordinator] mooncake_master failed (rc={proc.poll()}):\n{tail}", file=sys.stderr, flush=True)
            except Exception:
                pass
            raise
        return BackendServiceHandle(
            procs,
            log_files,
            {
                "master_addr": f"{coord_ip}:{rpc_port}",
                "metadata_url": f"http://{coord_ip}:{http_port}/metadata",
            },
        )

    if backend == "tpu_sync":
        tpu_sync_root = os.environ.get("TPU_SYNC_ROOT", os.path.expanduser("~/tpu-sync-1279"))
        reg_bin = find_binary(
            "global_registry_server",
            [
                args.tpusync_registry_bin,
                "/opt/mooncake-bench/bin/global_registry_server",
                os.path.join(
                    tpu_sync_root,
                    "tpu_sync/kv_cache/global_registry/global_registry_server",
                ),
            ],
        )
        reg_port = pick_free_port("0.0.0.0")
        reg_log = open(os.path.join(log_dir, "global_registry.log"), "w", encoding="utf-8")
        log_files.append(reg_log)
        p_reg = subprocess.Popen(
            [reg_bin, f"--port={reg_port}"],
            stdout=reg_log,
            stderr=subprocess.STDOUT,
        )
        procs.append(p_reg)
        wait_port_open("127.0.0.1", reg_port, timeout_s=30.0)

        info: Dict[str, Any] = {
            "registry_addr": f"{coord_ip}:{reg_port}",
            "pool_group": f"kv_bench_{uuid.uuid4().hex[:6]}",
        }
        return BackendServiceHandle(procs, log_files, info)

    raise ValueError(f"Unknown backend: {backend!r}")


def maybe_start_tpusync_store_node(
    svc: BackendServiceHandle,
    coord_ip: str,
    args: argparse.Namespace,
    log_dir: str,
) -> None:
    if args.tpusync_store_node_gb <= 0:
        return
    tpu_sync_root = os.environ.get("TPU_SYNC_ROOT", os.path.expanduser("~/tpu-sync-1279"))
    node_bin = find_binary(
        "kv_cache_host_store_node_main",
        [
            args.tpusync_store_node_bin,
            "/opt/mooncake-bench/bin/kv_cache_host_store_node_main",
            os.path.join(
                tpu_sync_root,
                "tpu_sync/store_node/kv_cache_host_store_node_main",
            ),
        ],
    )
    budget_bytes = int(args.tpusync_store_node_gb * (1024**3))
    log_f = open(os.path.join(log_dir, "tpusync_store_node.log"), "w", encoding="utf-8")
    svc.log_files.append(log_f)
    p = subprocess.Popen(
        [
            node_bin,
            "--job_name=kvbench_store_node",
            f"--global_registry_address={svc.info['registry_addr']}",
            f"--kv_pool_group={svc.info['pool_group']}",
            f"--store_server_ip={coord_ip}",
            "--evict_tier=1",
            f"--dram_budget_bytes={budget_bytes}",
            f"--parallelism={args.tpusync_parallelism}",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    svc.procs.append(p)
    # Give store node a moment to discover KVTransferSpec from serving nodes.
    time.sleep(2.0)


class CoordinatorServer:
    """Accepts connections from NodeAgents and WorkerProcesses on one port."""

    def __init__(self, bind_host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, port))
        self.sock.listen(64)
        self.port = int(self.sock.getsockname()[1])
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self.agents: Dict[int, Tuple[JsonConn, Dict[str, Any]]] = {}
        self.workers: Dict[Tuple[int, int], Tuple[JsonConn, Dict[str, Any]]] = {}
        self._stop = False
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._stop:
            try:
                self.sock.settimeout(1.0)
                cli, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            cli.settimeout(None)
            conn = JsonConn(cli)
            try:
                hello = conn.recv()
            except Exception:
                conn.close()
                continue
            with self._cv:
                mtype = hello.get("type")
                if mtype == "agent_hello":
                    self.agents[int(hello["node_id"])] = (conn, hello)
                elif mtype == "worker_ready":
                    key = (int(hello["node_id"]), int(hello["generation"]))
                    self.workers[key] = (conn, hello)
                else:
                    conn.close()
                    continue
                self._cv.notify_all()

    def wait_for_agents(self, num_nodes: int, timeout_s: float = 600.0) -> Dict[int, Dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while len(self.agents) < num_nodes:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError(
                        f"Expected {num_nodes} node agents, only {sorted(self.agents)} joined"
                    )
                self._cv.wait(timeout=min(2.0, rem))
            return {nid: meta for nid, (_, meta) in self.agents.items()}

    def wait_for_workers(
        self, expected: Sequence[Tuple[int, int]], timeout_s: float = 300.0
    ) -> Dict[int, Tuple[JsonConn, Dict[str, Any]]]:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while not all(k in self.workers for k in expected):
                rem = deadline - time.monotonic()
                if rem <= 0:
                    missing = [k for k in expected if k not in self.workers]
                    raise TimeoutError(f"Workers did not become ready: {missing}")
                self._cv.wait(timeout=min(2.0, rem))
            return {nid: self.workers[(nid, gen)] for nid, gen in expected}

    def close(self) -> None:
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass
        for conn, _ in list(self.agents.values()):
            conn.close()
        for conn, _ in list(self.workers.values()):
            conn.close()


def run_distributed_coordinator(args: argparse.Namespace, local_agent_procs: List[subprocess.Popen]) -> None:
    geom = make_geometry(
        args.geometry,
        num_layers=args.num_layers,
        hbm_blocks=args.hbm_blocks,
        host_blocks=args.host_blocks,
        tokens_per_block=args.tokens_per_block,
        head_shards=args.head_shards,
        heads_per_shard=args.heads_per_shard,
        dtype=args.dtype,
    )
    hbm_cache_blocks = (
        args.hbm_cache_blocks if args.hbm_cache_blocks is not None else max(0, geom.hbm_blocks // 4)
    )
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    trace_dir = args.trace_dir or default_trace_dir()
    os.makedirs(args.output_dir, exist_ok=True)

    coord_ip = args.coord_ip or ("127.0.0.1" if args.role == "local" else detect_reachable_ip())
    server = CoordinatorServer("0.0.0.0", args.listen_port)
    listen_port = server.port

    # Launch local NodeAgent 0 when running in coordinator mode (CDK node 0),
    # or launch ALL N local agents in --role local mode!
    if args.role == "coordinator":
        cmd0 = [
            sys.executable,
            os.path.abspath(__file__),
            "--role",
            "agent",
            "--node-id",
            "0",
            "--coordinator",
            f"127.0.0.1:{listen_port}",
            "--my-ip",
            coord_ip,
        ]
        if args.tpu_chips:
            cmd0 += ["--tpu-chips", args.tpu_chips, "--tpu-bounds", args.tpu_bounds]
        local_agent_procs.append(subprocess.Popen(cmd0))
    elif args.role == "local":
        # Split local TPU VM across N processes cleanly.
        chips_per_node = args.chips_per_node or max(1, 8 // args.num_nodes)
        bounds_map = {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "2,4,1"}
        bounds = bounds_map.get(chips_per_node, "1,1,1")
        for nid in range(args.num_nodes):
            chip_list = ",".join(str(nid * chips_per_node + c) for c in range(chips_per_node))
            cmd_i = [
                sys.executable,
                os.path.abspath(__file__),
                "--role",
                "agent",
                "--node-id",
                str(nid),
                "--coordinator",
                f"127.0.0.1:{listen_port}",
                "--my-ip",
                "127.0.0.1",
                "--tpu-chips",
                chip_list,
                "--tpu-bounds",
                bounds,
            ]
            local_agent_procs.append(subprocess.Popen(cmd_i))

    all_results: List[Dict[str, Any]] = []
    generation = 1
    try:
        print(
            f"[coordinator] listening on {coord_ip}:{listen_port}, waiting for "
            f"{args.num_nodes} node agent(s)...",
            flush=True,
        )
        agent_meta = server.wait_for_agents(args.num_nodes, timeout_s=600.0)
        for nid in sorted(agent_meta):
            m = agent_meta[nid]
            print(
                f"[coordinator] node {nid} ready: {m['hostname']} ({m['ip']})",
                flush=True,
            )
        print(
            f"[coordinator] geometry={geom.name} layers={geom.num_layers} "
            f"dtype={geom.dtype} block={geom.block_total_bytes() / (1024*1024):.2f} MiB "
            f"hbm_blocks={geom.hbm_blocks} (cache={hbm_cache_blocks}) "
            f"host_blocks={geom.host_blocks}/node ({geom.host_pool_bytes() / (1024**3):.2f} GiB/node)",
            flush=True,
        )

        # Build workload plans once so every backend replays identical requests!
        run_tag = uuid.uuid4().hex[:6]
        plans = [
            build_workload(
                wname,
                num_nodes=args.num_nodes,
                geometry=geom,
                trace_dir=trace_dir,
                num_requests=args.num_requests,
                working_set_ratio=args.working_set_ratio,
                routing=args.routing,
                replay_scale=args.replay_scale,
                max_blocks_per_req=args.max_blocks_per_req,
                seed=args.seed,
                run_tag=run_tag,
            )
            for wname in workloads
        ]

        for backend in backends:
            for plan in plans:
                print(
                    f"\n[coordinator] >>> backend={backend} workload={plan.name}: "
                    f"{plan.description}",
                    flush=True,
                )
                svc_log_dir = os.path.join(args.output_dir, f"logs_{backend}_{plan.name}")
                svc = start_backend_services(backend, coord_ip, args, geom, svc_log_dir)
                shm_key = (
                    f"kvbench_shm_{run_tag}_{plan.name}"
                    if (backend == "tpu_sync" and plan.name == "worker_restart")
                    else ""
                )
                try:
                    # Spawn fresh workers across all nodes for clean state & TPU isolation.
                    base_wcfg = {
                        "backend": backend,
                        "geometry": geom.to_dict(),
                        "hbm_cache_blocks": hbm_cache_blocks,
                        "prepare_compute_overlap": plan.overlap_compute or args.overlap_compute,
                        "master_addr": svc.info.get("master_addr", ""),
                        "metadata_url": svc.info.get("metadata_url", ""),
                        "mc_protocol": args.mc_protocol,
                        "mc_connections": args.mc_connections,
                        "mc_replicas": args.mc_replicas,
                        "mc_prefer_local": args.mc_prefer_local,
                        "mc_adapter": args.mc_adapter,
                        "registry_addr": svc.info.get("registry_addr", ""),
                        "ts_pool_group": svc.info.get("pool_group", "kv_bench_pool"),
                        "ts_parallelism": args.tpusync_parallelism,
                        "ts_enable_global": args.num_nodes > 1 or args.tpusync_store_node_gb > 0,
                        "ts_enable_evict_sweep": args.tpusync_evict_sweep,
                        "ts_evict_low_watermark": args.tpusync_evict_low_wm,
                        "ts_evict_high_watermark": args.tpusync_evict_high_wm,
                        "ts_evict_sweep_period_s": args.tpusync_evict_period_s,
                        "ts_enable_shm": bool(shm_key),
                        "ts_shm_key": shm_key,
                        "ts_staging_blocks": args.tpusync_staging_blocks,
                    }

                    node_gens: Dict[int, int] = {}
                    active_workers: Dict[int, Tuple[JsonConn, Dict[str, Any]]] = {}
                    # In single-VM local mode, two processes share one host's libtpu
                    # driver and must not initialize libtpu simultaneously (same rule
                    # as tpu_hbm_bench.py). On multi-host CDK, spawn all pods in parallel.
                    stagger_spawn = args.role == "local"
                    for nid in range(args.num_nodes):
                        node_gens[nid] = generation
                        wcfg = dict(base_wcfg, node_id=nid, generation=generation)
                        agent_conn, _ = server.agents[nid]
                        agent_conn.send({"op": "spawn_worker", "worker_cfg": wcfg})
                        agent_conn.recv()
                        if stagger_spawn:
                            ready_one = server.wait_for_workers(
                                [(nid, node_gens[nid])], timeout_s=180.0
                            )
                            active_workers[nid] = ready_one[nid]
                    generation += 1

                    if not stagger_spawn:
                        active_workers = server.wait_for_workers(
                            [(nid, node_gens[nid]) for nid in range(args.num_nodes)],
                            timeout_s=300.0,
                        )
                    if backend == "tpu_sync" and args.tpusync_store_node_gb > 0:
                        maybe_start_tpusync_store_node(svc, coord_ip, args, svc_log_dir)

                    # Small settle for global registry / master segment discovery.
                    time.sleep(1.0)

                    all_records: List[RequestMetric] = []
                    node_stats_agg: Dict[int, Dict[str, Any]] = {}
                    restart_s: Optional[float] = None
                    measured_wall_s = 0.0

                    for phase in plan.phases:
                        if "restart_node" in phase:
                            rnode = int(phase["restart_node"])
                            print(
                                f"[coordinator]   phase=restart: SIGKILL + respawn node {rnode}",
                                flush=True,
                            )
                            t_r0 = time.perf_counter()
                            # Close stale worker socket on coordinator side, SIGKILL child.
                            old_wconn, _ = active_workers[rnode]
                            old_wconn.close()
                            ac, _ = server.agents[rnode]
                            ac.send({"op": "kill_worker", "sigkill": True})
                            ac.recv()
                            time.sleep(1.0)
                            # Respawn with same shm_key so tpu-sync recovers from /dev/shm.
                            node_gens[rnode] = generation
                            wcfg = dict(base_wcfg, node_id=rnode, generation=generation)
                            generation += 1
                            ac.send({"op": "spawn_worker", "worker_cfg": wcfg})
                            ac.recv()
                            rejoined = server.wait_for_workers([(rnode, node_gens[rnode])], timeout_s=300.0)
                            active_workers[rnode] = rejoined[rnode]
                            restart_s = time.perf_counter() - t_r0
                            print(
                                f"[coordinator]   node {rnode} respawned in {restart_s:.2f}s",
                                flush=True,
                            )
                            time.sleep(0.5)
                            continue

                        phase_name = phase["name"]
                        reqs: List[KVRequest] = phase["requests"]
                        by_node: Dict[int, List[dict]] = {nid: [] for nid in range(args.num_nodes)}
                        for r in reqs:
                            by_node[r.node_id % args.num_nodes].append(r.to_dict())

                        t_ph0 = time.perf_counter()
                        for nid in range(args.num_nodes):
                            wconn, _ = active_workers[nid]
                            wconn.send(
                                {
                                    "op": "run_phase",
                                    "phase_name": phase_name,
                                    "requests": by_node[nid],
                                    "concurrency": args.concurrency,
                                    "verify_rate": args.verify_rate,
                                    "prefill_tok_per_s": args.prefill_tok_per_s,
                                    "replay_paced": args.replay_scale > 0,
                                    "overlap_compute": (
                                        (plan.overlap_compute or args.overlap_compute)
                                        and phase_name != "warm"
                                    ),
                                }
                            )

                        for nid in range(args.num_nodes):
                            wconn, _ = active_workers[nid]
                            reply = wconn.recv()
                            if reply.get("type") != "phase_done":
                                raise RuntimeError(f"Unexpected worker reply from node {nid}: {reply}")
                            for rec_d in reply["records"]:
                                all_records.append(RequestMetric.from_dict(rec_d))
                            ns = reply["node_stats"]
                            if nid not in node_stats_agg:
                                node_stats_agg[nid] = dict(ns)
                            else:
                                cur = node_stats_agg[nid]
                                cur["cpu_s"] += ns["cpu_s"]
                                cur["peak_rss_bytes"] = max(
                                    cur["peak_rss_bytes"], ns["peak_rss_bytes"]
                                )
                                cur["compute_baseline_ms"].extend(ns.get("compute_baseline_ms", []))
                                cur["compute_active_ms"].extend(ns.get("compute_active_ms", []))

                        ph_elapsed = time.perf_counter() - t_ph0
                        if phase_name != "warm":
                            measured_wall_s += ph_elapsed
                        # Allow tpu-sync global registry propagation between warm and main phases.
                        if phase.get("barrier_after"):
                            time.sleep(0.5 if backend == "tpu_sync" else 0.1)

                    summary = aggregate_workload_metrics(
                        backend=backend,
                        workload=plan.name,
                        records=all_records,
                        wall_elapsed_s=max(measured_wall_s, 1e-6),
                        node_stats=list(node_stats_agg.values()),
                        block_total_bytes=geom.block_total_bytes(),
                    )
                    if restart_s is not None:
                        summary["restart_respawn_s"] = round(restart_s, 3)
                    all_results.append(summary)
                    print(
                        f"[coordinator] <<< {backend}/{plan.name}: "
                        f"prefix_hit={summary['cache']['prefix_hit_rate']*100:.1f}% "
                        f"(local={summary['cache']['local_dram_hit_rate']*100:.1f}%, "
                        f"remote={summary['cache']['remote_dram_hit_rate']*100:.1f}%) "
                        f"reject={summary['cache']['insert_reject_rate']*100:.1f}% "
                        f"load_p99={summary['latency_ms']['load']['p99']:.2f}ms "
                        f"qps={summary['qps']:.1f}",
                        flush=True,
                    )

                    # Gracefully shut down all workers before next run.
                    for nid in range(args.num_nodes):
                        wconn, _ = active_workers[nid]
                        try:
                            wconn.send({"op": "shutdown"})
                            wconn.recv()
                        except Exception:
                            pass
                        wconn.close()
                        ac, _ = server.agents[nid]
                        ac.send({"op": "wait_worker_exit"})
                        ac.recv()
                        if shm_key:
                            ac.send({"op": "cleanup_shm", "shm_key": shm_key})
                            ac.recv()
                finally:
                    svc.stop()

        # Print comparison table and persist full JSON + text report.
        table = format_comparison_table(all_results)
        print(table, flush=True)
        out_json = os.path.join(args.output_dir, "tpu_kv_store_bench.json")
        out_txt = os.path.join(args.output_dir, "tpu_kv_store_bench.txt")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "geometry": geom.to_dict(),
                    "num_nodes": args.num_nodes,
                    "backends": backends,
                    "workloads": workloads,
                    "results": all_results,
                },
                f,
                indent=2,
            )
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"[coordinator] saved results to {out_json} and {out_txt}", flush=True)

        # Tell all agents to shut down cleanly.
        for nid in sorted(server.agents):
            ac, _ = server.agents[nid]
            try:
                ac.send({"op": "shutdown"})
                ac.recv()
            except Exception:
                pass
    finally:
        server.close()
        for p in local_agent_procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


# ---------------------------------------------------------------------------
# Fast CPU-only mock runner (validates workloads, restart, metrics, tables)
# ---------------------------------------------------------------------------


def run_mock_mode(args: argparse.Namespace) -> None:
    from backends import HBMBlockPool, MockClusterRegistry, MockStoreBackend

    geom = make_geometry(
        args.geometry,
        num_layers=args.num_layers,
        hbm_blocks=args.hbm_blocks,
        host_blocks=args.host_blocks,
        tokens_per_block=args.tokens_per_block,
        head_shards=args.head_shards,
        heads_per_shard=args.heads_per_shard,
        dtype=args.dtype,
    )
    hbm_cache_blocks = (
        args.hbm_cache_blocks if args.hbm_cache_blocks is not None else max(0, geom.hbm_blocks // 4)
    )
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    trace_dir = args.trace_dir or default_trace_dir()
    os.makedirs(args.output_dir, exist_ok=True)
    run_tag = uuid.uuid4().hex[:6]

    plans = [
        build_workload(
            wname,
            num_nodes=args.num_nodes,
            geometry=geom,
            trace_dir=trace_dir,
            num_requests=args.num_requests,
            working_set_ratio=args.working_set_ratio,
            routing=args.routing,
            replay_scale=args.replay_scale,
            max_blocks_per_req=args.max_blocks_per_req,
            seed=args.seed,
            run_tag=run_tag,
        )
        for wname in workloads
    ]

    all_results: List[Dict[str, Any]] = []
    for bname in backends:
        for plan in plans:
            reg = MockClusterRegistry()
            enable_shm = bname == "tpu_sync" and plan.name == "worker_restart"
            nodes = {
                nid: (
                    MockStoreBackend(
                        mode=bname,
                        geometry=geom,
                        node_id=nid,
                        registry=reg,
                        enable_shm=enable_shm,
                    ),
                    HBMBlockPool(geom.hbm_blocks, hbm_cache_blocks),
                )
                for nid in range(args.num_nodes)
            }
            all_records: List[RequestMetric] = []
            wall_s = 0.0
            res0 = snapshot_resources()
            for phase in plan.phases:
                if "restart_node" in phase:
                    rnode = int(phase["restart_node"])
                    if not enable_shm:
                        reg.clear_node(rnode)
                    nodes[rnode] = (
                        MockStoreBackend(
                            mode=bname,
                            geometry=geom,
                            node_id=rnode,
                            registry=reg,
                            enable_shm=enable_shm,
                        ),
                        HBMBlockPool(geom.hbm_blocks, hbm_cache_blocks),
                    )
                    continue
                t0 = time.perf_counter()
                rng = random.Random(args.seed)
                for idx, req in enumerate(phase["requests"]):
                    nid = req.node_id % args.num_nodes
                    be, hp = nodes[nid]
                    all_records.append(
                        execute_request(
                            req,
                            seq_idx=idx,
                            backend=be,
                            hbm_pool=hp,
                            geometry=geom,
                            prefill_tok_per_s=args.prefill_tok_per_s,
                            verify_rate=args.verify_rate,
                            rng=rng,
                        )
                    )
                if phase["name"] != "warm":
                    wall_s += time.perf_counter() - t0
            res1 = snapshot_resources()
            cpu_s = (res1["cpu_user_s"] - res0["cpu_user_s"]) + (
                res1["cpu_sys_s"] - res0["cpu_sys_s"]
            )
            node_stats = []
            for nid, (be, _) in nodes.items():
                ab, nb = be.memory_accounting()
                node_stats.append(
                    {
                        "node_id": nid,
                        "cpu_s": cpu_s / args.num_nodes,
                        "allocated_host_bytes": ab,
                        "nominal_host_bytes": nb,
                        "peak_rss_bytes": int(res1["hwm_bytes"]),
                    }
                )
            all_results.append(
                aggregate_workload_metrics(
                    backend=bname,
                    workload=plan.name,
                    records=all_records,
                    wall_elapsed_s=max(wall_s, 1e-6),
                    node_stats=node_stats,
                    block_total_bytes=geom.block_total_bytes(),
                )
            )

    table = format_comparison_table(all_results)
    print(table, flush=True)
    out_json = os.path.join(args.output_dir, "tpu_kv_store_bench_mock.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"[mock] saved results to {out_json}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tier 2 multi-host trace-driven TPU KV-store benchmark"
    )
    p.add_argument(
        "--role",
        choices=("coordinator", "agent", "worker", "local", "mock"),
        default="local",
        help="Process role: coordinator/agent (CDK multi-host), local (1 VM), mock (CPU)",
    )
    p.add_argument("--node-id", type=int, default=0)
    p.add_argument("--num-nodes", type=int, default=2)
    p.add_argument("--listen-port", type=int, default=5566)
    p.add_argument("--coordinator", default="127.0.0.1:5566")
    p.add_argument("--coord-ip", default="")
    p.add_argument("--my-ip", default="")
    p.add_argument("--tpu-chips", default="")
    p.add_argument("--tpu-bounds", default="2,2,1")
    p.add_argument(
        "--chips-per-node",
        type=int,
        default=0,
        help="Chips per local node in --role local (0 = auto 8 // num_nodes)",
    )

    # Benchmark matrix
    p.add_argument("--backends", default="mooncake,tpu_sync")
    p.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOADS),
        help=f"Comma-separated subset of {','.join(ALL_WORKLOADS)}",
    )
    p.add_argument("--geometry", choices=sorted(GEOMETRY_PRESETS), default="medium")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--hbm-blocks", type=int, default=None)
    p.add_argument(
        "--hbm-cache-blocks",
        type=int,
        default=None,
        help="Tier-0 HBM prefix cache slots (0 = pure external-store test)",
    )
    p.add_argument("--host-blocks", type=int, default=None)
    p.add_argument("--tokens-per-block", type=int, default=None)
    p.add_argument("--head-shards", type=int, default=None)
    p.add_argument("--heads-per-shard", type=int, default=None)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default=None)

    p.add_argument("--num-requests", type=int, default=160)
    p.add_argument("--working-set-ratio", type=float, default=2.0)
    p.add_argument(
        "--routing",
        choices=("round_robin", "session_sticky", "random", "hash_prefix"),
        default="round_robin",
    )
    p.add_argument("--replay-scale", type=float, default=0.0)
    p.add_argument("--max-blocks-per-req", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--overlap-compute", action="store_true")
    p.add_argument("--verify-rate", type=float, default=0.05)
    p.add_argument("--prefill-tok-per-s", type=float, default=16000.0)
    p.add_argument("--trace-dir", default="")
    p.add_argument("--output-dir", default="/tmp/tpu_kv_store_bench")
    p.add_argument("--seed", type=int, default=42)

    # Mooncake Store tunables
    p.add_argument("--mc-protocol", choices=("tcp", "hp_tcp"), default="hp_tcp")
    p.add_argument("--mc-connections", type=int, default=8)
    p.add_argument("--mc-replicas", type=int, default=1)
    p.add_argument("--mc-prefer-local", action="store_true", default=True)
    p.add_argument("--no-mc-prefer-local", dest="mc_prefer_local", action="store_false")
    p.add_argument("--mc-eviction-ratio", type=float, default=0.1)
    p.add_argument("--mc-eviction-hwm", type=float, default=0.85)
    p.add_argument("--mc-lease-ttl-ms", type=int, default=5000)
    p.add_argument("--mc-client-ttl-s", type=int, default=5)
    p.add_argument("--mc-adapter", default=None)
    p.add_argument("--mooncake-master-bin", default="")

    # tpu-sync KVCacheStore tunables
    p.add_argument("--tpusync-parallelism", type=int, default=8)
    p.add_argument("--tpusync-evict-sweep", action="store_true", default=True)
    p.add_argument("--no-tpusync-evict-sweep", dest="tpusync_evict_sweep", action="store_false")
    p.add_argument("--tpusync-evict-low-wm", type=float, default=0.15)
    p.add_argument("--tpusync-evict-high-wm", type=float, default=0.30)
    p.add_argument("--tpusync-evict-period-s", type=int, default=1)
    p.add_argument("--tpusync-staging-blocks", type=int, default=24)
    p.add_argument("--tpusync-store-node-gb", type=float, default=0.0)
    p.add_argument("--tpusync-registry-bin", default="")
    p.add_argument("--tpusync-store-node-bin", default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.role == "worker":
        worker_main(args)
    elif args.role == "agent":
        agent_main(args)
    elif args.role == "mock":
        run_mock_mode(args)
    else:
        run_distributed_coordinator(args, [])


if __name__ == "__main__":
    main()
