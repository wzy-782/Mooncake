#!/usr/bin/env python3
"""Tier 3 End-to-End vLLM Serving Benchmark Harness (multi-host CDK).

Drives live OpenAI-compatible vLLM endpoints (`/v1/completions` SSE streaming)
across realistic production scenarios, scrapes Prometheus `/metrics` from every
vLLM pod before and after each scenario, computes TTFT/TPOT/ITL/E2EL percentiles,
SLO goodput, and HBM + external prefix cache hit rates, and formats side-by-side
comparison tables across `baseline_hbm`, `tpu_sync`, and `mooncake`.

Zero third-party dependencies (stdlib asyncio + http.client) so it works inside
any vLLM-TPU container image cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import http.client
import json
import math
import os
import random
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Deterministic vocabulary for constructing prompts with exact shared prefixes
# ---------------------------------------------------------------------------

_VOCAB = [
    f"tok{i:04d}" for i in range(4096)
]


def _make_block_text(block_id: int, approx_tokens: int) -> str:
    """Deterministic text chunk for a logical KV block ID.

    Each token like ' tok0123' is ~2-3 BPE tokens; we scale word count so
    shared block_ids produce byte-identical prompt prefixes across requests.
    """
    rng = random.Random(block_id * 1000003 + 17)
    # ~2 tokens per word under Qwen/Llama BPE; scale cleanly.
    n_words = max(8, approx_tokens // 2)
    return " ".join(_VOCAB[rng.randrange(len(_VOCAB))] for _ in range(n_words))


@dataclass
class ServeRequest:
    req_id: int
    prompt: str
    max_tokens: int
    arrival_s: float
    approx_prompt_tokens: int
    phase: str = "main"


@dataclass
class ServeRequestResult:
    req_id: int
    phase: str
    ok: bool
    error: str
    ttft_ms: float
    e2el_ms: float
    tpot_ms: float
    itls_ms: List[float]
    prompt_tokens: int
    output_tokens: int
    meets_slo: bool


# ---------------------------------------------------------------------------
# Production serving workload generators
# ---------------------------------------------------------------------------


def gen_shared_prefix_offload(
    *,
    num_prefixes: int = 8,
    prefix_tokens: int = 8192,
    suffix_tokens: int = 128,
    output_tokens: int = 128,
    num_prompts: int = 80,
    request_rate: float = 2.0,
    seed: int = 42,
) -> List[ServeRequest]:
    """Multi-prefix shared offloading workload sized to overflow HBM and hit DRAM.

    Cycles across `num_prefixes` distinct long prefixes (so HBM LRU evicts older
    prefixes before they return, forcing external host/cluster DRAM loads).
    """
    rng = random.Random(seed)
    chunk_tok = 512
    chunks_per_prefix = max(1, prefix_tokens // chunk_tok)
    prefixes: List[str] = []
    for p in range(num_prefixes):
        parts = [_make_block_text(10000 + p * 1000 + c, chunk_tok) for c in range(chunks_per_prefix)]
        prefixes.append(" ".join(parts))

    reqs: List[ServeRequest] = []
    t = 0.0
    for i in range(num_prompts):
        if i > 0 and request_rate > 0:
            t += rng.expovariate(request_rate)
        p_idx = i % num_prefixes
        suffix = _make_block_text(900000 + i, suffix_tokens)
        reqs.append(
            ServeRequest(
                req_id=i,
                prompt=f"{prefixes[p_idx]} {suffix}",
                max_tokens=output_tokens,
                arrival_s=t,
                approx_prompt_tokens=prefix_tokens + suffix_tokens,
                phase="warm" if i < num_prefixes else "main",
            )
        )
    return reqs


def gen_multiturn_chat(
    *,
    num_sessions: int = 16,
    turns_per_session: int = 4,
    system_tokens: int = 4096,
    turn_tokens: int = 512,
    output_tokens: int = 128,
    request_rate: float = 2.0,
    seed: int = 43,
) -> List[ServeRequest]:
    """Interleaved multi-turn chat sessions sharing 4 system prompts."""
    rng = random.Random(seed)
    num_sys = max(2, min(4, num_sessions // 2))
    sys_texts = [_make_block_text(20000 + s, system_tokens) for s in range(num_sys)]
    session_histories = [sys_texts[s % num_sys] for s in range(num_sessions)]
    session_toks = [system_tokens for _ in range(num_sessions)]

    order: List[int] = []
    for _ in range(turns_per_session):
        batch = list(range(num_sessions))
        rng.shuffle(batch)
        order.extend(batch)

    reqs: List[ServeRequest] = []
    t = 0.0
    for i, s in enumerate(order):
        if i > 0 and request_rate > 0:
            t += rng.expovariate(request_rate)
        turn_text = _make_block_text(300000 + i, turn_tokens)
        session_histories[s] = f"{session_histories[s]} {turn_text}"
        session_toks[s] += turn_tokens
        reqs.append(
            ServeRequest(
                req_id=i,
                prompt=session_histories[s],
                max_tokens=output_tokens,
                arrival_s=t,
                approx_prompt_tokens=session_toks[s],
                phase="warm" if i < num_sessions else "main",
            )
        )
    return reqs


def gen_fast25_replay(
    trace_path: str,
    *,
    max_prompts: int = 80,
    max_blocks: int = 16,
    max_output_tokens: int = 256,
    request_rate: float = 2.0,
    seed: int = 44,
) -> List[ServeRequest]:
    """Replay real Mooncake FAST'25 JSONL prefix-sharing structure as live prompts."""
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(f"FAST25 trace not found: {trace_path}")
    rng = random.Random(seed)
    block_cache: Dict[int, str] = {}
    reqs: List[ServeRequest] = []
    t = 0.0
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(reqs) >= max_prompts:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            hids = obj.get("hash_ids") or []
            if not hids:
                continue
            hids = hids[:max_blocks]
            parts: List[str] = []
            for hid in hids:
                hid_int = int(hid)
                if hid_int not in block_cache:
                    block_cache[hid_int] = _make_block_text(hid_int + 500000, 512)
                parts.append(block_cache[hid_int])
            if reqs and request_rate > 0:
                t += rng.expovariate(request_rate)
            out_tok = max(16, min(max_output_tokens, int(obj.get("output_length", 128))))
            reqs.append(
                ServeRequest(
                    req_id=len(reqs),
                    prompt=" ".join(parts),
                    max_tokens=out_tok,
                    arrival_s=t,
                    approx_prompt_tokens=len(hids) * 512,
                    phase="main",
                )
            )
    return reqs


def gen_disagg_workload(
    *,
    input_tokens: int = 8192,
    output_tokens: int = 256,
    num_prompts: int = 40,
    request_rate: float = 2.0,
    shared_prefix_ratio: float = 0.25,
    seed: int = 45,
) -> List[ServeRequest]:
    """Long-input disaggregated 1P1D serving workload for SLO goodput sweeps."""
    rng = random.Random(seed)
    shared_toks = int(input_tokens * shared_prefix_ratio)
    unique_toks = max(128, input_tokens - shared_toks)
    shared_pool = [
        _make_block_text(70000 + k, shared_toks) if shared_toks > 0 else ""
        for k in range(4)
    ]
    reqs: List[ServeRequest] = []
    t = 0.0
    for i in range(num_prompts):
        if i > 0 and request_rate > 0:
            t += rng.expovariate(request_rate)
        pref = shared_pool[i % len(shared_pool)]
        uniq = _make_block_text(800000 + i, unique_toks)
        prompt = f"{pref} {uniq}".strip()
        reqs.append(
            ServeRequest(
                req_id=i,
                prompt=prompt,
                max_tokens=output_tokens,
                arrival_s=t,
                approx_prompt_tokens=input_tokens,
                phase="main",
            )
        )
    return reqs


# ---------------------------------------------------------------------------
# Prometheus `/metrics` scraper for vLLM servers
# ---------------------------------------------------------------------------


_PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|NaN|Inf|-Inf)$")


def scrape_prometheus_metrics(endpoints: Sequence[str]) -> Dict[str, float]:
    """Sum relevant vLLM Prometheus counters/gauges across all server endpoints."""
    wanted_prefixes = (
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:num_preemptions_total",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_waiting",
    )
    totals: Dict[str, float] = {}
    gauge_counts: Dict[str, int] = {}
    for ep in endpoints:
        if not ep:
            continue
        url = ep if "://" in ep else f"http://{ep}"
        u = urlparse(url)
        host = u.hostname or "127.0.0.1"
        port = int(u.port or 8000)
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5.0)
            conn.request("GET", "/metrics")
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            if resp.status != 200:
                continue
        except Exception:
            continue
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            m = _PROM_LINE_RE.match(line.strip())
            if not m:
                continue
            name, _, val_s = m.groups()
            if not any(name == p or name.startswith(p) for p in wanted_prefixes):
                continue
            try:
                val = float(val_s)
                if math.isnan(val) or math.isinf(val):
                    continue
            except ValueError:
                continue
            totals[name] = totals.get(name, 0.0) + val
            gauge_counts[name] = gauge_counts.get(name, 0) + 1

    # Average usage gauges across pods.
    for g in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        if g in totals and gauge_counts.get(g, 0) > 1:
            totals[g] /= gauge_counts[g]
    return totals


def diff_prometheus_metrics(
    before: Dict[str, float], after: Dict[str, float]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v_after in after.items():
        if k.endswith("_total"):
            out[k] = max(0.0, v_after - before.get(k, 0.0))
        else:
            out[k] = v_after
    # Derive HBM & external prefix cache hit rates when counters exist.
    h_q = out.get("vllm:prefix_cache_queries_total", 0.0)
    h_h = out.get("vllm:prefix_cache_hits_total", 0.0)
    out["hbm_prefix_hit_rate"] = (h_h / h_q) if h_q > 0 else 0.0

    e_q = out.get("vllm:external_prefix_cache_queries_total", 0.0)
    e_h = out.get("vllm:external_prefix_cache_hits_total", 0.0)
    out["external_prefix_hit_rate"] = (e_h / e_q) if e_q > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Async SSE streaming OpenAI `/v1/completions` client
# ---------------------------------------------------------------------------


async def send_streaming_completion(
    host: str,
    port: int,
    model: str,
    req: ServeRequest,
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    timeout_s: float = 300.0,
) -> ServeRequestResult:
    payload = json.dumps(
        {
            "model": model,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")

    raw_http = (
        f"POST /v1/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("utf-8") + payload

    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    prev_token_t: Optional[float] = None
    itls_ms: List[float] = []
    output_tokens = 0
    prompt_tokens = req.approx_prompt_tokens

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=30.0
        )
        try:
            writer.write(raw_http)
            await writer.drain()

            # Read HTTP status line + headers.
            status_line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            if not status_line:
                raise RuntimeError("empty HTTP response")
            parts = status_line.decode("latin-1").strip().split(" ", 2)
            status_code = int(parts[1]) if len(parts) >= 2 else 500
            is_chunked = False
            while True:
                hline = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                hstr = hline.decode("latin-1").lower()
                if "transfer-encoding:" in hstr and "chunked" in hstr:
                    is_chunked = True

            if status_code != 200:
                err_body = await asyncio.wait_for(reader.read(4096), timeout=10.0)
                raise RuntimeError(
                    f"HTTP {status_code}: {err_body.decode('utf-8', errors='replace')[:200]}"
                )

            async def _iter_lines():
                if not is_chunked:
                    while True:
                        line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
                        if not line:
                            break
                        yield line
                else:
                    buf = b""
                    while True:
                        size_line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
                        if not size_line:
                            break
                        size_s = size_line.strip().split(b";", 1)[0]
                        if not size_s:
                            continue
                        n = int(size_s, 16)
                        if n == 0:
                            await reader.readline()
                            break
                        chunk = await asyncio.wait_for(reader.readexactly(n), timeout=timeout_s)
                        await reader.readline()  # trailing CRLF
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            yield line + b"\n"
                    if buf:
                        yield buf

            async for raw_line in _iter_lines():
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_s = line[5:].strip()
                if not data_s or data_s == "[DONE]":
                    continue
                try:
                    evt = json.loads(data_s)
                except json.JSONDecodeError:
                    continue
                usage = evt.get("usage")
                if isinstance(usage, dict):
                    if usage.get("prompt_tokens"):
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens"):
                        output_tokens = int(usage["completion_tokens"])
                choices = evt.get("choices") or []
                if choices and (choices[0].get("text") or choices[0].get("delta")):
                    now = time.perf_counter()
                    if first_token_t is None:
                        first_token_t = now
                    else:
                        itls_ms.append((now - prev_token_t) * 1000.0)  # type: ignore[operator]
                    prev_token_t = now
                    output_tokens = max(output_tokens, len(itls_ms) + 1)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        return ServeRequestResult(
            req_id=req.req_id,
            phase=req.phase,
            ok=False,
            error=str(exc),
            ttft_ms=0.0,
            e2el_ms=(time.perf_counter() - t_start) * 1000.0,
            tpot_ms=0.0,
            itls_ms=[],
            prompt_tokens=prompt_tokens,
            output_tokens=0,
            meets_slo=False,
        )

    t_end = time.perf_counter()
    e2el_ms = (t_end - t_start) * 1000.0
    ttft_ms = ((first_token_t - t_start) * 1000.0) if first_token_t is not None else e2el_ms
    decode_tokens = max(0, output_tokens - 1)
    tpot_ms = ((e2el_ms - ttft_ms) / decode_tokens) if decode_tokens > 0 else 0.0
    meets_slo = (ttft_ms <= ttft_slo_ms) and (tpot_ms <= tpot_slo_ms)
    return ServeRequestResult(
        req_id=req.req_id,
        phase=req.phase,
        ok=True,
        error="",
        ttft_ms=round(ttft_ms, 3),
        e2el_ms=round(e2el_ms, 3),
        tpot_ms=round(tpot_ms, 3),
        itls_ms=[round(x, 3) for x in itls_ms],
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        meets_slo=meets_slo,
    )


async def run_scenario(
    *,
    endpoint: str,
    model: str,
    requests: Sequence[ServeRequest],
    max_concurrency: int,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> Tuple[List[ServeRequestResult], float]:
    url = endpoint if "://" in endpoint else f"http://{endpoint}"
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = int(u.port or 8000)

    sem = asyncio.Semaphore(max(1, max_concurrency))
    t0 = time.perf_counter()

    async def _dispatch(req: ServeRequest) -> ServeRequestResult:
        delay = req.arrival_s - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        async with sem:
            return await send_streaming_completion(
                host, port, model, req, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms
            )

    tasks = [asyncio.create_task(_dispatch(r)) for r in requests]
    results = await asyncio.gather(*tasks)
    elapsed_s = time.perf_counter() - t0
    return list(results), elapsed_s


# ---------------------------------------------------------------------------
# Aggregation & comparison table
# ---------------------------------------------------------------------------


def _pct(vals: Sequence[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] * (1.0 - (idx - lo)) + s[hi] * (idx - lo))


def _dist(vals: Sequence[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": round(statistics.fmean(vals), 2),
        "p50": round(_pct(vals, 50.0), 2),
        "p95": round(_pct(vals, 95.0), 2),
        "p99": round(_pct(vals, 99.0), 2),
    }


def summarize_scenario(
    *,
    label: str,
    scenario: str,
    results: Sequence[ServeRequestResult],
    elapsed_s: float,
    prom_delta: Dict[str, float],
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> Dict[str, Any]:
    non_warm = [r for r in results if r.phase != "warm"]
    measured = non_warm if non_warm else list(results)
    ok = [r for r in measured if r.ok]
    failed = len(measured) - len(ok)
    all_itls = [x for r in ok for x in r.itls_ms]
    slo_ok = sum(1 for r in ok if r.meets_slo)

    total_in = sum(r.prompt_tokens for r in ok)
    total_out = sum(r.output_tokens for r in ok)

    return {
        "label": label,
        "scenario": scenario,
        "requests_total": len(measured),
        "requests_ok": len(ok),
        "requests_failed": failed,
        "elapsed_s": round(elapsed_s, 3),
        "throughput_rps": round(len(ok) / elapsed_s, 3) if elapsed_s > 0 else 0.0,
        "input_tok_per_s": round(total_in / elapsed_s, 1) if elapsed_s > 0 else 0.0,
        "output_tok_per_s": round(total_out / elapsed_s, 1) if elapsed_s > 0 else 0.0,
        "slo": {
            "ttft_slo_ms": ttft_slo_ms,
            "tpot_slo_ms": tpot_slo_ms,
            "goodput_ratio": round(slo_ok / len(measured), 4) if measured else 0.0,
            "goodput_rps": round(slo_ok / elapsed_s, 3) if elapsed_s > 0 else 0.0,
        },
        "ttft_ms": _dist([r.ttft_ms for r in ok]),
        "tpot_ms": _dist([r.tpot_ms for r in ok]),
        "itl_ms": _dist(all_itls),
        "e2el_ms": _dist([r.e2el_ms for r in ok]),
        "server_metrics": {
            "hbm_prefix_hit_rate": round(prom_delta.get("hbm_prefix_hit_rate", 0.0), 4),
            "external_prefix_hit_rate": round(prom_delta.get("external_prefix_hit_rate", 0.0), 4),
            "kv_cache_usage_perc": round(
                prom_delta.get("vllm:kv_cache_usage_perc", prom_delta.get("vllm:gpu_cache_usage_perc", 0.0)),
                4,
            ),
            "preemptions": int(prom_delta.get("vllm:num_preemptions_total", 0.0)),
        },
    }


def format_serving_comparison(summaries: Sequence[Dict[str, Any]]) -> str:
    by_scen: Dict[str, List[Dict[str, Any]]] = {}
    for s in summaries:
        by_scen.setdefault(s["scenario"], []).append(s)

    lines: List[str] = []
    for scen, runs in by_scen.items():
        lines.append(f"\n=== Serving Scenario: {scen} ===")
        labels = [r["label"] for r in runs]
        hdr = f"{'Metric':<34}" + "".join(f"{lb:>18}" for lb in labels)
        lines.append(hdr)
        lines.append("-" * len(hdr))

        def row(name: str, fn):
            lines.append(f"{name:<34}" + "".join(f"{fn(r):>18}" for r in runs))

        row("Completed / Failed reqs", lambda r: f"{r['requests_ok']}/{r['requests_failed']}")
        row("TTFT p50 / p99 (ms)", lambda r: f"{r['ttft_ms']['p50']:.1f}/{r['ttft_ms']['p99']:.1f}")
        row("TPOT p50 / p99 (ms)", lambda r: f"{r['tpot_ms']['p50']:.1f}/{r['tpot_ms']['p99']:.1f}")
        row("ITL p50 / p99 (ms)", lambda r: f"{r['itl_ms']['p50']:.1f}/{r['itl_ms']['p99']:.1f}")
        row("E2EL p50 / p99 (ms)", lambda r: f"{r['e2el_ms']['p50']:.1f}/{r['e2el_ms']['p99']:.1f}")
        row("Goodput (% meeting SLO)", lambda r: f"{r['slo']['goodput_ratio']*100:6.2f}%")
        row("Throughput (req/s)", lambda r: f"{r['throughput_rps']:8.2f}")
        row("Output throughput (tok/s)", lambda r: f"{r['output_tok_per_s']:8.1f}")
        row("HBM prefix hit rate (/metrics)", lambda r: f"{r['server_metrics']['hbm_prefix_hit_rate']*100:6.2f}%")
        row("External prefix hit rate", lambda r: f"{r['server_metrics']['external_prefix_hit_rate']*100:6.2f}%")
        row("KV cache usage / Preemptions", lambda r: f"{r['server_metrics']['kv_cache_usage_perc']*100:.1f}%/{r['server_metrics']['preemptions']}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Built-in mock streaming vLLM server for fast local validation of Tier 3
# ---------------------------------------------------------------------------


async def run_self_test_mock() -> None:
    """Spins up a tiny local mock streaming vLLM server and validates serve_bench."""
    metrics_state = {"queries": 0, "hits": 0, "ext_queries": 0, "ext_hits": 0}
    seen_prefixes: set[str] = set()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            req_line = (await reader.readline()).decode("latin-1").strip()
            if not req_line:
                return
            parts = req_line.split(" ")
            path = parts[1] if len(parts) >= 2 else "/"
            content_len = 0
            while True:
                h = await reader.readline()
                if not h or h in (b"\r\n", b"\n"):
                    break
                hs = h.decode("latin-1").lower()
                if hs.startswith("content-length:"):
                    content_len = int(hs.split(":", 1)[1].strip())
            body = await reader.readexactly(content_len) if content_len > 0 else b""

            if path == "/metrics":
                txt = (
                    f"vllm:prefix_cache_queries_total {metrics_state['queries']}\n"
                    f"vllm:prefix_cache_hits_total {metrics_state['hits']}\n"
                    f"vllm:external_prefix_cache_queries_total {metrics_state['ext_queries']}\n"
                    f"vllm:external_prefix_cache_hits_total {metrics_state['ext_hits']}\n"
                    f"vllm:kv_cache_usage_perc 0.42\n"
                    f"vllm:num_preemptions_total 0\n"
                ).encode("utf-8")
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(txt)}\r\nConnection: close\r\n\r\n".encode()
                    + txt
                )
                await writer.drain()
                return

            payload = json.loads(body.decode("utf-8"))
            prompt = payload.get("prompt", "")
            prefix_key = prompt[:128]
            metrics_state["queries"] += 10
            metrics_state["ext_queries"] += 10
            is_hit = prefix_key in seen_prefixes
            seen_prefixes.add(prefix_key)
            if is_hit:
                metrics_state["hits"] += 2
                metrics_state["ext_hits"] += 8
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.03)

            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            out_n = min(6, int(payload.get("max_tokens", 4)))
            for i in range(out_n):
                evt = {"choices": [{"text": f" w{i}"}]}
                writer.write(f"data: {json.dumps(evt)}\n\n".encode())
                await writer.drain()
                await asyncio.sleep(0.004)
            usage_evt = {
                "choices": [],
                "usage": {"prompt_tokens": 512, "completion_tokens": out_n},
            }
            writer.write(f"data: {json.dumps(usage_evt)}\n\n".encode())
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    ep = f"127.0.0.1:{port}"
    try:
        reqs = gen_shared_prefix_offload(
            num_prefixes=3, prefix_tokens=1024, suffix_tokens=64, output_tokens=6,
            num_prompts=12, request_rate=20.0,
        )
        m0 = await asyncio.to_thread(scrape_prometheus_metrics, [ep])
        res, dt = await run_scenario(
            endpoint=ep, model="mock-model", requests=reqs,
            max_concurrency=4, ttft_slo_ms=500.0, tpot_slo_ms=50.0,
        )
        m1 = await asyncio.to_thread(scrape_prometheus_metrics, [ep])
        delta = diff_prometheus_metrics(m0, m1)
        summary = summarize_scenario(
            label="mock_mooncake", scenario="shared_prefix_offload",
            results=res, elapsed_s=dt, prom_delta=delta,
            ttft_slo_ms=500.0, tpot_slo_ms=50.0,
        )
        print(format_serving_comparison([summary]), flush=True)
        assert summary["requests_ok"] == 9 and summary["requests_failed"] == 0, summary
        assert summary["server_metrics"]["external_prefix_hit_rate"] > 0.5, summary
        print("[serve_bench] mock self-test PASSED", flush=True)
    finally:
        server.close()
        await server.wait_closed()


def main() -> None:
    p = argparse.ArgumentParser(description="Tier 3 end-to-end vLLM serving benchmark")
    p.add_argument("--mode", choices=("run", "compare", "self_test"), default="run")
    p.add_argument("--endpoint", default="http://127.0.0.1:8000")
    p.add_argument(
        "--metrics-endpoints",
        default="",
        help="Comma-separated vLLM server host:port list for /metrics scraping",
    )
    p.add_argument("--label", default="mooncake")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument(
        "--scenarios",
        default="shared_prefix_offload,multiturn_chat,fast25_conversation",
    )
    p.add_argument("--trace-dir", default="/opt/mooncake-vllm/traces")
    p.add_argument("--num-prompts", type=int, default=60)
    p.add_argument("--num-prefixes", type=int, default=8)
    p.add_argument("--prefix-tokens", type=int, default=8192)
    p.add_argument("--suffix-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--qps", type=float, default=2.0)
    p.add_argument("--qps-sweep", default="1,2,4")
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    p.add_argument("--tpot-slo-ms", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default="/tmp/vllm_bench_result.json")
    p.add_argument("--compare-files", nargs="*", default=[])
    args = p.parse_args()

    if args.mode == "self_test":
        asyncio.run(run_self_test_mock())
        return

    if args.mode == "compare":
        all_summaries: List[Dict[str, Any]] = []
        for path in args.compare_files:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_summaries.extend(data.get("summaries", data if isinstance(data, list) else []))
        print(format_serving_comparison(all_summaries), flush=True)
        return

    metrics_eps = [x.strip() for x in args.metrics_endpoints.split(",") if x.strip()]
    if not metrics_eps:
        metrics_eps = [args.endpoint]

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    summaries: List[Dict[str, Any]] = []

    for scen in scenarios:
        if scen == "disagg_slo_sweep":
            qps_values = [float(x) for x in args.qps_sweep.split(",") if x.strip()]
            for qps in qps_values:
                scen_name = f"disagg_qps_{qps:g}"
                reqs = gen_disagg_workload(
                    input_tokens=args.prefix_tokens,
                    output_tokens=args.output_tokens,
                    num_prompts=args.num_prompts,
                    request_rate=qps,
                    seed=args.seed,
                )
                m0 = scrape_prometheus_metrics(metrics_eps)
                res, dt = asyncio.run(
                    run_scenario(
                        endpoint=args.endpoint,
                        model=args.model,
                        requests=reqs,
                        max_concurrency=args.max_concurrency,
                        ttft_slo_ms=args.ttft_slo_ms,
                        tpot_slo_ms=args.tpot_slo_ms,
                    )
                )
                m1 = scrape_prometheus_metrics(metrics_eps)
                summaries.append(
                    summarize_scenario(
                        label=args.label,
                        scenario=scen_name,
                        results=res,
                        elapsed_s=dt,
                        prom_delta=diff_prometheus_metrics(m0, m1),
                        ttft_slo_ms=args.ttft_slo_ms,
                        tpot_slo_ms=args.tpot_slo_ms,
                    )
                )
            continue

        if scen == "shared_prefix_offload":
            reqs = gen_shared_prefix_offload(
                num_prefixes=args.num_prefixes,
                prefix_tokens=args.prefix_tokens,
                suffix_tokens=args.suffix_tokens,
                output_tokens=args.output_tokens,
                num_prompts=args.num_prompts,
                request_rate=args.qps,
                seed=args.seed,
            )
        elif scen == "multiturn_chat":
            reqs = gen_multiturn_chat(
                num_sessions=max(4, args.num_prompts // 4),
                turns_per_session=4,
                system_tokens=max(1024, args.prefix_tokens // 2),
                turn_tokens=512,
                output_tokens=args.output_tokens,
                request_rate=args.qps,
                seed=args.seed,
            )
        elif scen in ("fast25_conversation", "fast25_toolagent"):
            fname = "conversation_trace.jsonl" if scen == "fast25_conversation" else "toolagent_trace.jsonl"
            reqs = gen_fast25_replay(
                os.path.join(args.trace_dir, fname),
                max_prompts=args.num_prompts,
                max_blocks=max(4, args.prefix_tokens // 512),
                max_output_tokens=args.output_tokens,
                request_rate=args.qps,
                seed=args.seed,
            )
        else:
            raise ValueError(f"Unknown serving scenario: {scen!r}")

        print(f"[serve_bench] running {args.label} / {scen} ({len(reqs)} prompts)...", flush=True)
        m0 = scrape_prometheus_metrics(metrics_eps)
        res, dt = asyncio.run(
            run_scenario(
                endpoint=args.endpoint,
                model=args.model,
                requests=reqs,
                max_concurrency=args.max_concurrency,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            )
        )
        m1 = scrape_prometheus_metrics(metrics_eps)
        summaries.append(
            summarize_scenario(
                label=args.label,
                scenario=scen,
                results=res,
                elapsed_s=dt,
                prom_delta=diff_prometheus_metrics(m0, m1),
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            )
        )

    table = format_serving_comparison(summaries)
    print(table, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "model": args.model,
                "endpoint": args.endpoint,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"[serve_bench] saved {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
