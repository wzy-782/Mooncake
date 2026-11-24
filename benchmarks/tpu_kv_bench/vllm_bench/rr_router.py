#!/usr/bin/env python3
"""Lightweight multi-host OpenAI-compatible HTTP router for Tier 3 vLLM offload benchmarks.

Routes `/v1/completions` and `/v1/chat/completions` across N vLLM-TPU pods using
either round-robin or prefix-aware hashing (so we can measure both cluster-wide
sharing under round-robin and cache locality under prefix routing). Zero third-party
dependencies (uses stdlib `http.server` + `http.client` with streaming passthrough).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import threading
from typing import List, Tuple
from urllib.parse import urlparse


class MultiPodRouter:
    def __init__(self, backends: List[Tuple[str, int]], policy: str = "round_robin"):
        self.backends = backends
        self.policy = policy
        self._rr = itertools.cycle(range(len(backends)))
        self._lock = threading.Lock()

    def pick_backend(self, body: bytes) -> Tuple[str, int]:
        if len(self.backends) == 1:
            return self.backends[0]
        if self.policy == "prefix_hash" and body:
            try:
                payload = json.loads(body.decode("utf-8"))
                prompt = payload.get("prompt", "")
                if isinstance(prompt, list):
                    key_bytes = str(prompt[:128]).encode("utf-8")
                else:
                    key_bytes = str(prompt)[:512].encode("utf-8")
                idx = int(hashlib.blake2b(key_bytes, digest_size=4).hexdigest(), 16) % len(self.backends)
                return self.backends[idx]
            except Exception:
                pass
        with self._lock:
            return self.backends[next(self._rr)]


def make_handler(router: MultiPodRouter):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _proxy(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b""

            if self.path == "/health":
                all_ok = True
                for host, port in router.backends:
                    try:
                        c = http.client.HTTPConnection(host, port, timeout=3.0)
                        c.request("GET", "/health")
                        r = c.getresponse()
                        r.read()
                        c.close()
                        if r.status != 200:
                            all_ok = False
                    except Exception:
                        all_ok = False
                code = 200 if all_ok else 503
                out = b'{"status":"ok"}\n' if all_ok else b'{"status":"not_ready"}\n'
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return

            host, port = router.pick_backend(body)
            fwd_headers = {}
            for k, v in self.headers.items():
                lk = k.lower()
                if lk in ("host", "connection", "transfer-encoding"):
                    continue
                fwd_headers[k] = v
            fwd_headers["Host"] = f"{host}:{port}"
            if body:
                fwd_headers["Content-Length"] = str(len(body))

            conn = http.client.HTTPConnection(host, port, timeout=600.0)
            try:
                conn.request(method, self.path, body=body if body else None, headers=fwd_headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                is_stream = "text/event-stream" in (resp.getheader("Content-Type") or "")
                for k, v in resp.getheaders():
                    lk = k.lower()
                    if lk in ("transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                if is_stream:
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while True:
                        chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    self.close_connection = True
                else:
                    data = resp.read()
                    if resp.getheader("Content-Length") is None:
                        self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            finally:
                conn.close()

        def do_GET(self):
            self._proxy("GET")

        def do_POST(self):
            self._proxy("POST")

    return Handler


def parse_backends(items: List[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for raw in items:
        if "://" in raw:
            u = urlparse(raw)
            out.append((u.hostname or "127.0.0.1", int(u.port or 8000)))
        else:
            h, p = raw.rsplit(":", 1)
            out.append((h, int(p)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-pod HTTP router for vLLM Tier 3 benchmarks")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--backends", nargs="+", required=True, help="List of host:port backend servers")
    p.add_argument("--policy", choices=("round_robin", "prefix_hash"), default="round_robin")
    args = p.parse_args()

    backends = parse_backends(args.backends)
    router = MultiPodRouter(backends, policy=args.policy)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(router))
    print(
        f"[rr_router] listening on {args.host}:{args.port} -> {backends} (policy={args.policy})",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
