#!/usr/bin/env bash
# Run Tier 3 multi-host end-to-end vLLM serving benchmark on CDK.
#
# Builds a lightweight overlay (~300 MiB) containing Mooncake, tpu-sync, our
# Tier 3 connectors/router/harness, toy_proxy_server.py, and FAST'25 traces,
# maps it onto `IMAGE` (default: vllm/vllm-tpu:nightly), and submits a multi-host
# CDK JobSet.
#
# Examples:
#   # Distributed prefix-cache offload cluster (2 TPU pods):
#   local_test/cdk/run_tpu_vllm_bench.sh MODE=offload_cluster BACKEND=mooncake PEERS=1 MODEL=Qwen/Qwen3-32B HF_TOKEN=$HF_TOKEN
#   local_test/cdk/run_tpu_vllm_bench.sh MODE=offload_cluster BACKEND=tpu_sync PEERS=1 MODEL=Qwen/Qwen3-32B HF_TOKEN=$HF_TOKEN
#   local_test/cdk/run_tpu_vllm_bench.sh MODE=offload_cluster BACKEND=baseline PEERS=1 MODEL=Qwen/Qwen3-32B HF_TOKEN=$HF_TOKEN
#
#   # Disaggregated 1P1D serving (1 Prefill pod + 1 Decode pod):
#   local_test/cdk/run_tpu_vllm_bench.sh MODE=disagg_1p1d BACKEND=mooncake PEERS=1 MODEL=Qwen/Qwen3-32B HF_TOKEN=$HF_TOKEN
#   local_test/cdk/run_tpu_vllm_bench.sh MODE=disagg_1p1d BACKEND=tpu_sync PEERS=1 MODEL=Qwen/Qwen3-32B HF_TOKEN=$HF_TOKEN
#
#   # Combine saved JSON runs locally into a side-by-side comparison table:
#   python3 local_test/cdk/vllm_bench/serve_bench.py --mode compare --compare-files run_baseline.json run_tpusync.json run_mooncake.json
#
# Environment overrides:
#   CDK_HOST, VENV, SITE (e.g. a tpu-6-store / dma-map pip --target dir),
#   TPU_SYNC, TPU_INFERENCE, BUNDLE, CDK_TAGS, CDK_ACK, BUNDLE_ONLY=1, RENDER_ONLY=1.
set -euo pipefail

CDK_HOST=${CDK_HOST:-wangziyi@10.128.0.211}
VENV=${VENV:-$HOME/.venvs/mooncake}
SITE=${SITE:-}
TPU_SYNC=${TPU_SYNC:-$HOME/tpu-sync-1279}
TPU_INFERENCE=${TPU_INFERENCE:-$HOME/tpu-inference}
BUNDLE=${BUNDLE:-/mnt/disks/vllm-data/wangziyi/mooncake-cdk-vllm-overlay}
TAGS=${CDK_TAGS:-mooncake-tpu-vllm-bench}${CDK_ACK:+,antigravity,agent-jobs}
RECIPE=mooncake-tpu-vllm-bench
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
BENCH_DIR="$HERE/vllm_bench"
TRACE_DIR=${TRACE_DIR:-$REPO/FAST25-release/traces}
[[ -d $TRACE_DIR ]] || TRACE_DIR="$HOME/Mooncake/FAST25-release/traces"
SSH=(ssh -o BatchMode=yes)

echo "== 1/4 bundling lightweight vLLM overlay into $BUNDLE"
"$VENV/bin/python" - "$BUNDLE" "$TPU_SYNC" "$TPU_INFERENCE" "$BENCH_DIR" "$TRACE_DIR" "$SITE" <<'PY'
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import time

bundle, tpu_sync, tpu_inf, bench_dir, trace_dir, extra_site = sys.argv[1:]
site, ts, lib, bindir, bdest, tdest = (
    os.path.join(bundle, d)
    for d in ("site-packages", "tpu-sync", "lib", "bin", "vllm_bench", "traces")
)
for d in (site, ts, lib, bindir, bdest, tdest):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.realpath(src), dst)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


# 1. Mooncake package ONLY (Python, JAX, libtpu, vLLM come from the container image).
dists = {norm(d.metadata["Name"]): d for d in md.distributions()}
if extra_site:
    dists.update({norm(d.metadata["Name"]): d for d in md.distributions(path=[extra_site])})
mc_dist = dists.get("mooncake-transfer-engine")
if mc_dist is None:
    raise RuntimeError("mooncake-transfer-engine not found in VENV/SITE")
for f in mc_dist.files or []:
    if f.parts[0] != ".." and "__pycache__" not in f.parts:
        copy(f.locate(), os.path.join(site, f))

# Locate mooncake_master binary into $B/bin/mooncake_master.
mc_master_cands = [
    os.path.join(site, "mooncake", "mooncake_master"),
    os.path.join(os.path.dirname(sys.executable), "mooncake_master"),
]
if extra_site:
    mc_master_cands.insert(0, os.path.join(extra_site, "bin", "mooncake_master"))
    mc_master_cands.insert(0, os.path.join(extra_site, "mooncake", "mooncake_master"))
for cand in mc_master_cands:
    if os.path.isfile(cand):
        copy(cand, os.path.join(bindir, "mooncake_master"))
        os.chmod(os.path.join(bindir, "mooncake_master"), 0o755)
        break
else:
    raise FileNotFoundError(f"mooncake_master not found in {mc_master_cands}")

# 2. tpu-sync Python sources (including generated *_pb2.py) + JAX C++ extensions.
for root, dirs, files in os.walk(os.path.join(tpu_sync, "tpu_sync")):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py") or f.startswith(("_mooncake_provider.", "_tpu_raiden_jax.")):
            src = os.path.join(root, f)
            copy(src, os.path.join(ts, os.path.relpath(src, tpu_sync)))

for rel_bin, dst_name in (
    ("tpu_sync/kv_cache/global_registry/global_registry_server", "global_registry_server"),
    ("tpu_sync/store_node/kv_cache_host_store_node_main", "kv_cache_host_store_node_main"),
):
    src = os.path.join(tpu_sync, rel_bin)
    if os.path.exists(src):
        dst = os.path.join(bindir, dst_name)
        copy(src, dst)
        os.chmod(dst, 0o755)

# 3. Tier 3 vLLM connectors, router, harness, and toy_proxy_server.py.
for f in os.listdir(bench_dir):
    if f.endswith(".py"):
        copy(os.path.join(bench_dir, f), os.path.join(bdest, f))
toy_proxy = os.path.join(tpu_inf, "examples", "disagg", "toy_proxy_server.py")
if os.path.isfile(toy_proxy):
    copy(toy_proxy, os.path.join(bdest, "toy_proxy_server.py"))

if os.path.isdir(trace_dir):
    for f in ("conversation_trace.jsonl", "toolagent_trace.jsonl"):
        src = os.path.join(trace_dir, f)
        if os.path.isfile(src):
            copy(src, os.path.join(tdest, f))

# 4. Non-glibc / non-python shared libraries required by bundled binaries & extensions.
#    Container system libs (libc, libstdc++, libgcc_s, libpython) stay authoritative.
skip_lib = re.compile(
    r"(ld-linux|libc|libm|libmvec|libpthread|libdl|librt|libresolv|libutil|"
    r"libnsl|libanl|libnss_\w+|libBrokenLocale|libthread_db|libpython|"
    r"libstdc\+\+|libgcc_s)[.-]"
)
objects = [os.path.join(bindir, b) for b in os.listdir(bindir)]
for root, _, files in os.walk(bundle):
    objects += [os.path.join(root, f) for f in files if re.search(r"\.so(\.\d+)*$", f)]
needed = {}
for obj in objects:
    ldd = subprocess.run(["ldd", obj], capture_output=True, text=True).stdout
    for name, path in re.findall(r"^\s*(\S+) => (/\S+)", ldd, re.M):
        if not skip_lib.match(name) and not path.startswith(bundle):
            needed[name] = path
for name, path in needed.items():
    copy(path, os.path.join(lib, name))

with open(os.path.join(bundle, "BUNDLE_INFO"), "w") as out:
    out.write(
        f"vllm-overlay bundle built {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {os.uname().nodename}\n"
        f"mooncake-transfer-engine {mc_dist.version} (extra_site={extra_site or 'none'})\n"
        f"tpu-sync {tpu_sync}\n"
    )
PY
cat "$BUNDLE/BUNDLE_INFO"
size_kib=$(du -sk "$BUNDLE" | cut -f1)
echo "overlay bundle: $((size_kib / 1024)) MiB"
if ((size_kib > 2 * 1024 * 1024)); then
    echo "overlay bundle exceeds --map-dir's 2 GiB cap" >&2
    exit 1
fi

if [[ -n ${BUNDLE_ONLY:-} ]]; then
    echo "== BUNDLE_ONLY=1: stopping after overlay bundle verification"
    exit 0
fi

echo "== 2/4 syncing overlay + recipe to $CDK_HOST"
"${SSH[@]}" "$CDK_HOST" "mkdir -p mooncake-cdk-vllm cloud-devkit/recipes/experimental/$RECIPE"
rsync -a --delete -e "${SSH[*]}" "$BUNDLE/" "$CDK_HOST:mooncake-cdk-vllm/bundle/"
rsync -a -e "${SSH[*]}" "$HERE/tpu_vllm_bench.yml" \
    "$CDK_HOST:cloud-devkit/recipes/experimental/$RECIPE/values.yml"
"${SSH[@]}" "$CDK_HOST" bash -s -- "$RECIPE" "$(whoami)" <<'SH'
f=~/cloud-devkit/recipes.yml
grep -q "name: \"$1\"" "$f" && exit 0
[ -z "$(tail -c1 "$f")" ] || echo >>"$f"
printf -- '- name: "%s"\n  owner: "%s"\n  k8s_file: "recipes/experimental/%s/values.yml"\n  require_gcs_mount: true\n' \
    "$1" "$2" "$1" >>"$f"
echo "registered $1 in $f"
SH

ack=${CDK_ACK:+--agent-code=$CDK_ACK}
args=$(printf ' %q' "$@")
if [[ -n ${RENDER_ONLY:-} ]]; then
    echo "== 3/4 cdk recipe render $RECIPE $* (RENDER_ONLY)"
    "${SSH[@]}" "$CDK_HOST" "cdk $ack recipe render $RECIPE$args"
    exit 0
fi

echo "== 3/4 cdk job create $RECIPE $*"
out=$("${SSH[@]}" "$CDK_HOST" "cdk $ack job create $RECIPE -t $TAGS \
    --active-deadline-seconds 7200 \
    --map-dir \"\$HOME/mooncake-cdk-vllm/bundle:*:/opt/mooncake-vllm\"$args" 2>&1) || {
    echo "$out" >&2
    exit 1
}
echo "$out"
job=$(grep -oE 'j-[0-9a-f]{8}(-[0-9a-f]{4}){3}' <<<"$out" | head -1)
[[ -n $job ]] || { echo "no job id in cdk's output" >&2; exit 1; }

echo "== 4/4 $job: waiting for the initiator, then following its log"
"${SSH[@]}" "$CDK_HOST" bash -s -- "$job" "$ack" <<'SH'
job=$1 ack=$2
status() {
    cdk $ack job list -o json | python3 -c "
import json, sys
print(next((j['job_status'] for j in json.load(sys.stdin) if j['id'] == '$job'), 'unknown'))"
}
until cdk $ack job log "$job" -c initiator >/dev/null 2>&1; do
    s=$(status)
    echo "  $(date +%T) $s"
    case $s in Succeeded | Failed | Deleted*) break ;; esac
    sleep 30
done
cdk $ack job log "$job" -c initiator -f || true
echo "== $job: $(status)"
SH
