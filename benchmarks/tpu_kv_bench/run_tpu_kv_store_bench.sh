#!/usr/bin/env bash
# Run Tier 2 multi-host TPU KV-store benchmark (Mooncake Store vs tpu-sync) on CDK.
#
#   local_test/cdk/run_tpu_kv_store_bench.sh [KEY=VALUE ...]
#   local_test/cdk/run_tpu_kv_store_bench.sh \
#     PEERS=1 \
#     BENCH_ARGS="--backends mooncake,tpu_sync --geometry medium --workloads multiturn_tree,hot_prefix_fanout,burst_write_flood,p2d_overlap,worker_restart,fast25_conversation"
#
# Environment overrides:
#   CDK_HOST, VENV, SITE (e.g. a tpu-6-store / dma-map pip --target dir),
#   TPU_SYNC, BUNDLE, CDK_TAGS, CDK_ACK, BUNDLE_ONLY=1, RENDER_ONLY=1.
set -euo pipefail

CDK_HOST=${CDK_HOST:-wangziyi@10.128.0.211}
VENV=${VENV:-$HOME/.venvs/mooncake}
SITE=${SITE:-}
TPU_SYNC=${TPU_SYNC:-$HOME/tpu-sync-1279}
BUNDLE=${BUNDLE:-/mnt/disks/vllm-data/wangziyi/mooncake-cdk-kv-bundle}
TAGS=${CDK_TAGS:-mooncake-tpu-kv-bench}${CDK_ACK:+,antigravity,agent-jobs}
RECIPE=mooncake-tpu-kv-store-bench
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
BENCH_DIR="$HERE/kv_store_bench"
TRACE_DIR=${TRACE_DIR:-$REPO/FAST25-release/traces}
[[ -d $TRACE_DIR ]] || TRACE_DIR="$HOME/Mooncake/FAST25-release/traces"
SSH=(ssh -o BatchMode=yes)

echo "== 1/4 bundling into $BUNDLE"
"$VENV/bin/python" - "$BUNDLE" "$TPU_SYNC" "$BENCH_DIR" "$TRACE_DIR" "$SITE" <<'PY'
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import time

bundle, tpu_sync, bench_dir, trace_dir, extra_site = sys.argv[1:]
py, site, ts, lib, bindir, bdest, tdest = (
    os.path.join(bundle, d)
    for d in ("python", "site-packages", "tpu-sync", "lib", "bin", "kv_store_bench", "traces")
)
for d in (py, site, ts, lib, bindir, bdest, tdest):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.realpath(src), dst)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


# 1. Python 3.12 + stdlib
copy("/usr/bin/python3.12", os.path.join(py, "bin", "python3.12"))
shutil.copytree(
    "/usr/lib/python3.12",
    os.path.join(py, "lib", "python3.12"),
    ignore=shutil.ignore_patterns("__pycache__", "test", "dist-packages", "config-3.12-*"),
)

# 2. jax, jaxlib, libtpu, mooncake-transfer-engine (+ transitive deps)
dists = {norm(d.metadata["Name"]): d for d in md.distributions()}
if extra_site:
    dists.update({norm(d.metadata["Name"]): d for d in md.distributions(path=[extra_site])})
todo, wanted = ["jax", "jaxlib", "libtpu", "mooncake-transfer-engine"], set()
while todo:
    name = todo.pop()
    if name in wanted or name not in dists:
        continue
    wanted.add(name)
    for req in dists[name].requires or []:
        if "extra ==" not in req:
            todo.append(norm(re.split(r"[\s;<>=!~\[(]", req, maxsplit=1)[0]))
for name in wanted:
    for f in dists[name].files or []:
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

# 3. tpu-sync Python sources + both JAX C++ extensions (_mooncake_provider + _tpu_raiden_jax)
for root, dirs, files in os.walk(os.path.join(tpu_sync, "tpu_sync")):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".py") or f.startswith(("_mooncake_provider.", "_tpu_raiden_jax.")):
            src = os.path.join(root, f)
            copy(src, os.path.join(ts, os.path.relpath(src, tpu_sync)))

# tpu-sync control-plane binaries (global_registry_server + optional host store node).
for rel_bin, dst_name in (
    ("tpu_sync/kv_cache/global_registry/global_registry_server", "global_registry_server"),
    ("tpu_sync/store_node/kv_cache_host_store_node_main", "kv_cache_host_store_node_main"),
):
    src = os.path.join(tpu_sync, rel_bin)
    if os.path.exists(src):
        dst = os.path.join(bindir, dst_name)
        copy(src, dst)
        os.chmod(dst, 0o755)

# 4. Benchmark scripts + FAST'25 trace files.
for f in os.listdir(bench_dir):
    if f.endswith(".py"):
        copy(os.path.join(bench_dir, f), os.path.join(bdest, f))
if os.path.isdir(trace_dir):
    for f in ("conversation_trace.jsonl", "toolagent_trace.jsonl", "synthetic_trace.jsonl"):
        src = os.path.join(trace_dir, f)
        if os.path.isfile(src):
            copy(src, os.path.join(tdest, f))

# 5. Collect all non-glibc shared libraries referenced by binaries & extensions.
glibc = re.compile(
    r"(ld-linux|libc|libm|libmvec|libpthread|libdl|librt|libresolv|libutil|"
    r"libnsl|libanl|libnss_\w+|libBrokenLocale|libthread_db)[.-]"
)
objects = [os.path.join(py, "bin", "python3.12")]
for b in os.listdir(bindir):
    objects.append(os.path.join(bindir, b))
for root, _, files in os.walk(bundle):
    objects += [os.path.join(root, f) for f in files if re.search(r"\.so(\.\d+)*$", f)]
needed = {}
ldd_env = dict(os.environ)
ldd_env["LD_LIBRARY_PATH"] = os.path.join(site, "mooncake") + ":" + ldd_env.get("LD_LIBRARY_PATH", "")
for obj in objects:
    ldd = subprocess.run(["ldd", obj], capture_output=True, text=True, env=ldd_env).stdout
    for name, path in re.findall(r"^\s*(\S+) => (/\S+)", ldd, re.M):
        if not glibc.match(name) and not path.startswith(bundle):
            needed[name] = path
for name, path in needed.items():
    copy(path, os.path.join(lib, name))

with open(os.path.join(bundle, "BUNDLE_INFO"), "w") as out:
    out.write(
        f"kv-store bundle built {time.strftime('%Y-%m-%d %H:%M:%S %Z')} on {os.uname().nodename}\n"
        f"mooncake-transfer-engine {dists['mooncake-transfer-engine'].version} "
        f"(extra_site={extra_site or 'none'})\n"
        f"jax {dists['jax'].version}, libtpu {dists['libtpu'].version}, tpu-sync {tpu_sync}\n"
    )
PY
cat "$BUNDLE/BUNDLE_INFO"
size_kib=$(du -sk "$BUNDLE" | cut -f1)
echo "bundle: $((size_kib / 1024)) MiB"
if ((size_kib > 2 * 1024 * 1024)); then
    echo "bundle exceeds --map-dir's 2 GiB cap" >&2
    exit 1
fi

if [[ -n ${BUNDLE_ONLY:-} ]]; then
    echo "== BUNDLE_ONLY=1: stopping after bundle verification"
    exit 0
fi

echo "== 2/4 syncing to $CDK_HOST"
"${SSH[@]}" "$CDK_HOST" "mkdir -p mooncake-cdk-kv cloud-devkit/recipes/experimental/$RECIPE"
rsync -a --delete -e "${SSH[*]}" "$BUNDLE/" "$CDK_HOST:mooncake-cdk-kv/bundle/"
rsync -a -e "${SSH[*]}" "$HERE/tpu_kv_store_bench.yml" \
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
    --active-deadline-seconds 3600 \
    --map-dir \"\$HOME/mooncake-cdk-kv/bundle:*:/opt/mooncake-bench\"$args" 2>&1) || {
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
