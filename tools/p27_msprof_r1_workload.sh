#!/usr/bin/env bash
# 由 msprof --application 直接拉起，采集 graph on R1 decode + F2。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export P27_SITE_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID:-2}"
export PORT="${PORT:-8001}"
export HOST="${HOST:-127.0.0.1}"
export SGLANG_NPU_PROFILE_ENABLE=0
unset KT_ACTIVATION_FREQ_PATH

LOG="${MSPROF_WORKLOAD_LOG:-${REPO}/tools/msprof_dbg/msprof_workload.log}"
mkdir -p "$(dirname "$LOG")"
: >"$LOG"

bash "$REPO/tools/p27_launch_ds4flash_npu.sh" >>"$LOG" 2>&1 &
SRV=$!

cleanup() {
  kill "$SRV" 2>/dev/null || true
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 840); do
  if grep -q "The server is fired up" "$LOG" 2>/dev/null; then break; fi
  if grep -qiE "NPU out of memory|Traceback|Bus error|SIGBUS|ERR 107027" "$LOG" 2>/dev/null; then
    if ! grep -q "The server is fired up" "$LOG" 2>/dev/null; then
      tail -60 "$LOG" >&2
      exit 1
    fi
  fi
  if (( i % 24 == 0 )); then
    layer=$(grep -oE "TP MOE layer [0-9]+" "$LOG" 2>/dev/null | tail -1 || true)
    echo "[workload] loading... $((i * 5))s ${layer:-}" >>"$LOG"
  fi
  sleep 5
done
grep -q "The server is fired up" "$LOG" || { echo "server timeout" >&2; exit 1; }

sleep 5
PYBIN="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
export HOST PORT PYTHON_BIN="$PYBIN"
"$PYBIN" - <<'PY' | tee -a "$LOG"
import json, os, time, urllib.request

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "8001")
base = f"http://{host}:{port}/generate"
prompts = [
    (1, 64, "Below is a Python function to compute Fibonacci numbers:"),
    (2, 128, "Explain the difference between supervised and unsupervised learning in three short paragraphs.\n\n"),
    (3, 80, "请用一句话解释什么是 transformer 模型："),
    (4, 128, "什么是 transformer 模型："),
]
results = []
for pid, mt, text in prompts:
    print(f"========== prompt {pid} (max_new_tokens={mt}) ==========")
    body = {"text": text, "sampling_params": {"max_new_tokens": mt, "temperature": 0}}
    req = urllib.request.Request(
        base, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
    el = time.perf_counter() - t0
    meta = json.loads(raw.decode())
    ct = (meta.get("meta_info") or {}).get("completion_tokens") or mt
    tp = ct / el if el > 0 else 0
    results.append((pid, tp))
    print(f"[timing] prompt={pid} elapsed={el:.2f}s tokens={ct} throughput={tp:.3f} tok/s")
steady = [t for p, t in results if p in (2, 4)]
if steady:
    print(f"[steady] avg tok/s (prompt 2+4) = {sum(steady)/len(steady):.3f}")
print("[f2] done")
PY
sleep 3
