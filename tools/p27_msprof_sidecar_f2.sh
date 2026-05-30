#!/usr/bin/env bash
# Sidecar：等 msprof+exec 服务 ready 后跑 F2（与 p27_msprof_launch_exec.sh 配合）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
PYBIN="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"

_abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    echo "$p"
    return
  fi
  local cand dir base
  for cand in "$PWD/$p" "$REPO/$p" "$SCRIPT_DIR/$p"; do
    if [[ -e "$cand" ]]; then
      dir="$(cd "$(dirname "$cand")" && pwd)"
      base="$(basename "$cand")"
      echo "${dir}/${base}"
      return
    fi
  done
  echo "$REPO/$p"
}

LOG="$(_abs_path "${1:-tools/msprof_dbg/msprof_exec.log}")"
OUT_F2="$(_abs_path "${2:-tools/msprof_dbg/f2_sidecar.log}")"
mkdir -p "$(dirname "$OUT_F2")"

echo "[sidecar] waiting server log=$LOG (resolved)"
for i in $(seq 1 840); do
  if grep -q "The server is fired up" "$LOG" 2>/dev/null; then
    echo "[sidecar] ready at ${i}x5s"
    break
  fi
  if grep -qiE "Segmentation fault|NPU out of memory|Traceback" "$LOG" 2>/dev/null; then
    if ! grep -q "The server is fired up" "$LOG" 2>/dev/null; then
      echo "[sidecar] server failed" >&2
      tail -40 "$LOG" >&2
      exit 1
    fi
  fi
  sleep 5
done
grep -q "The server is fired up" "$LOG" || { echo "[sidecar] timeout" >&2; exit 1; }

# msprof --delay 从 msprof 启动起算；ready 后须等采数窗口打开再跑 F2
MSPROF_DELAY="${MSPROF_DELAY:-900}"
MSPROF_START="${MSPROF_START:-0}"
if [[ "$MSPROF_START" -gt 0 ]]; then
  now=$(date +%s)
  wait_left=$(( MSPROF_START + MSPROF_DELAY - now ))
  if (( wait_left > 0 )); then
    echo "[sidecar] msprof delay: wait ${wait_left}s before F2 (profiling opens at +${MSPROF_DELAY}s)"
    sleep "$wait_left"
  else
    echo "[sidecar] msprof delay already passed (${wait_left}s), run F2 now"
  fi
fi

sleep 5
export HOST PORT
"$PYBIN" - <<'PY' | tee "$OUT_F2"
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
    req = urllib.request.Request(base, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
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
echo "[sidecar] F2 complete -> $OUT_F2"
