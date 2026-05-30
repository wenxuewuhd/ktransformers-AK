#!/usr/bin/env bash
# R1 msprof 基线采集：graph on、inline profiler off、hybrid N=32 prefix、单并发 F2。
#
# 用法（仓库根或任意目录）：
#   bash tools/p27_msprof_graph_baseline.sh
#   NPU_DEVICE_ID=2 PORT=8001 bash tools/p27_msprof_graph_baseline.sh
#
# 环境覆盖：
#   NPU_DEVICE_ID          物理 NPU 卡号（默认 2）
#   PORT                   HTTP 端口（默认 8001，避免与 8000 冲突）
#   MSPROF_OUTPUT_ROOT     产物根目录（默认 tools/msprof_dbg）
#   MSPROF_DURATION_SEC    dynamic attach 采集时长秒（默认 240）
#   MSPROF_MODE            application | dynamic（默认 dynamic）
#   SKIP_SERVER=1          仅跑 F2 + 解析（server 已在外部运行）
#   SKIP_PARSE=1           只采集不解析
#
# 约束：勿设 SGLANG_NPU_PROFILE_ENABLE=1、勿设 KT_ACTIVATION_FREQ_PATH、勿开 KT_DEBUG_*。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

NPU_DEVICE_ID="${NPU_DEVICE_ID:-2}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
MSPROF_OUTPUT_ROOT="${MSPROF_OUTPUT_ROOT:-${REPO}/tools/msprof_dbg}"
MSPROF_DURATION_SEC="${MSPROF_DURATION_SEC:-300}"
MSPROF_MODE="${MSPROF_MODE:-application}"
SERVER_WAIT_SEC="${SERVER_WAIT_SEC:-4200}"
MSPROF_ATTACH_ONLY="${MSPROF_ATTACH_ONLY:-0}"
SERVER_PID="${SERVER_PID:-}"
PYBIN="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
MSPROF_BIN="${MSPROF_BIN:-$(command -v msprof)}"
RUN_TAG="r1_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${MSPROF_OUTPUT_ROOT}/${RUN_TAG}"
SERVER_LOG="${OUT_DIR}/server.log"
F2_LOG="${OUT_DIR}/f2_throughput.log"
META_JSON="${OUT_DIR}/run_meta.json"

mkdir -p "$OUT_DIR"

if [[ -f "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh" || true
fi

# ---------- 强制 R1 形态 ----------
export P27_SITE_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID}"
export PORT
export SGLANG_NPU_PROFILE_ENABLE=0
unset KT_ACTIVATION_FREQ_PATH
unset KT_COLLECT_MOE_ACTIVATION_FREQ
unset KT_DEBUG_HYBRID_MOE
unset KT_DEBUG_MOE_OUT

echo "[msprof-r1] REPO=$REPO"
echo "[msprof-r1] NPU card=${NPU_DEVICE_ID} PORT=${PORT} mode=${MSPROF_MODE}"
echo "[msprof-r1] output=${OUT_DIR}"
echo "[msprof-r1] commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

CANN_VER=""
for vf in /usr/local/Ascend/cann-8.5.0/version.cfg /usr/local/Ascend/ascend-toolkit/latest/version.cfg; do
  if [[ -f "$vf" ]]; then CANN_VER="$(grep -E '^version=' "$vf" | head -1 | cut -d= -f2- || true)"; break; fi
done
echo "[msprof-r1] CANN=${CANN_VER:-unknown} msprof=${MSPROF_BIN}"

wait_for_server() {
  local logfile="$1" timeout_sec="${2:-900}"
  local i
  for ((i = 0; i < timeout_sec; i += 5)); do
    if grep -q "The server is fired up" "$logfile" 2>/dev/null; then
      echo "[msprof-r1] server ready (${i}s) — log marker"
      return 0
    fi
    if curl -sf -m 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[msprof-r1] server ready (${i}s) — HTTP /v1/models"
      return 0
    fi
    if (( i > 0 && i % 120 == 0 )); then
      local layer
      layer=$(grep -oE "TP MOE layer [0-9]+" "$logfile" 2>/dev/null | tail -1 || true)
      echo "[msprof-r1] still loading... ${i}s elapsed ${layer:-"(no layer log yet)"}"
    fi
    if grep -qiE "Bus error|SIGBUS|ERR 107027" "$logfile" 2>/dev/null; then
      echo "[msprof-r1][ERROR] server failed early:" >&2
      tail -80 "$logfile" >&2 || true
      return 1
    fi
    if grep -q "Traceback (most recent call last)" "$logfile" 2>/dev/null; then
      if ! grep -q "The server is fired up" "$logfile" 2>/dev/null; then
        echo "[msprof-r1][ERROR] server traceback before ready:" >&2
        tail -80 "$logfile" >&2 || true
        return 1
      fi
    fi
    sleep 5
  done
  echo "[msprof-r1][ERROR] timeout waiting for server (${timeout_sec}s)" >&2
  tail -40 "$logfile" >&2 || true
  return 1
}

find_launch_server_pid() {
  # 取 launch_server 主进程（排除 bash 包装）
  pgrep -f "sglang.launch_server.*--port ${PORT}" | head -1
}

run_f2_with_timing() {
  export HOST PORT PYTHON_BIN="$PYBIN"
  "$PYBIN" - <<'PY' | tee "$F2_LOG"
import json
import os
import time
import urllib.request

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "8001")
base = f"http://{host}:{port}/generate"

prompts = [
    (1, 64, "Below is a Python function to compute Fibonacci numbers:"),
    (
        2,
        128,
        "Explain the difference between supervised and unsupervised learning in three short paragraphs.\n\n",
    ),
    (3, 80, "请用一句话解释什么是 transformer 模型："),
    (4, 128, "什么是 transformer 模型："),
]

results = []
for pid, max_tok, text in prompts:
    print(f"========== prompt {pid} (max_new_tokens={max_tok}) ==========")
    body = {"text": text, "sampling_params": {"max_new_tokens": max_tok, "temperature": 0}}
    req = urllib.request.Request(
        base,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
    elapsed = time.perf_counter() - t0
    meta = json.loads(raw.decode())
    meta_info = meta.get("meta_info") or {}
    completion_tokens = meta_info.get("completion_tokens")
    if completion_tokens is None:
        # /generate 有时只有 text；用 max_tok 粗估 decode 段
        completion_tokens = max_tok
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
    rec = {
        "prompt_id": pid,
        "max_new_tokens": max_tok,
        "elapsed_s": round(elapsed, 3),
        "completion_tokens": completion_tokens,
        "tok_per_s": round(tok_s, 3),
    }
    results.append(rec)
    print(f"[timing] prompt={pid} elapsed={elapsed:.2f}s tokens={completion_tokens} throughput={tok_s:.3f} tok/s")
    print((meta.get("text") or "")[:800])
    print()

# 稳态参考：prompt 2 / 4（128 decode）
steady = [r for r in results if r["prompt_id"] in (2, 4)]
if steady:
    avg = sum(r["tok_per_s"] for r in steady) / len(steady)
    print(f"[steady] avg tok/s (prompt 2+4) = {avg:.3f}")
print("[f2] done")
PY
}

stop_server() {
  local pid="${1:-}"
  pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[msprof-r1] stopping server pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 2
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  sleep 2
}

export_prof_results() {
  # 进程外 msprof 产物需 export 才出 ASCEND_PROFILER_OUTPUT（若尚未存在）
  if compgen -G "${OUT_DIR}/**/ASCEND_PROFILER_OUTPUT/kernel_details.csv" >/dev/null 2>&1; then
    echo "[msprof-r1] ASCEND_PROFILER_OUTPUT already present"
    return 0
  fi
  if [[ ! -d "$OUT_DIR" ]]; then return 1; fi
  echo "[msprof-r1] running msprof --export=on ..."
  "$MSPROF_BIN" --export=on --output="$OUT_DIR" --type=text --summary-format=csv 2>&1 | tee "${OUT_DIR}/export.log" || true
}

write_meta() {
  local server_pid="${1:-}" msprof_ec="${2:-0}"
  "$PYBIN" - <<PY
import json, os, subprocess
from datetime import datetime, timezone

repo = ${REPO@Q}
out = ${OUT_DIR@Q}
server_pid_val = """${server_pid}""" or None
msprof_ec_val = int("""${msprof_ec}""")
meta = {
    "run_tag": ${RUN_TAG@Q},
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "commit": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip() if os.path.isdir(os.path.join(repo, ".git")) else "unknown",
    "npu_device_id": ${NPU_DEVICE_ID@Q},
    "port": ${PORT@Q},
    "msprof_mode": ${MSPROF_MODE@Q},
    "msprof_output": out,
    "server_pid": server_pid_val,
    "msprof_exit_code": msprof_ec_val,
    "cann_version": ${CANN_VER@Q},
    "msprof_bin": ${MSPROF_BIN@Q},
    "env": {
        "SGLANG_NPU_PROFILE_ENABLE": "0",
        "KT_ACTIVATION_FREQ_PATH": "",
        "graph": "on (default launch)",
        "kt_num_gpu_experts": 32,
        "placement": "prefix",
    },
}
with open(${META_JSON@Q}, "w") as f:
    json.dump(meta, f, indent=2)
print("[msprof-r1] meta ->", ${META_JSON@Q})
PY
}

SERVER_PID=""
MSPROF_EC=0

if [[ "$MSPROF_MODE" == "application" && "${MSPROF_ATTACH_ONLY}" != "1" && "${SKIP_SERVER:-0}" != "1" ]]; then
  stop_server
  echo "[msprof-r1] msprof --application (server+F2 under msprof, ~50min load)"
  WL="$REPO/tools/p27_msprof_r1_workload.sh"
  export MSPROF_WORKLOAD_LOG="${OUT_DIR}/server.log"
  chmod +x "$WL"
  set +e
  "$MSPROF_BIN" \
    --application="$WL" \
    --output="$OUT_DIR" \
    --aic-mode=task-based \
    --aicpu=on \
    --task-time=on \
    --runtime-api=on \
    --hccl=off \
    2>&1 | tee "${OUT_DIR}/msprof_collect.log"
  MSPROF_EC=$?
  set -e
  if [[ -f "${OUT_DIR}/server.log" ]]; then
    grep -E "\[timing\]|\[steady\]|prompt [0-9]" "${OUT_DIR}/server.log" >"${F2_LOG}" 2>/dev/null || cp -f "${OUT_DIR}/server.log" "$F2_LOG"
  fi
  export_prof_results
  write_meta "" "$MSPROF_EC"
  if [[ "${SKIP_PARSE:-0}" != "1" ]]; then
    echo "[msprof-r1] parsing ..."
    "$PYBIN" "$REPO/tools/p27_parse_msprof_baseline.py" \
      --run-dir "$OUT_DIR" --label R1 --f2-log "$F2_LOG" --meta "$META_JSON" \
      --write-json "${OUT_DIR}/parsed_summary.json" || true
  fi
  echo "[msprof-r1] DONE output=${OUT_DIR}"
  exit "$MSPROF_EC"
elif [[ "${SKIP_SERVER:-0}" != "1" && "${MSPROF_ATTACH_ONLY}" != "1" ]]; then
  stop_server
  : >"$SERVER_LOG"

  echo "[msprof-r1] starting server (graph on, profiler off) ..."
  bash "$REPO/tools/p27_launch_ds4flash_npu.sh" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$SERVER_LOG" "$SERVER_WAIT_SEC"

  LAUNCH_PID="$(find_launch_server_pid || true)"
elif [[ "${MSPROF_ATTACH_ONLY}" == "1" ]]; then
  LAUNCH_PID="${SERVER_PID:-$(find_launch_server_pid || true)}"
  if [[ -z "$LAUNCH_PID" ]]; then
    echo "[msprof-r1][ERROR] MSPROF_ATTACH_ONLY=1 but no launch_server pid (set SERVER_PID=)" >&2
    exit 1
  fi
  echo "[msprof-r1] attach-only mode, pid=${LAUNCH_PID}"
else
  LAUNCH_PID=""
fi

if [[ -n "${LAUNCH_PID:-}" ]]; then
  if [[ -z "$LAUNCH_PID" ]]; then
    echo "[msprof-r1][ERROR] cannot find sglang.launch_server pid" >&2
    tail -50 "$SERVER_LOG" >&2
    exit 1
  fi
  echo "[msprof-r1] launch_server pid=${LAUNCH_PID} (wrapper pid=${SERVER_PID})"
  grep -E "Decode npu graph|Prefill npu graph|kt placement" "$SERVER_LOG" | tail -5 || true

  if [[ "$MSPROF_MODE" == "dynamic" ]]; then
    echo "[msprof-r1] msprof dynamic attach pid=${LAUNCH_PID} (no duration; stop after F2)"
    set +e
    "$MSPROF_BIN" \
      --output="$OUT_DIR" \
      --dynamic=on \
      --pid="$LAUNCH_PID" \
      --aic-mode=task-based \
      --aicpu=on \
      --task-time=on \
      --runtime-api=on \
      --hccl=off \
      >"${OUT_DIR}/msprof_collect.log" 2>&1 &
    MSPROF_WRAPPER_PID=$!
    set -e
    sleep 8
    echo "[msprof-r1] running F2 prompts under msprof window ..."
    run_f2_with_timing
    echo "[msprof-r1] stopping msprof pid=${MSPROF_WRAPPER_PID} ..."
    kill -INT "$MSPROF_WRAPPER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$MSPROF_WRAPPER_PID" 2>/dev/null || break
      sleep 2
    done
    kill -9 "$MSPROF_WRAPPER_PID" 2>/dev/null || true
    wait "$MSPROF_WRAPPER_PID" 2>/dev/null || true
    MSPROF_EC=$?
  else
    echo "[msprof-r1][ERROR] unsupported MSPROF_MODE=$MSPROF_MODE in attach path" >&2
    exit 1
  fi

  export_prof_results
  if [[ "${MSPROF_ATTACH_ONLY}" != "1" ]]; then
    stop_server "$SERVER_PID"
  fi
else
  echo "[msprof-r1] SKIP_SERVER=1 — assume server already ran; only parse if data exists"
  run_f2_with_timing || true
fi

write_meta "${LAUNCH_PID:-$SERVER_PID}" "$MSPROF_EC"

if [[ "${SKIP_PARSE:-0}" != "1" ]]; then
  echo "[msprof-r1] parsing ..."
  "$PYBIN" "$REPO/tools/p27_parse_msprof_baseline.py" \
    --run-dir "$OUT_DIR" \
    --label R1 \
    --f2-log "$F2_LOG" \
    --meta "$META_JSON" \
    --write-json "${OUT_DIR}/parsed_summary.json"
fi

echo "[msprof-r1] DONE output=${OUT_DIR}"
echo "[msprof-r1] logs: server=${SERVER_LOG} f2=${F2_LOG}"
