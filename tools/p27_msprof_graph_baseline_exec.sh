#!/usr/bin/env bash
# R1 msprof exec+sidecar 采集（推荐）：msprof exec 拉起服务，sidecar 跑 F2。
# Runbook: doc/zh/Profiling_R1_msprof_Baseline.md
# 须在仓库根启动；MSPROF_DELAY 默认 900（跳过加载/编译，勿用 4200）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

NPU_DEVICE_ID="${NPU_DEVICE_ID:-2}"
PORT="${PORT:-8001}"
MSPROF_DELAY="${MSPROF_DELAY:-900}"
MSPROF_BIN="${MSPROF_BIN:-$(command -v msprof)}"
PYBIN="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
RUN_TAG="r1_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO}/tools/msprof_dbg/${RUN_TAG}"
EXEC_LOG="${OUT_DIR}/server.log"
F2_LOG="${OUT_DIR}/f2_throughput.log"

mkdir -p "$OUT_DIR"
export P27_SITE_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID}"
export NPU_DEVICE_ID
export PORT
export SGLANG_NPU_PROFILE_ENABLE=0
export MSPROF_WORKLOAD_LOG="$EXEC_LOG"
unset KT_ACTIVATION_FREQ_PATH

if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh || true
fi

pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
sleep 3

chmod +x "$REPO/tools/p27_msprof_launch_exec.sh" "$REPO/tools/p27_msprof_sidecar_f2.sh"

echo "[msprof-exec] OUT=$OUT_DIR card=$NPU_DEVICE_ID delay=${MSPROF_DELAY}s"
"$MSPROF_BIN" \
  --application="$REPO/tools/p27_msprof_launch_exec.sh" \
  --output="$OUT_DIR" \
  --delay="${MSPROF_DELAY}" \
  --aic-mode=task-based \
  --task-time=on \
  --runtime-api=on \
  --aicpu=on \
  --ascendcl=on \
  --hccl=off \
  >"${OUT_DIR}/msprof_collect.log" 2>&1 &
MPID=$!
MSPROF_START=$(date +%s)
export MSPROF_START MSPROF_DELAY

bash "$REPO/tools/p27_msprof_sidecar_f2.sh" "$EXEC_LOG" "$F2_LOG" || {
  kill "$MPID" 2>/dev/null || true
  exit 1
}

# F2 完成后给 msprof 刷盘，再优雅停服（避免 exit 137 导致 device_2/data 空）
sleep 60
LAUNCH_PID="$(pgrep -f "sglang.launch_server.*--port ${PORT}" | head -1 || true)"
if [[ -n "$LAUNCH_PID" ]]; then
  echo "[msprof-exec] SIGTERM launch_server pid=$LAUNCH_PID"
  kill -TERM "$LAUNCH_PID" 2>/dev/null || true
  for _ in $(seq 1 90); do
    kill -0 "$LAUNCH_PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "[msprof-exec] WARN: launch_server still alive after 180s, sending SIGKILL"
    kill -KILL "$LAUNCH_PID" 2>/dev/null || true
  fi
else
  echo "[msprof-exec] WARN: launch_server pid not found (pgrep miss?)"
fi
echo "[msprof-exec] post-stop flush sleep 60s ..."
sleep 60
wait "$MPID" 2>/dev/null || true

has_kernel_csv() {
  find "$OUT_DIR" -name 'kernel_details.csv' -print -quit | grep -q .
}

# msprof --application 通常已自动 export；无 kernel_details 时对每个 PROF_* 补 export
if ! has_kernel_csv; then
  : >"${OUT_DIR}/export.log"
  for prof in "$OUT_DIR"/PROF_*; do
    [[ -d "$prof" ]] || continue
    echo "[msprof-exec] export PROF $(basename "$prof")" | tee -a "${OUT_DIR}/export.log"
    "$MSPROF_BIN" --export=on --output="$prof" --type=text --summary-format=csv \
      2>&1 | tee -a "${OUT_DIR}/export.log" || true
  done
fi

"$PYBIN" "$REPO/tools/p27_parse_msprof_baseline.py" \
  --run-dir "$OUT_DIR" --label R1 --f2-log "$F2_LOG" \
  --write-json "${OUT_DIR}/parsed_summary.json" || true

echo "[msprof-exec] DONE $OUT_DIR"
