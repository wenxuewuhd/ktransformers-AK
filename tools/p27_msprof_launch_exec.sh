#!/usr/bin/env bash
# msprof --application 入口：exec launch_server（进程树在 msprof 下，勿 background）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export P27_SITE_PATCH=1
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID:-2}"
export PORT="${PORT:-8001}"
export NPU_DEVICE_ID="${NPU_DEVICE_ID:-2}"
export SGLANG_NPU_PROFILE_ENABLE=0
unset KT_ACTIVATION_FREQ_PATH
LOG="${MSPROF_WORKLOAD_LOG:-${REPO}/tools/msprof_dbg/msprof_exec.log}"
if [[ "$LOG" != /* ]]; then
  LOG="$REPO/$LOG"
fi
mkdir -p "$(dirname "$LOG")"
# 勿用 | tee：pipe 会使 msprof 跟踪 bash+tee 子进程，停服易 exit 137
bash "$REPO/tools/p27_launch_ds4flash_npu.sh" >>"$LOG" 2>&1
