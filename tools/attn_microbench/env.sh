#!/usr/bin/env bash
# attn_microbench 独立环境：只读依赖仓库 third_party/sglang，不修改主干。
set -euo pipefail
MICROBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MICROBENCH_ROOT}/../.." && pwd)"

export REPO_ROOT
export MICROBENCH_ROOT
export PYTHONPATH="${MICROBENCH_ROOT}:${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:$PYTHONPATH}"

export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME

# 与 tools/p27_launch_ds4flash_npu.sh 对齐（DSv4 attention / custom_ops 通路）
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export IS_DEEPSEEK_V4="${IS_DEEPSEEK_V4:-1}"
export USE_FUSED_COMPRESSOR="${USE_FUSED_COMPRESSOR:-1}"
export LI_KV_DTYPE_INT8="${LI_KV_DTYPE_INT8:-1}"
export USE_PA_DECODE="${USE_PA_DECODE:-1}"
export USE_PA_PREFILL="${USE_PA_PREFILL:-1}"
export USE_FUSED_HC_POST_ASCENDC="${USE_FUSED_HC_POST_ASCENDC:-1}"
export USE_FUSED_HC_PRE_ASCENDC="${USE_FUSED_HC_PRE_ASCENDC:-1}"
export USE_NPU_MOE_GATING_TOP_K="${USE_NPU_MOE_GATING_TOP_K:-1}"
export USE_FUSED_TRANSPOSE_BATCHMATMUL="${USE_FUSED_TRANSPOSE_BATCHMATMUL:-1}"
export USE_ROPE_PARTIAL_IN_PLACE_ASCENDC="${USE_ROPE_PARTIAL_IN_PLACE_ASCENDC:-1}"
export ASCEND_USE_FIA="${ASCEND_USE_FIA:-1}"

# custom_ops.so 依赖 libc10.so；与 p27 一致先挂 kml，再挂当前 Python 的 torch/lib
export LD_LIBRARY_PATH="/usr/local/kml/lib:${LD_LIBRARY_PATH:-}"

if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh" || true
fi

# ---------- NPU 卡选择 ----------
# 进程内逻辑设备恒为 npu:0；物理卡由 ASCEND_RT_VISIBLE_DEVICES 指定。
# 例：export ASCEND_RT_VISIBLE_DEVICES=3  →  绑定物理 NPU 3
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID:-0}"
fi
echo "[attn_microbench] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} (logical npu:0)"

_probe_py() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || return 1
  PYTHONPATH="$PYTHONPATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" "$bin" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("yaml", "torch", "torch_npu", "custom_ops"):
    importlib.import_module(m)
PY
}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for _cand in python3 python3.11 /usr/local/python3.11.14/bin/python3.11 \
               /usr/local/python3.11.14/bin/python3; do
    if _probe_py "$_cand"; then
      PYTHON_BIN="$(command -v "$_cand")"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "[attn_microbench][ERROR] 没找到能 import torch/torch_npu/custom_ops 的 python。" >&2
  echo "[attn_microbench][ERROR] 建议 export PYTHON_BIN=/usr/local/python3.11.14/bin/python3.11" >&2
  exit 2
fi
export PYTHON_BIN

_TORCH_LIB="$("${PYTHON_BIN}" - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
if [[ -n "${_TORCH_LIB}" && -d "${_TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH}"
fi
echo "[attn_microbench] PYTHON_BIN=${PYTHON_BIN}"

cd "${MICROBENCH_ROOT}"
