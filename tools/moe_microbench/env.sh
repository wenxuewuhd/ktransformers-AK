#!/usr/bin/env bash
# moe_microbench 独立环境：复用 p27_launch_ds4flash_npu.sh 的 MoE 相关环境子集。
# 不自动选 NPU 卡 — 共享资源，请先看 npu-smi info 再 export ASCEND_RT_VISIBLE_DEVICES。
set -euo pipefail
MICROBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MICROBENCH_ROOT}/../.." && pwd)"

export MICROBENCH_ROOT
export REPO_ROOT
export PYTHONPATH="${MICROBENCH_ROOT}:${REPO_ROOT}/third_party/sglang/python${PYTHONPATH:+:$PYTHONPATH}"

# ---- CANN ----
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh" || true
fi
export LD_LIBRARY_PATH="/usr/local/kml/lib:${LD_LIBRARY_PATH:-}"

# ---- 性能 / 调度（与 p27 launch 对齐） ----
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# ---- DSv4 / MoE flag（保留以防 sgl_kernel_npu 内部读取） ----
export IS_DEEPSEEK_V4="${IS_DEEPSEEK_V4:-1}"
export USE_NPU_MOE_GATING_TOP_K="${USE_NPU_MOE_GATING_TOP_K:-1}"

# ---- Python（必须含 torch + yaml） ----
_probe_py() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || return 1
  PYTHONPATH="$PYTHONPATH" "$bin" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("yaml", "torch"):
    importlib.import_module(m)
PY
}
if [[ -z "${PYTHON_BIN:-}" ]] || ! _probe_py "${PYTHON_BIN}"; then
  for _cand in /usr/local/python3.11.14/bin/python3.11 python3.11 python3; do
    if _probe_py "$_cand"; then
      PYTHON_BIN="$_cand"
      export PYTHON_BIN
      break
    fi
  done
fi
if ! _probe_py "${PYTHON_BIN:-}"; then
  echo "[moe_microbench][ERROR] 未找到含 torch+yaml 的 python；export PYTHON_BIN=/usr/local/python3.11.14/bin/python3.11" >&2
  exit 2
fi

# ---- 选卡提示（不自动 export） ----
if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "[moe_microbench] 未设置 ASCEND_RT_VISIBLE_DEVICES。建议:"
  echo "  npu-smi info       # 看哪张卡空"
  echo "  export ASCEND_RT_VISIBLE_DEVICES=2   # 选空卡（举例）"
else
  echo "[moe_microbench] 使用 NPU 卡 ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
fi

cd "${MICROBENCH_ROOT}"
