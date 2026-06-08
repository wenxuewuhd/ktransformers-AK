#!/usr/bin/env bash
# P2.7：单卡 NPU + KT(LLAMAFILE) 启动 SGLang HTTP 服务（DeepSeek-V4-Flash W8A8）。
# 底盘已切到基线 iforgetmyname/sglang@dsv4_release（third_party/sglang），KT MoE 在该
# 基线内即原生支持；本脚本相当于基线 8 卡 launch_ds4flash_sglang.sh 的「单卡 KT 子集」。
#
# 量化：--quantization compressed-tensors，对齐磁盘 W8A8 (compressed-tensors / int-quantized)。
# 基线已不再读取 SGLANG_APPLY_CONFIG_BACKUP，相关历史变量已移除。
#
# 用法（在任意目录）：
#   bash /path/to/ktransformers-AK/tools/p27_launch_ds4flash_npu.sh
#   bash .../p27_launch_ds4flash_npu.sh 3          # 与 NPU_DEVICE_ID=3 等价，指定物理 NPU 卡号
#
# 常用覆盖（环境变量）：
#   REPO              默认本脚本所在仓库根
#   MODEL_PATH        默认 /workspace/models/DeepSeek-V4-Flash-W8A8
#   KT_GGUF_TEMPLATE  默认 /workspace/models/cache/dsv4_layer{layer_idx}.gguf（勿在 ``${var:-...}`` 默认值里写花括号）
#   PORT              默认 8000
#   ASCEND_TOOLKIT_HOME  默认 /usr/local/Ascend/ascend-toolkit/latest
#   NPU_DEVICE_ID     可选，物理 NPU 序号（如 2）。设置后会 export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICE_ID。
#   CHUNKED_PREFILL_SIZE  默认 2048（必须是 page-size=128 的倍数，且 >= page-size）。
#                         注意：不能传 -1。KT(LLAMAFILE) C++ MoE 内部 fp32 输出
#                         buffer 按 max_possible_qlen()=max(max_len, group_max_len)
#                         分配；-1 会被算成 1，prefill qlen>1 时立刻越界写堆
#                         → glibc tcache abort。详见 Handoff 附录 Z.7。
#   QUANTIZATION      默认 compressed-tensors（与基线一致）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=tools/p27_ensure_kt_kernel.sh
source "${SCRIPT_DIR}/p27_ensure_kt_kernel.sh"
p27_ensure_kt_kernel "$REPO"
export PYTHONPATH="${REPO}/third_party/sglang/python:${REPO}/kt-kernel/python${PYTHONPATH:+:$PYTHONPATH}"

# ---------- 选定 Python 解释器 ----------
# 现象：本镜像默认 PATH 里 ``python3`` 指 ``/usr/bin/python3``（系统 python，没装
# torch_npu/numpy/sglang）。真正可用的解释器在 ``/usr/local/python3.11.14``。
# 用户自定义 shell（例如 conda / 直接 export PATH）会把 3.11.14 提前到前面，
# 但一旦从某个 cron / systemd / clean bash -lic 拉起就会踩坑。下面做一次性探测：
# 1) 允许 ``PYTHON_BIN`` 覆盖；2) 否则按 (``python3`` → ``python3.11`` → 已知绝对
# 路径) 顺序找第一个能 import torch_npu+sglang+numpy 的解释器。
_probe_py() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || return 1
  PYTHONPATH="$PYTHONPATH" "$bin" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("numpy", "torch", "torch_npu", "sglang"):
    importlib.import_module(m)
PY
}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for _cand in python3 python3.11 /usr/local/python3.11.14/bin/python3.11 \
               /usr/local/python3.11.14/bin/python3 /opt/conda/bin/python3; do
    if _probe_py "$_cand"; then
      PYTHON_BIN="$(command -v "$_cand")"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "[p27][ERROR] 没找到能 import numpy/torch/torch_npu/sglang 的 python。" >&2
  echo "[p27][ERROR] 当前 PATH 中的 python3 = $(command -v python3 || echo none)；" >&2
  echo "[p27][ERROR] 建议显式 export PYTHON_BIN=/usr/local/python3.11.14/bin/python3.11 后重试。" >&2
  exit 2
fi
export PYTHON_BIN
echo "[p27] PYTHON_BIN=${PYTHON_BIN}"

# 可选首个参数：纯数字则视为物理 NPU 卡号（与 NPU_DEVICE_ID 一致）
if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ && -z "${NPU_DEVICE_ID:-}" ]]; then
  NPU_DEVICE_ID="$1"
  shift
fi

MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8}"
# 勿用 KT_GGUF_TEMPLATE="${KT:-...dsv4_layer{layer_idx}.gguf}"：bash 会把 {layer_idx} 的第一个 ``}`` 当成 ``${...:-}`` 的结束符，路径会变成 ``...{layer_idx.gguf}``。
# 默认 Q8_0（批量 convert 输出 dsv4_layer{L}.gguf）。须先 cp 新 kt_kernel_ext.so 到 kt-kernel/python/（手册 §2.4）。
# BF16 回退：export KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_bf16.gguf'
if [[ -z "${KT_GGUF_TEMPLATE:-}" ]]; then
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}.gguf'
fi
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
PORT="${PORT:-8000}"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME
export LD_LIBRARY_PATH="/usr/local/kml/lib:${LD_LIBRARY_PATH:-}"

# 与 script/launch_ds4flash_sglang.sh 对齐：单卡省略 HCCL/DeepEP/MTP 等；保留 CPU/Ascend 与融合 kernel。
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

export SGLANG_NPU_PROFILE_ENABLE="${SGLANG_NPU_PROFILE_ENABLE:-0}"
export SGLANG_NPU_PROFILE_DECODE_TOKEN="${SGLANG_NPU_PROFILE_DECODE_TOKEN:-2}"
export SGLANG_NPU_PROFILE_DIR="${SGLANG_NPU_PROFILE_DIR:-./npu_results_dbg}"
export SGLANG_NPU_PROFILE_LEVEL="${SGLANG_NPU_PROFILE_LEVEL:-0}"
export SGLANG_NPU_PROFILE_ANALYSE="${SGLANG_NPU_PROFILE_ANALYSE:-0}"
export SGLANG_NPU_PROFILE_DISABLE_GRAPH="${SGLANG_NPU_PROFILE_DISABLE_GRAPH:-1}"
export SGLANG_NPU_PROFILE_KEEP_EAGER_AFTER="${SGLANG_NPU_PROFILE_KEEP_EAGER_AFTER:-1}"
if [[ "${SGLANG_NPU_PROFILE_ENABLE}" == "1" && "${EXTRA_FLAGS:-}" != *"--disable-cuda-graph"* ]]; then
  EXTRA_FLAGS="${EXTRA_FLAGS:+$EXTRA_FLAGS }--disable-cuda-graph"
  echo "[p27] SGLANG_NPU_PROFILE_ENABLE=1: auto append EXTRA_FLAGS=--disable-cuda-graph"
fi

# 可选：若机器上存在 CANN set_env，则加载（失败则忽略）。
if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh" || true
fi

ulimit -n 65536 2>/dev/null || true

if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "[p27] 保留环境变量 ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
elif [[ -n "${NPU_DEVICE_ID:-}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE_ID}"
  echo "[p27] 已设置 ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}（物理卡，进程内为逻辑 npu:0）"
else
  echo "[p27] 提示: 未设置 NPU_DEVICE_ID 且未设置 ASCEND_RT_VISIBLE_DEVICES，将使用系统当前可见的全部 NPU；"
  echo "[p27]       单卡服务通常仍绑定逻辑设备 0（常为物理 0 号卡）。若 0 号卡被占用，请执行:"
  echo "[p27]         NPU_DEVICE_ID=2 bash $0   或   bash $0 2"
fi

echo "[p27] REPO=$REPO"
echo "[p27] PYTHONPATH head: ${PYTHONPATH%%:*}"
echo "[p27] chunked-prefill-size=${CHUNKED_PREFILL_SIZE}（正数须为 page_size 倍数；见脚本头注释）"
echo "[p27] kt-weight-path template=${KT_GGUF_TEMPLATE}"
echo "[p27] quantization=${QUANTIZATION} IS_DEEPSEEK_V4=${IS_DEEPSEEK_V4:-}"
echo "[p27] SGLANG_NPU_PROFILE_ENABLE=${SGLANG_NPU_PROFILE_ENABLE} DECODE_TOKEN=${SGLANG_NPU_PROFILE_DECODE_TOKEN}"
"${PYTHON_BIN}" -c "import sglang; print('[p27] sglang file:', sglang.__file__)"

# EXTRA_FLAGS 用于临时附加任意 sglang.launch_server 参数（不需要改脚本本体），例如：
#   EXTRA_FLAGS="--disable-cuda-graph"   bash tools/p27_launch_ds4flash_npu.sh   # 关图捕获走 eager
#   EXTRA_FLAGS="--cuda-graph-bs 2"      bash tools/p27_launch_ds4flash_npu.sh
# 调试 NPU aicore / aclnn 错误时可再叠加： ASCEND_LAUNCH_BLOCKING=1 bash ...
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
if [[ -n "${EXTRA_FLAGS}" ]]; then
  echo "[p27] EXTRA_FLAGS=${EXTRA_FLAGS}"
fi
# shellcheck disable=SC2086
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --device npu \
  --tensor-parallel-size 1 \
  --page-size 128 \
  --attention-backend ascend \
  --quantization "$QUANTIZATION" \
  --disable-shared-experts-fusion \
  --dtype bfloat16 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --max-prefill-tokens 65535 \
  --context-length 65536 \
  --watchdog-timeout 18000 \
  --skip-server-warmup \
  --kt-method LLAMAFILE \
  --kt-num-gpu-experts 32 \
  --kt-weight-path "$KT_GGUF_TEMPLATE" \
  --kt-threadpool-count 8 \
  --kt-cpuinfer 24 \
  --max-running-requests 1 \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${EXTRA_FLAGS}
# cuda-graph 已启用：kt-kernel ACL callback worker + kt_ep_wrapper NPU graph
# host callback（见 kt-kernel/cpu_backend/ascend_callback_worker.*）。
# frequency placement 示例：
#   EXTRA_FLAGS="--kt-expert-placement-strategy frequency --kt-activation-freq-path /path/to/activation_freq.pt"
# 调试同步路径：KT_FORCE_SYNC_SUBMIT=1；回退无 graph：EXTRA_FLAGS="--disable-cuda-graph"
# 生产勿开 KT_DEBUG_HYBRID_MOE / KT_DEBUG_MOE_OUT。graph 性能用 msprof，勿长期开 SGLANG_NPU_PROFILE_ENABLE。
