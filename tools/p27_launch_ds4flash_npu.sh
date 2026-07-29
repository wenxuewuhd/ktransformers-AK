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
#   KT_GGUF_TEMPLATE  默认 dsv4_layer{layer_idx}.gguf(Q8_0);KT_MXFP4_DEPOOL=1 时默认改 _mxfp4.gguf
#                     (CPU MoE 带宽bound,Q8_0 让 depool decode 慢~2×;见 memory depool-decode-needs-mxfp4-gguf)
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

# MODEL_PATH 默认:自动探测(本盒 /mnt/workspace、旧镜像 /workspace),显式 env 仍优先。
if [[ -z "${MODEL_PATH:-}" ]]; then
  for _c in /mnt/workspace/models/DeepSeek-V4-Flash-W8A8 \
            /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8; do
    [[ -d "$_c" ]] && { MODEL_PATH="$_c"; break; }
  done
  MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8}"
fi

# Streaming-prefill checkpoints (only read when KT_PREFILL_STREAM=1). kt_stream_prefill.py
# reads config.json / model.safetensors.index.json from these; its own defaults are the old
# image's /workspace/... paths, so DERIVE them from MODEL_PATH here (any new container just
# needs MODEL_PATH). _CKPT = the W8A8 serving ckpt = MODEL_PATH; _MXFP4_CKPT = the native
# MXFP4 source = sibling dir with the -W8A8 suffix stripped. Explicit env still wins.
export KT_PREFILL_STREAM_CKPT="${KT_PREFILL_STREAM_CKPT:-$MODEL_PATH}"
export KT_MXFP4_CKPT="${KT_MXFP4_CKPT:-${MODEL_PATH%-W8A8}}"

# ---------- KT MoE 最优全量默认（2026-06 验证；任一可被显式 env 覆盖）----------
# 全开 = depool + dynamic-hot + 流式prefill + side-stream + GGUF dedup：
#   * 精度对齐 PR（GPQA off 72–75%，commit e5f53ad 根治异步竞态后 force-sync=0 即对）；
#   * decode ~18 tok/s（dynamic 热专家 + side-stream 重叠 + mask-remap 修复）；
#   * host DDR ~146G（dedup 复用 CPU 已 mmap 的 GGUF，省 ~137G；NPU 常驻从同源 mxfp4 现转 → 与 CPU 逐 bit 同）。
#   * KT_FORCE_SYNC_SUBMIT 不设（=0）：异步竞态已根治，关 = 又对又快；设 1 只是慢路径。
# 想要轻量 prefix-32 baseline（~16 tok/s、不建 mxfp4 池）：
#   显式 KT_MXFP4_DEPOOL=0 KT_MXFP4_GGUF_DEDUP=0 KT_DYNAMIC_RESIDENT=0 KT_PREFILL_STREAM=0
# 见 memory depool-dynamic-correct-convert-folded / gguf-dedup-saves-137g / prefill-async-race-fixed。
export KT_MXFP4_DEPOOL="${KT_MXFP4_DEPOOL:-1}"
export KT_MXFP4_GGUF_DEDUP="${KT_MXFP4_GGUF_DEDUP:-1}"   # 依赖 depool；默认 GGUF 模板下面会选 mxfp4
export KT_DYNAMIC_RESIDENT="${KT_DYNAMIC_RESIDENT:-1}"
export KT_PREFILL_STREAM="${KT_PREFILL_STREAM:-1}"
export KT_SIDE_STREAM="${KT_SIDE_STREAM:-1}"
# NSA compressor 算子 ABI 选择器(非调优项,绑 CANN 版本):single=CANN 9.0.0+ 公开 18 参数/单交织
# buffer;split=CANN 8.5.0 私有 19 参数/双分离 buffer。两者不兼容、无运行时回退(见 nsa_compressor_mode.py)。
# 按 CANN 主版本自动探测(从 ASCEND_TOOLKIT_HOME/HOME_PATH 路径里的 cann-X):>=9 → single,否则 split。
# 这样 910C(cann-9.0.0)与 910B(cann-8.5.0)裸启动都对;显式 KT_NSA_COMPRESSOR_MODE 仍优先。
if [[ -z "${KT_NSA_COMPRESSOR_MODE:-}" ]]; then
  _cann_major="$(echo "${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}" | grep -oE 'cann-[0-9]+' | grep -oE '[0-9]+' | head -1)"
  if [ "${_cann_major:-0}" -ge 9 ]; then KT_NSA_COMPRESSOR_MODE=single; else KT_NSA_COMPRESSOR_MODE=split; fi
fi
export KT_NSA_COMPRESSOR_MODE
# 流式 prefill 的最小 chunk 长度门槛(kt_stream_prefill.py:_T)。prefill chunk 的 token 数 >= 此值
# 才走流式 prefill;而 **dynamic hot expert(KT_DYNAMIC_RESIDENT)的常驻热专家槽正是在流式 prefill
# 路径里刷新的** —— 所以低于门槛的短 prefill 走 hybrid、那一发不更新热专家。想让更短的 prompt 也
# 享受流式+动态热专家,把它调小(如 128);调大则只有更长 prefill 才流式。默认 512(=代码默认,零回归)。
export KT_PREFILL_STREAM_THRESHOLD="${KT_PREFILL_STREAM_THRESHOLD:-512}"

# 勿用 KT_GGUF_TEMPLATE="${KT:-...dsv4_layer{layer_idx}.gguf}"：bash 会把 {layer_idx} 的第一个 ``}`` 当成 ``${...:-}`` 的结束符，路径会变成 ``...{layer_idx.gguf}``。
# 默认 Q8_0（批量 convert 输出 dsv4_layer{L}.gguf）。须先 cp 新 kt_kernel_ext.so 到 kt-kernel/python/（手册 §2.4）。
# BF16 回退：export KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_bf16.gguf'
# GGUF 权重缓存目录:自动探测(本盒 /mnt/workspace、旧镜像 /workspace),可用 KT_GGUF_CACHE_DIR 覆盖。
if [[ -z "${KT_GGUF_CACHE_DIR:-}" ]]; then
  for _c in /mnt/workspace/models/cache /workspace/models/cache; do
    [[ -d "$_c" ]] && { KT_GGUF_CACHE_DIR="$_c"; break; }
  done
  KT_GGUF_CACHE_DIR="${KT_GGUF_CACHE_DIR:-/workspace/models/cache}"
fi
if [[ -z "${KT_GGUF_TEMPLATE:-}" ]]; then
  if [[ "${KT_MXFP4_DEPOOL:-}" == "1" ]]; then
    # depool：NPU 侧本就 MXFP4，CPU 专家也走 MXFP4 GGUF（3.4GB/层 vs Q8_0 6.8GB/层）。CPU MoE 是
    # 内存带宽 bound，Q8_0 默认会让 depool decode off_cpu ~2×(22 vs 14ms,13 vs 16tps);MXFP4 ≈ plain。
    # 见 memory depool-decode-needs-mxfp4-gguf。显式 export KT_GGUF_TEMPLATE 仍优先(不被本默认覆盖)。
    KT_GGUF_TEMPLATE="$KT_GGUF_CACHE_DIR/dsv4_layer{layer_idx}_mxfp4.gguf"
  else
    KT_GGUF_TEMPLATE="$KT_GGUF_CACHE_DIR/dsv4_layer{layer_idx}.gguf"
  fi
fi
# ★必须 export：除了 --kt-weight-path（CPU MoE）外，GGUF dedup（KT_MXFP4_GGUF_DEDUP=1）从
# os.environ 读 KT_GGUF_TEMPLATE 来复用 CPU 的 mmap GGUF；不 export → dedup 报 "template empty" 回退建 codes 池。
export KT_GGUF_TEMPLATE
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"   # ≥ 常见 prompt(GPQA max 2577)，避坑⑯ NSA 跨 chunk 崩；32k/64k 长序列须显式调更大
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
# CPU MoE is memory-bandwidth-bound; scale threads to raise effective DDR bandwidth.
# Default is now 128 (16/NUMA). Isolated decode micro-bench (tools/p27_cpu_moe_bw_bench.py,
# real layer3 weights, output verified) shows effective bandwidth has a KNEE at 128 then a
# noisy plateau: 96(12/NUMA)=88, 112(14)=96, 128(16)=114, 144=109, 160(20)=110, 176=116 GB/s;
# only 192 (24/NUMA = ALL cores) COLLAPSES (no spare core for the NumaJobDistributor spin
# threads + NPU host callback + python/OS -> oversubscription thrash).
# End-to-end server (single card, 32 GPU experts): 96 -> 128 cuts CPU MoE 67.7 -> 55.1
# ms/token, decode 6.84 -> 8.52 tok/s (+24%), F2 coherent (accuracy preserved). 128 and 160
# give IDENTICAL decode throughput (CPU MoE is ~co-equal/overlapped with NPU past 128), so 128
# wins on safety: 8 cores/NUMA headroom vs 160's 4. The old "<=96, >=128 thrashes" note was a
# live-server-contention artifact, not intrinsic. Override with KT_CPUINFER (160 fine too).
# (profiling: doc/zh/dsv4_single_npu/graph_decode_bandwidth_handoff.md, 2026-06-09)
#
# 默认改为「按机器自动探测」(显式 env 仍优先):
#   * KT_THREADPOOL_COUNT = 实际 NUMA 节点数(kt-kernel 每节点一个子池并绑核;必须 == 机器 NUMA 数,
#     否则 "NUMA node N not found" + set_mempolicy 失败)。参考镜像=8;本盒 910C(A3)=1。
#   * KT_CPUINFER = 总核数 − 8×NUMA(每 NUMA 留 8 核给 NumaJobDistributor 自旋线程 + NPU host 回调 +
#     python/OS,避免占满核 thrash)。参考镜像 192−64=128(=旧默认);本盒 40−8=32。threads/subpool =
#     KT_CPUINFER / KT_THREADPOOL_COUNT。占满全核会崩(无余量给 NPU host 线程)。
_NUMA_N="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)"; [ "${_NUMA_N:-0}" -lt 1 ] && _NUMA_N=1
_NCORE="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
_INFER=$(( _NCORE - 8 * _NUMA_N )); [ "$_INFER" -lt 1 ] && _INFER="$_NCORE"
KT_CPUINFER="${KT_CPUINFER:-$_INFER}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-$_NUMA_N}"
PORT="${PORT:-8000}"
# ASCEND_TOOLKIT_HOME is the SINGLE anchor — ATB(nnal), opp vendors, custom-op paths all derive
# from it. Take it from the environment (CANN's set_env.sh exports both ASCEND_TOOLKIT_HOME and
# ASCEND_HOME_PATH); if absent, auto-detect known layouts; if still not found, HARD-FAIL with
# guidance instead of silently using a wrong hardcoded path (that's what broke on new containers).
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH:-}"
fi
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  for _cand in /usr/local/Ascend/ascend-toolkit/latest \
               "${HOME}"/Ascend/ascend-toolkit/latest \
               /home/developer/Ascend/cann-9.0.0 \
               /home/*/Ascend/cann-* ; do
    [[ -f "${_cand}/set_env.sh" ]] && { ASCEND_TOOLKIT_HOME="${_cand}"; break; }
  done
  unset _cand
fi
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" || ! -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
  echo "[p27][ERROR] 未拿到有效的 ASCEND_TOOLKIT_HOME(CANN 根)。请显式设置,或先 source CANN 的 set_env.sh:" >&2
  echo "  export ASCEND_TOOLKIT_HOME=/home/developer/Ascend/cann-9.0.0   # ← 按本机实际路径" >&2
  exit 1
fi
export ASCEND_TOOLKIT_HOME
echo "[p27] ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME}（ATB/vendors 由此派生）"
# KML (Kunpeng Math Library) — only on Kunpeng hosts; prepend only if present so a missing
# dir doesn't sit dead in LD_LIBRARY_PATH. Override with KML_LIB_DIR.
_KML_LIB_DIR="${KML_LIB_DIR:-/usr/local/kml/lib}"
[[ -d "${_KML_LIB_DIR}" ]] && export LD_LIBRARY_PATH="${_KML_LIB_DIR}:${LD_LIBRARY_PATH:-}"

# Drop any inherited proxy. sglang's startup warmup (SKIP_WARMUP=0) POSTs to the server's OWN
# port; with http_proxy=127.0.0.1:7890 set, that localhost call is intercepted -> 502 ->
# "warmup error: AssertionError res=<Response [502]>" -> Initialization failed (server exits).
# This is exactly why warmup kept being disabled with SKIP_WARMUP=1. A local inference server
# never needs an outbound proxy, so just unset them (also spares every curl the --noproxy dance).
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

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

# CANN toolkit + ATB + 自定义算子 vendor 环境（best-effort）。让本脚本自包含，不再隐式依赖 shell
# profile（.bashrc/profile 才 source 它们）：非交互/非登录 shell 或干净 container 直接拉起也能找到算子。
# 机器无 vendors/config.ini，自定义算子靠 ASCEND_CUSTOM_OPP_PATH（由各 set_env.bash 设），故必须 source。
# ⚠️ 这些 vendor 脚本不是 set -u 干净的（如 atb/set_env.sh 引用未定义的 ZSH_VERSION）→ 在本脚本的
#    `set -euo pipefail` 下会触发 unbound variable 直接退出。故 source 期间临时放开 -e/-u，之后恢复。
set +eu
ASCEND_OPP_VENDORS_DIR="${ASCEND_TOOLKIT_HOME}/opp/vendors"
# ATB (nnal) set_env location is NOT under the toolkit dir and differs per image:
#   reference image /usr/local/Ascend/nnal/...  vs  this box /home/developer/Ascend/nnal/...
# It's the sibling `nnal/atb` of the Ascend root (= dir that holds the toolkit). Honor an
# explicit ATB_SET_ENV, else search derived + known locations; silent-skip only if none exist
# (then ATB ops won't register — set ATB_SET_ENV to fix).
if [[ -z "${ATB_SET_ENV:-}" ]]; then
  _ascend_root="$(dirname "${ASCEND_TOOLKIT_HOME}")"                 # e.g. /home/developer/Ascend/cann-9.0.0 -> .../Ascend
  [[ "$(basename "${_ascend_root}")" == "ascend-toolkit" ]] && _ascend_root="$(dirname "${_ascend_root}")"  # /usr/local/Ascend/ascend-toolkit -> /usr/local/Ascend
  for _atb in "${_ascend_root}/nnal/atb/set_env.sh" \
              /home/developer/Ascend/nnal/atb/set_env.sh \
              /usr/local/Ascend/nnal/atb/set_env.sh; do
    [[ -f "${_atb}" ]] && { ATB_SET_ENV="${_atb}"; break; }
  done
fi
[[ -n "${ATB_SET_ENV:-}" ]] && echo "[p27] ATB set_env: ${ATB_SET_ENV}" || echo "[p27][warn] ATB set_env not found (ATB ops may be unavailable); set ATB_SET_ENV=<path> if needed"
for _kt_env in \
  "${ASCEND_TOOLKIT_HOME}/set_env.sh" \
  "${ATB_SET_ENV:-/nonexistent}" \
  "${ASCEND_OPP_VENDORS_DIR}/customize/bin/set_env.bash" \
  "${ASCEND_OPP_VENDORS_DIR}/custom_transformer/bin/set_env.bash"; do
  if [[ -f "${_kt_env}" ]]; then
    # shellcheck source=/dev/null
    source "${_kt_env}"
  fi
done
unset _kt_env _atb _ascend_root
set -eu

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
# SKIP_WARMUP=1 传 --skip-server-warmup 跳过开机预热；=0 开启预热。
# 流式自暖（2026-06-25）：流式 prefill 从不跑 CPU MoE(kt_kernel)，所以 stream-everything 的服务会一直
# 冷、decode ~11 而非 ~18（drop_caches 证明是 kt_kernel 进程内状态、非页缓存）。修法=启动时用一发
# 强制-hybrid 的 forward 把 kt_kernel 暖起来：开 sglang 内部 warmup(SKIP_WARMUP=0) + 强制那一发走
# hybrid(KT_STREAM_WARMUP=1)。端到端实测：之后第一发长 prompt 也能流式且 decode ~18；暖机一发即够、
# 终生不复冷。故 KT_PREFILL_STREAM=1 时两者默认开（均可被显式覆盖；非流式跑维持基线 warmup 关）。
if [[ "${KT_PREFILL_STREAM:-}" == "1" ]]; then
  SKIP_WARMUP="${SKIP_WARMUP:-0}"
  export KT_STREAM_WARMUP="${KT_STREAM_WARMUP:-1}"
fi
SKIP_WARMUP="${SKIP_WARMUP:-1}"
WARMUP_FLAG="--skip-server-warmup"
if [[ "${SKIP_WARMUP}" == "0" ]]; then
  WARMUP_FLAG=""
fi
echo "[p27] SKIP_WARMUP=${SKIP_WARMUP} (warmup_flag='${WARMUP_FLAG}') KT_STREAM_WARMUP=${KT_STREAM_WARMUP:-0}"
# 可调 env（2026-06-11 加）：
#   KT_NUM_GPU_EXPERTS  每层放 NPU 的 expert 数，默认 32。每多 1 个 ≈ +1.0GB HBM。
#       实测上限（context 65536）：40 可起（KV max_total=135k，仍≥2×context），42 崩
#       （SWA 多池 c128 盘口算负）。想更多须降 --context-length 或改 KV 分配算法。
#   MEM_FRACTION  默认 0.85。它是 (权重 + KV 池) 的上限；权重是固定的 48.3GB，所以它实际
#       只在「KV 池 ↔ 激活余量」之间挪。910C/A3 实测（ctx 65536）：
#         0.85 → KV 3.66GB（max_total_num_tokens 577,536），激活余量 ~8.8GB
#         0.82 → KV ~1.9GB（max_total_num_tokens 258,048），激活余量 ~10.6GB
#       ★ 调高没用（KV 池有上限，0.85/0.92 avail 相同）；★ 调低有用（能给激活腾地）。
#       ⚠️ 长上下文必看：32k 单块 prefill 的激活峰值 ~10GB，默认 0.85 只剩 ~8.8GB → 会 OOM。
#       跑 32k 请用 MEM_FRACTION=0.82（本仓 32k 实测即用此值）。详见总纲 §7.1.1。
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
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  --disable-radix-cache \
  --max-prefill-tokens 65535 \
  --context-length 65536 \
  --watchdog-timeout 18000 \
  ${WARMUP_FLAG} \
  --kt-method LLAMAFILE \
  --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS:-32}" \
  --kt-weight-path "$KT_GGUF_TEMPLATE" \
  --kt-threadpool-count "$KT_THREADPOOL_COUNT" \
  --kt-cpuinfer "$KT_CPUINFER" \
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
