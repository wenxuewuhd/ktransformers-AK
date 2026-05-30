#!/usr/bin/env bash
# 32k 长 prompt prefill + decode profiling 专用 launch。
#
# 关键：CHUNKED_PREFILL_SIZE 须 >= 目标 prompt 且为 page_size=128 倍数（默认 32768）。
# 32k 单 chunk prefill 依赖 ascend_backend.py 的 q.squeeze(0) 守护（T=1 chunk 不压维）。
#
# 终端 1 — seq32k decode #32 profiling：
#   SGLANG_NPU_PROFILE_ENABLE=1 \
#   SGLANG_NPU_PROFILE_DECODE_TOKEN=32 \
#   SGLANG_NPU_PROFILE_DIR=./tools/npu_results_dbg/seq32k_decode32 \
#   CHUNKED_PREFILL_SIZE=32768 \
#   PORT=8001 ASCEND_RT_VISIBLE_DEVICES=2 \
#   ./tools/p27_launch_ds4flash_npu_longcontext.sh
#
# 终端 2：
#   PORT=8001 PROMPT_LEN=32768 MAX_NEW=64 bash tools/p27_curl_long_prompt_sweep.sh
#
# 注意：MAX_NEW 必须 >= SGLANG_NPU_PROFILE_DECODE_TOKEN，否则 profile 不触发。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}"
export USE_PA_PREFILL="${USE_PA_PREFILL:-1}"
export USE_PA_DECODE="${USE_PA_DECODE:-1}"

echo "[p27-longctx] CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE} (prompt 须 <= 此值且为 128 倍数)"
echo "[p27-longctx] USE_PA_PREFILL=${USE_PA_PREFILL} USE_PA_DECODE=${USE_PA_DECODE}"

exec "${SCRIPT_DIR}/p27_launch_ds4flash_npu_num_expert_0.sh" "$@"
