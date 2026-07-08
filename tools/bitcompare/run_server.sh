#!/usr/bin/env bash
# Boot the DSV4-Flash single-card service for the bit-level regression harness, on a
# spare port/device, with this box's actual model + mxfp4 GGUF paths. Thin wrapper over
# the production launcher (tools/p27_launch_ds4flash_npu.sh) so the captured golden goes
# through the exact production code path. Env vars below can be overridden.
set -euo pipefail
export REPO="${REPO:-/mnt/workspace/gitCode/kt-cleancode}"
export MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-W8A8}"
# NOTE: do NOT write ${KT_GGUF_TEMPLATE:-...{layer_idx}...} — bash treats the first '}'
# of {layer_idx} as the end of the ${...:-} default (same trap the launcher documents).
if [[ -z "${KT_GGUF_TEMPLATE:-}" ]]; then
  KT_GGUF_TEMPLATE='/mnt/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf'
fi
export KT_GGUF_TEMPLATE
export NPU_DEVICE_ID="${NPU_DEVICE_ID:-0}"     # logical die on the single 910C (0 or 1); both free
# This box has the CANN 9.0.0 PUBLIC 18-arg compressor op; the code defaults to the private
# 19-arg split-state ABI, which fails graph capture here. Select the single-state ABI.
export KT_NSA_COMPRESSOR_MODE="${KT_NSA_COMPRESSOR_MODE:-single}"
export PORT="${PORT:-8021}"                     # avoid main service ports 8000/8020
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
# This box is a single-NUMA, 40-core 910C(A3) host. The launcher's 8-NUMA defaults
# (threadpool=8, cpuinfer=128) bind subpools to NUMA nodes 1..7 that don't exist here
# (log shows "alloc N from other numa" fallbacks). Match the hardware: 1 pool, <=cores.
export KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-1}"
export KT_CPUINFER="${KT_CPUINFER:-32}"
exec bash "${REPO}/tools/p27_launch_ds4flash_npu.sh"
