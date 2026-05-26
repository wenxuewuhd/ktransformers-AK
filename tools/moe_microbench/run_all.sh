#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/env.sh"

OUT_DIR="${OUT_DIR:-${ROOT}/results}"
mkdir -p "${OUT_DIR}"

EXTRA=()
[[ -n "${NUM_TOKENS:-}" ]] && EXTRA+=(--num-tokens "${NUM_TOKENS}")
[[ -n "${TOP_K:-}" ]] && EXTRA+=(--top-k "${TOP_K}")
[[ -n "${N_ACTIVE:-}" ]] && EXTRA+=(--n-active-experts "${N_ACTIVE}")
[[ -n "${TOKENS_PER_EXPERT:-}" ]] && EXTRA+=(--tokens-per-expert "${TOKENS_PER_EXPERT}")
[[ -n "${WARMUP:-}" ]] && EXTRA+=(--warmup "${WARMUP}")
[[ -n "${REPEAT:-}" ]] && EXTRA+=(--repeat "${REPEAT}")
[[ "${QUICK:-0}" == "1" ]] && EXTRA+=(--quick-mode)

if [[ "${PROFILE:-0}" == "1" ]]; then
  PROFILE_DIR="${OUT_DIR}/profile"
  EXTRA+=(--profile-dir "${PROFILE_DIR}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for K in act_quant gemm_up silu_mul gemm_down routed_full shared_expert grouped_vs_loop; do
    echo "--- dry-run: $K ---"
    "${PYTHON_BIN}" -m "moe_bench.bench_${K}" --dry-run "${EXTRA[@]:-}"
  done
  exit 0
fi

run_bench() {
  local kind="$1"
  local outdir="${2:-${OUT_DIR}}"
  echo "--- run: ${kind} -> ${outdir} ---"
  "${PYTHON_BIN}" -m "moe_bench.bench_${kind}" --out "${outdir}/${kind}.json" "${EXTRA[@]:-}"
}

if [[ -n "${SWEEP_N_ACTIVE:-}" ]]; then
  for NA in ${SWEEP_N_ACTIVE}; do
    SD="${OUT_DIR}/n_active_${NA}"
    mkdir -p "${SD}"
    N_ACTIVE="${NA}" run_bench gemm_up "${SD}"
    N_ACTIVE="${NA}" run_bench gemm_down "${SD}"
    N_ACTIVE="${NA}" run_bench routed_full "${SD}"
  done
fi

run_bench act_quant
run_bench gemm_up
echo "--- run: silu_mul_unfused ---"
"${PYTHON_BIN}" -m moe_bench.bench_silu_mul --no-fused --out "${OUT_DIR}/silu_mul_unfused.json" "${EXTRA[@]:-}"
echo "--- run: silu_mul (auto/fused probe) ---"
if "${PYTHON_BIN}" -m moe_bench.bench_silu_mul --fused --out "${OUT_DIR}/silu_mul_fused.json" "${EXTRA[@]:-}" 2>/dev/null; then
  :
else
  cp "${OUT_DIR}/silu_mul_unfused.json" "${OUT_DIR}/silu_mul_fused.json" 2>/dev/null || true
fi
run_bench gemm_down
run_bench routed_full
run_bench shared_expert
run_bench grouped_vs_loop

REPORT_INPUTS=(
  "${OUT_DIR}/act_quant.json"
  "${OUT_DIR}/gemm_up.json"
  "${OUT_DIR}/silu_mul_unfused.json"
  "${OUT_DIR}/silu_mul_fused.json"
  "${OUT_DIR}/gemm_down.json"
  "${OUT_DIR}/routed_full.json"
  "${OUT_DIR}/shared_expert.json"
  "${OUT_DIR}/grouped_vs_loop.json"
)

SWEEP_ARG=()
if [[ -n "${SWEEP_N_ACTIVE:-}" ]]; then
  SWEEP_ARG=(--sweep-dir "${OUT_DIR}")
fi

"${PYTHON_BIN}" -m moe_bench.report \
  --inputs "${REPORT_INPUTS[@]}" \
  "${SWEEP_ARG[@]}" \
  --out "${OUT_DIR}/summary_decode.md"

echo "[moe_microbench] done -> ${OUT_DIR}/summary_decode.md"
