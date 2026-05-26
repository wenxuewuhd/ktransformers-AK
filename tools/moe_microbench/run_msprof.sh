#!/usr/bin/env bash
# D5: msprof 硬件层测量 + comparison 报告（不替换 Event 路径）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/env.sh"

OUT_DIR="${OUT_DIR:-${ROOT}/results}"
MSPROF_OUT="${MSPROF_OUT:-${ROOT}/npu_results}"
mkdir -p "${OUT_DIR}" "${MSPROF_OUT}"

EXTRA=()
[[ -n "${N_ACTIVE:-}" ]] && EXTRA+=(--n-active-experts "${N_ACTIVE}")
[[ "${QUICK:-0}" == "1" ]] && EXTRA+=(--quick-mode)
MSPROF_EXTRA=(--msprof --msprof-out "${MSPROF_OUT}")

run_msprof_bench() {
  local seg="$1"
  local out="${OUT_DIR}/${seg}_msprof.json"
  echo "--- msprof: ${seg} -> ${out} ---"
  "${PYTHON_BIN}" -m "moe_bench.bench_${seg}" "${MSPROF_EXTRA[@]}" --out "${out}" "${EXTRA[@]:-}"
}

# D5.2: 8 段正式 msprof
for seg in act_quant gemm_up silu_mul gemm_down routed_full shared_expert; do
  run_msprof_bench "${seg}"
done

echo "--- msprof: silu_mul fused (optional) ---"
"${PYTHON_BIN}" -m moe_bench.bench_silu_mul "${MSPROF_EXTRA[@]}" --fused --out "${OUT_DIR}/silu_mul_fused_msprof.json" "${EXTRA[@]:-}" 2>/dev/null || \
  echo "[skip] fused silu kernel unavailable"

"${PYTHON_BIN}" -m moe_bench.bench_grouped_vs_loop "${MSPROF_EXTRA[@]}" --out "${OUT_DIR}/grouped_vs_loop_msprof.json" "${EXTRA[@]:-}"

# D5.5 optional sweep
if [[ -n "${SWEEP_N_ACTIVE:-}" ]]; then
  for N in ${SWEEP_N_ACTIVE}; do
    for seg in gemm_up gemm_down routed_full; do
      echo "--- msprof sweep n_active=${N} ${seg} ---"
      N_ACTIVE="${N}" "${PYTHON_BIN}" -m "moe_bench.bench_${seg}" \
        --msprof --msprof-out "${MSPROF_OUT}/n_active_${N}" \
        --out "${OUT_DIR}/${seg}_msprof_n${N}.json" "${EXTRA[@]:-}"
    done
  done
  "${PYTHON_BIN}" -m moe_bench.report --mode msprof-n-active-sweep \
    --inputs ${OUT_DIR}/*_msprof_n*.json \
    --out "${OUT_DIR}/msprof_n_active_sweep.md"
fi

# D5.3-D5.4: comparison report
EVENT_JSONS=(
  "${OUT_DIR}/act_quant.json"
  "${OUT_DIR}/gemm_up.json"
  "${OUT_DIR}/silu_mul_unfused.json"
  "${OUT_DIR}/gemm_down.json"
  "${OUT_DIR}/routed_full.json"
  "${OUT_DIR}/shared_expert.json"
  "${OUT_DIR}/grouped_vs_loop.json"
)

"${PYTHON_BIN}" -m moe_bench.report --mode comparison \
  --msprof-jsons \
    "${OUT_DIR}/act_quant_msprof.json" \
    "${OUT_DIR}/gemm_up_msprof.json" \
    "${OUT_DIR}/silu_mul_msprof.json" \
    "${OUT_DIR}/gemm_down_msprof.json" \
    "${OUT_DIR}/routed_full_msprof.json" \
    "${OUT_DIR}/shared_expert_msprof.json" \
    "${OUT_DIR}/grouped_vs_loop_msprof.json" \
  --event-jsons "${EVENT_JSONS[@]}" \
  --out "${OUT_DIR}/msprof_vs_python_comparison.md"

echo "[moe_microbench] D5 done -> ${OUT_DIR}/msprof_vs_python_comparison.md"
