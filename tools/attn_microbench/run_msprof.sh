#!/usr/bin/env bash
# P1.7: msprof hardware-only timing (not run by run_all.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/env.sh"

SEQ_LEN="${SEQ_LEN:-32768}"
MSPROF_OUT="${MSPROF_OUT:-${ROOT}/npu_results}"
OUT_DIR="${OUT_DIR:-${ROOT}/results}"

mkdir -p "${MSPROF_OUT}" "${OUT_DIR}"

echo "[run_msprof] seq_len=${SEQ_LEN} trace_root=${MSPROF_OUT}"

if [[ -n "${SEQ_LEN_SWEEP:-}" ]]; then
  for S in ${SEQ_LEN_SWEEP}; do
    echo "[run_msprof] CSA seq=${S} ..."
    "${PYTHON_BIN}" -m attn_bench.bench_csa --seq-len "${S}" --msprof \
      --msprof-out "${MSPROF_OUT}/seq_${S}" \
      --out "${OUT_DIR}/csa_msprof_seq_${S}.json"
  done
  "${PYTHON_BIN}" -m attn_bench.report --mode msprof-sweep \
    --inputs "${OUT_DIR}/csa_msprof_seq_*.json" \
    --out "${OUT_DIR}/msprof_seq_sweep.md"
  echo "[run_msprof] sweep -> ${OUT_DIR}/msprof_seq_sweep.md"
  exit 0
fi

echo "[run_msprof] SWA ..."
"${PYTHON_BIN}" -m attn_bench.bench_swa --seq-len "${SEQ_LEN}" --msprof \
  --msprof-out "${MSPROF_OUT}" --out "${OUT_DIR}/swa_msprof.json"

echo "[run_msprof] CSA ..."
"${PYTHON_BIN}" -m attn_bench.bench_csa --seq-len "${SEQ_LEN}" --msprof \
  --msprof-out "${MSPROF_OUT}" --out "${OUT_DIR}/csa_msprof.json"

echo "[run_msprof] HCA ..."
"${PYTHON_BIN}" -m attn_bench.bench_hca --seq-len "${SEQ_LEN}" --msprof \
  --msprof-out "${MSPROF_OUT}" --out "${OUT_DIR}/hca_msprof.json"

# Event JSONs: prefer seq sweep dir or root results
EVENT_SWA="${OUT_DIR}/swa.json"
EVENT_CSA="${OUT_DIR}/csa.json"
EVENT_HCA="${OUT_DIR}/hca.json"
[[ -f "${OUT_DIR}/seq_${SEQ_LEN}/swa.json" ]] && EVENT_SWA="${OUT_DIR}/seq_${SEQ_LEN}/swa.json"
[[ -f "${OUT_DIR}/seq_${SEQ_LEN}/csa.json" ]] && EVENT_CSA="${OUT_DIR}/seq_${SEQ_LEN}/csa.json"
[[ -f "${OUT_DIR}/seq_${SEQ_LEN}/hca.json" ]] && EVENT_HCA="${OUT_DIR}/seq_${SEQ_LEN}/hca.json"

if [[ ! -f "${EVENT_SWA}" || ! -f "${EVENT_CSA}" || ! -f "${EVENT_HCA}" ]]; then
  echo "[run_msprof] Event JSON missing; running Event mode once (repeat from yaml) ..."
  SANITY_FLAG="" bash "${ROOT}/run_all.sh"
fi

"${PYTHON_BIN}" -m attn_bench.report --mode comparison \
  --msprof-jsons "${OUT_DIR}/swa_msprof.json" "${OUT_DIR}/csa_msprof.json" "${OUT_DIR}/hca_msprof.json" \
  --event-jsons "${EVENT_SWA}" "${EVENT_CSA}" "${EVENT_HCA}" \
  --out "${OUT_DIR}/msprof_vs_python_comparison.md"

"${PYTHON_BIN}" "${ROOT}/scripts/summarize_network_hw.py" \
  --out "${OUT_DIR}/network_hw_estimate.md"

echo "[run_msprof] done -> ${OUT_DIR}/msprof_vs_python_comparison.md"
echo "[run_msprof]       -> ${OUT_DIR}/network_hw_estimate.md"
