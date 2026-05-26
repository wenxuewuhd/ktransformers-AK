#!/usr/bin/env bash
# Synthetic 32k SWA / CSA / HCA attention microbench（独立工作区，不改主干）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/env.sh"

SEQ_LEN="${SEQ_LEN:-32768}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUT_DIR="${OUT_DIR:-${ROOT}/results}"
SANITY_FLAG="${SANITY_FLAG:---sanity}"
REPEAT="${REPEAT:-}"
WARMUP="${WARMUP:-}"

mkdir -p "${OUT_DIR}"

EXTRA=()
[[ -n "${REPEAT}" ]] && EXTRA+=(--repeat "${REPEAT}")
[[ -n "${WARMUP}" ]] && EXTRA+=(--warmup "${WARMUP}")

_run_one() {
  local seq="$1"
  local out="${OUT_DIR}"
  if [[ -n "${SEQ_LEN_SWEEP:-}" ]]; then
    out="${OUT_DIR}/seq_${seq}"
    mkdir -p "${out}"
  fi
  echo "[attn_microbench] seq_len=${seq} batch=${BATCH_SIZE} out=${out}"
  local COMMON=(--seq-len "${seq}" --batch-size "${BATCH_SIZE}" "${EXTRA[@]:-}")

  echo "[attn_microbench] SWA ..."
  "${PYTHON_BIN}" -m attn_bench.bench_swa "${COMMON[@]}" ${SANITY_FLAG} --out "${out}/swa.json"

  echo "[attn_microbench] CSA ..."
  if [[ "${SKIP_INDEXER:-0}" == "1" ]]; then
    "${PYTHON_BIN}" -m attn_bench.bench_csa "${COMMON[@]}" ${SANITY_FLAG} --skip-indexer --out "${out}/csa.json"
  else
    "${PYTHON_BIN}" -m attn_bench.bench_csa "${COMMON[@]}" ${SANITY_FLAG} --out "${out}/csa.json" || {
      echo "[attn_microbench][WARN] CSA full path failed; retry attn-only (--skip-indexer)" >&2
      "${PYTHON_BIN}" -m attn_bench.bench_csa "${COMMON[@]}" ${SANITY_FLAG} --skip-indexer --out "${out}/csa_attn_only.json"
    }
  fi

  echo "[attn_microbench] HCA ..."
  "${PYTHON_BIN}" -m attn_bench.bench_hca "${COMMON[@]}" ${SANITY_FLAG} --out "${out}/hca.json"

  local INPUTS=("${out}/swa.json" "${out}/hca.json")
  [[ -f "${out}/csa.json" ]] && INPUTS+=("${out}/csa.json")
  [[ -f "${out}/csa_attn_only.json" ]] && INPUTS+=("${out}/csa_attn_only.json")

  "${PYTHON_BIN}" -m attn_bench.report \
    --inputs "${INPUTS[@]}" \
    --out "${out}/summary_${seq}.md"
  echo "[attn_microbench] done -> ${out}/summary_${seq}.md"
}

if [[ -n "${SEQ_LEN_SWEEP:-}" ]]; then
  for s in ${SEQ_LEN_SWEEP}; do
    _run_one "${s}"
  done
  SWEEP_SUMMARY="${OUT_DIR}/summary_seq_sweep_r${REPEAT:-1000}.md"
  {
    echo "# Seq sweep summary (repeat=${REPEAT:-1000})"
    echo ""
    echo "| seq_len | indexer mean±std (µs) | attn mean±std (µs) |"
    echo "|---------|----------------------|-------------------|"
    for s in ${SEQ_LEN_SWEEP}; do
      csa="${OUT_DIR}/seq_${s}/csa.json"
      if [[ -f "${csa}" ]]; then
        "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
d = json.loads(Path("${csa}").read_text())
idx = d.get("indexer_us") or {}
attn = d.get("attn_us") or {}
def fmt(x):
    if not x: return "n/a"
    return f"{x.get('mean', x.get('device_mean_us', 0)):.1f} ± {x.get('device_std_us', 0):.1f}"
print(f"| ${s} | {fmt(idx)} | {fmt(attn)} |")
PY
      fi
    done
  } > "${SWEEP_SUMMARY}"
  echo "[attn_microbench] sweep summary -> ${SWEEP_SUMMARY}"
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m attn_bench.bench_swa --dry-run --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}"
  "${PYTHON_BIN}" -m attn_bench.bench_csa --dry-run --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}"
  "${PYTHON_BIN}" -m attn_bench.bench_hca --dry-run --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}"
  exit 0
fi

_run_one "${SEQ_LEN}"
