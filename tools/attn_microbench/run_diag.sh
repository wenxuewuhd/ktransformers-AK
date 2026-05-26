#!/usr/bin/env bash
# P0 seq scaling diagnostics (Claude review). Uses repeat=100 by default for speed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/env.sh"

REPEAT="${REPEAT:-100}"
WARMUP="${WARMUP:-30}"
SEQ_LEN="${SEQ_LEN:-32768}"
OUT="${OUT:-${ROOT}/results/diag_seq_scaling.json}"

echo "[diag] repeat=${REPEAT} warmup=${WARMUP} seq_len=${SEQ_LEN}"
"${PYTHON_BIN}" -m attn_bench.bench_diag \
  --seq-len "${SEQ_LEN}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --scenario "${SCENARIO:-all}" \
  --out "${OUT}"

echo "[diag] reference check (seq=512) ..."
"${PYTHON_BIN}" -m attn_bench.reference_check --seq-len 512
