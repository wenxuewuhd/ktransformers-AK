#!/usr/bin/env bash
# Apply split patch series in order. Usage: apply_series.sh <main_repo|llama_cpp> [REPO_ROOT]
set -euo pipefail
SUB="${1:?usage: apply_series.sh main_repo|llama_cpp [REPO_ROOT]}"
REPO="${2:-$(cd "$(dirname "$0")/../.." && pwd)}"
DIR="$(dirname "$0")/${SUB}"
SERIES="${DIR}/SERIES"
if [[ ! -f "$SERIES" ]]; then
  echo "missing $SERIES" >&2
  exit 1
fi
cd "$REPO"
while IFS= read -r patch; do
  [[ -z "$patch" || "$patch" =~ ^# ]] && continue
  echo "[apply] $patch"
  git apply --3way "${DIR}/${patch}" || git am "${DIR}/${patch}"
done < "$SERIES"
echo "[done] $SUB"