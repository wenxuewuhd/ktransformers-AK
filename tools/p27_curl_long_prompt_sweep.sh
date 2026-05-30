#!/usr/bin/env bash
# 通过 input_ids 灌任意长度 prompt，不依赖 tokenizer，不占客户端 NPU。
#
# 用法：
#   PORT=8001 PROMPT_LEN=32768 MAX_NEW=64 bash tools/p27_curl_long_prompt_sweep.sh
#   PORT=8001 SWEEP=1 bash tools/p27_curl_long_prompt_sweep.sh   # 1k→32k sweep
#
# 32k JSON body ~200KB，用 python 写临时文件 + curl --data-binary @file 避免 ARG_MAX。
set -euo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_NEW="${MAX_NEW:-1}"
FILLER_ID="${FILLER_ID:-100}"

_one() {
  local _len=$1
  echo "===== prompt_len=${_len} max_new=${MAX_NEW} ====="
  local _body_file _out _start _code
  _body_file=$(mktemp)
  python3 -c "
import json, sys
n = int(sys.argv[1]); fid = int(sys.argv[2]); mx = int(sys.argv[3])
body = {'input_ids': [fid]*n, 'sampling_params': {'max_new_tokens': mx, 'temperature': 0}}
open(sys.argv[4], 'w').write(json.dumps(body))
" "${_len}" "${FILLER_ID}" "${MAX_NEW}" "${_body_file}"
  _out=$(mktemp)
  _start=$SECONDS
  _code=$(curl -sS -o "${_out}" -w '%{http_code}' \
    --max-time 7200 \
    -X POST "http://${HOST}:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    --data-binary @"${_body_file}") || {
    echo "[sweep][${_len}] curl failed"
    rm -f "${_out}" "${_body_file}"
    return 1
  }
  echo "[sweep][${_len}] HTTP=${_code} elapsed=$((SECONDS - _start))s"
  rm -f "${_body_file}"
  if [[ "${_code}" != "200" ]]; then
    head -c 2000 "${_out}"
    rm -f "${_out}"
    return 2
  fi
  head -c 400 "${_out}"
  echo
  rm -f "${_out}"
}

if [[ "${SWEEP:-0}" == "1" ]]; then
  for L in 1024 2048 4096 8192 16384 32768; do
    _one "${L}" || {
      echo "[sweep] STOPPED at len=${L}"
      exit 1
    }
  done
else
  _one "${PROMPT_LEN:-32768}"
fi
