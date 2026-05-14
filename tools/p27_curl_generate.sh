#!/usr/bin/env bash
# P2.7：对已启动的 HTTP 服务发一条 /generate 冒烟（需与 launch 脚本 PORT 一致）。
# 用法：HOST=127.0.0.1 PORT=8000 bash tools/p27_curl_generate.sh

set -euo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
curl -sS -X POST "http://${HOST}:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text": "你好，", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}' | head -c 4000
echo
