#!/usr/bin/env bash
# F2 整网冒烟：Handoff 附录 Z.8 四个 prompt（需服务已起，PORT 与 launch 一致）。
# 用法：HOST=127.0.0.1 PORT=8000 bash tools/p27_curl_f2_prompts.sh

set -euo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYBIN="${PYTHON_BIN:-${PYBIN:-/usr/local/python3.11.14/bin/python3.11}}"

"$PYBIN" - <<'PY'
import json
import urllib.request

host = __import__("os").environ.get("HOST", "127.0.0.1")
port = __import__("os").environ.get("PORT", "8000")
base = f"http://{host}:{port}/generate"

prompts = [
    (1, 64, "Below is a Python function to compute Fibonacci numbers:"),
    (
        2,
        128,
        "Explain the difference between supervised and unsupervised learning in three short paragraphs.\n\n",
    ),
    (3, 80, "请用一句话解释什么是 transformer 模型："),
    (4, 128, "什么是 transformer 模型："),
]

for pid, max_tok, text in prompts:
    print(f"========== prompt {pid} (max_new_tokens={max_tok}) ==========")
    body = {"text": text, "sampling_params": {"max_new_tokens": max_tok, "temperature": 0}}
    req = urllib.request.Request(
        base,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        print(r.read()[:2500].decode(errors="replace"))
    print()

print("[f2] done — 人工看 text：无 NaN/全感叹号/乱码即通过（base 模型不考核指令遵循）")
PY
