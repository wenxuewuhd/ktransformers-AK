#!/usr/bin/env bash
# 32k 长序列测试用例：构 ~32k token 的 prompt（需真正读长上下文的问题），输出 ~1000 token，
# 报 prefill TTFT / decode（长上下文下的 ms·token 与 tok/s）/ 总吞吐。用于量长序列推理性能。
#
# 前提：服务已起，且 context-length ≥ prompt+输出（默认 launch 是 65536，32k+1000 够）。
#   ⚠️ 32k 必须 CHUNKED_PREFILL_SIZE >= 实际 prompt_tokens（单 chunk），否则撞 compressor 跨 chunk bug
#      （报错 loc.numel()=1024 vs cache.shape[0]=513，见 runbook）。8192 会分 4 块 → 会 bug。
#      实际 prompt_tokens 须 ≤ CHUNKED_PREFILL_SIZE；本脚本按 ~3.8 char/token 估，故给 40960 留余量。
#      上限：--max-prefill-tokens 65535 / --context-length 65536；chunk 须为 page-size=128 倍数，勿 -1。
#   depool/长上下文拉起：
#     export KT_MXFP4_CKPT=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash
#     export KT_MXFP4_OP_DIR=/workspace/code/kt-G-mxfp4kernel/tools/ascendc_mxfp4
#     KT_PREFILL_STREAM=1 KT_MXFP4_DEPOOL=1 KT_MXFP4_NZ_CHUNK=32 KT_DYNAMIC_RESIDENT=1 \
#     MEM_FRACTION=0.72 NPU_DEVICE_ID=<空卡> PORT=8200 CHUNKED_PREFILL_SIZE=40960 \
#       bash tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/serve.log
#   （普通 MXFP4 GGUF 也可，把 depool env 去掉、显式 KT_GGUF_TEMPLATE=..._mxfp4.gguf）
#
# 用法：
#   PORT=8200 bash tools/p27_longseq_32k_test.sh
# 可调（env）：
#   PORT             默认 8200
#   HOST             默认 127.0.0.1
#   TARGET_TOKENS    目标 prompt token 数，默认 32000（按 ~3.8 char/token 估算字符数，实际以服务器返回为准）
#   MAX_NEW          输出 token 数，默认 1000（ignore_eos 强制跑满，保证 ~1000）
#   SKIP_PREFILL_PROBE=1  跳过 n=1 prefill 探针（省一次 32k prefill，但拿不到 prefill/decode 拆分）
set -euo pipefail

PORT="${PORT:-8200}"
HOST="${HOST:-127.0.0.1}"
TARGET_TOKENS="${TARGET_TOKENS:-32000}"
MAX_NEW="${MAX_NEW:-1000}"
SKIP_PREFILL_PROBE="${SKIP_PREFILL_PROBE:-0}"
PY="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
PROMPT_FILE=/tmp/longseq_32k_prompt.txt
MA=/tmp/longseq_mA.json
MC=/tmp/longseq_mC.json

# ---------- 0. 服务健康 ----------
if ! curl -sf -m5 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "[longseq][ERROR] 服务没起或不健康：http://${HOST}:${PORT}/health" >&2
  echo "[longseq][ERROR] 先按脚本头部的拉起命令起服务（PORT=${PORT}）。" >&2
  exit 1
fi

# ---------- 1. 构 ~TARGET_TOKENS 的长 prompt（去重的事实句 + 唯一编号，避免被 prefix-cache 折叠）----------
"$PY" - "$TARGET_TOKENS" "$PROMPT_FILE" <<'PYEOF'
import sys
target = int(sys.argv[1]); out = sys.argv[2]
facts = [
    "The Roman aqueducts carried water across vast distances using only gravity",
    "Beethoven composed his ninth symphony while almost completely deaf",
    "The mitochondria is often called the powerhouse of the cell",
    "Mount Everest grows a few millimeters taller each year due to plate tectonics",
    "The printing press was invented by Johannes Gutenberg around the year 1440",
    "Honeybees communicate the location of flowers through an intricate waggle dance",
    "The Great Barrier Reef is the largest living structure on the entire Earth",
    "Light from the sun takes about eight minutes and twenty seconds to reach our planet",
    "The Rosetta Stone was the key that helped scholars decipher Egyptian hieroglyphs",
    "Antarctica is simultaneously the driest, the windiest, and the coldest continent",
    "Octopuses have three hearts and blue copper-based blood instead of red iron-based blood",
    "The Eiffel Tower can grow more than fifteen centimeters taller during a hot summer",
    "A single bolt of lightning is roughly five times hotter than the surface of the sun",
    "The human brain contains approximately eighty-six billion individual neurons",
    "Venus rotates so slowly that one of its days is longer than one of its years",
]
# ~3.8 字符/token（英文）估算；多生成一点，实际 token 数以服务器返回为准
chars_needed = int(target * 3.8)
buf, n, clen = [], 0, 0
while clen < chars_needed:
    s = f"{facts[n % len(facts)]} (fact {n})."
    buf.append(s); clen += len(s) + 1; n += 1
body = " ".join(buf)
prompt = (
    body
    + "\n\nQuestion: The passage above is very long and contains many numbered facts. "
      "Cite three distinct facts by their fact numbers, then write a detailed multi-paragraph "
      "summary of the different topics covered.\nAnswer:"
)
open(out, "w").write(prompt)
print(f"[longseq] built prompt: ~{clen} chars, {n} sentences (target {target} tokens)", file=sys.stderr)
PYEOF

PR="$("$PY" -c "import json;print(json.dumps(open('${PROMPT_FILE}').read()))")"

# ---------- 2. prefill 探针（n=1）：拿 32k 的 TTFT ----------
if [[ "${SKIP_PREFILL_PROBE}" != "1" ]]; then
  echo "[longseq] prefill 探针 (max_new_tokens=1) ..."
  curl -s -m 900 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' \
    -d "{\"text\": ${PR}, \"sampling_params\": {\"max_new_tokens\": 1, \"temperature\": 0}}" \
    -o "${MA}" -w "  http %{http_code}  wall %{time_total}s\n"
fi

# ---------- 3. 全量生成（n=MAX_NEW，ignore_eos 跑满）----------
echo "[longseq] 全量生成 (max_new_tokens=${MAX_NEW}, ignore_eos) ..."
curl -s -m 1800 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' \
  -d "{\"text\": ${PR}, \"sampling_params\": {\"max_new_tokens\": ${MAX_NEW}, \"temperature\": 0, \"ignore_eos\": true}}" \
  -o "${MC}" -w "  http %{http_code}  wall %{time_total}s\n"

# ---------- 4. 解析报告 ----------
SKIP_PREFILL_PROBE="${SKIP_PREFILL_PROBE}" "$PY" - "${MA}" "${MC}" <<'PYEOF'
import json, os, sys
ma, mc = sys.argv[1], sys.argv[2]
c = json.load(open(mc))
m = c.get("meta_info")
if not m:
    print("[longseq][ERROR] 全量请求无 meta_info，原始返回：", json.dumps(c)[:300]); sys.exit(1)
pt = m["prompt_tokens"]; nc = m["completion_tokens"]; e3 = m["e2e_latency"]
print("=" * 64)
print(f"  prompt_tokens     = {pt}")
print(f"  completion_tokens = {nc}")
if os.environ.get("SKIP_PREFILL_PROBE") != "1":
    a = json.load(open(ma)); am = a.get("meta_info", {})
    e1 = am.get("e2e_latency")
    if e1:
        dec = (e3 - e1) / max(nc - 1, 1)
        print(f"  prefill (TTFT@{pt}tok) = {e1:.2f}s   ({pt/e1:.0f} tok/s)")
        print(f"  decode (长上下文)      = {dec*1000:.0f} ms/token   ({1/dec:.2f} tok/s)")
print(f"  total e2e (n={nc})    = {e3:.2f}s   (含 prefill，{nc/e3:.2f} tok/s 端到端)")
print("=" * 64)
print("  answer head:", repr(c.get("text", "")[:300]))
PYEOF
