#!/usr/bin/env bash
# 长序列测试用例：构多档长度的 prompt（默认 32k + 64k），各输出 ~MAX_NEW token，
# 报每档 prefill TTFT / 长上下文 decode(ms·token, tok/s) / 总吞吐，最后汇总对比表。
#
# 前提：服务已起，且 CHUNKED_PREFILL_SIZE >= 最长一档的实际 prompt_tokens（单 chunk，避 compressor 跨 chunk bug）。
#   ⚠️ context-length=65536 是硬上限：prompt + 输出 必须 ≤ 65536。64k 档 prompt 已接近上限，MAX_NEW 留小
#      （默认 1000：~63k + 1000 < 65536）。64k 单 chunk prefill 的 activation 峰值大、未验证，可能 OOM；
#      OOM 就降 MEM_FRACTION 或去掉 64k 档。
#   拉起（chunk 要覆盖 64k；65408 = 最大的 ≤ max-prefill-tokens(65535) 的 128 倍数）：
#     export KT_MXFP4_CKPT=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash
#     export KT_MXFP4_OP_DIR=/workspace/code/kt-G-mxfp4kernel/tools/ascendc_mxfp4
#     KT_PREFILL_STREAM=1 KT_MXFP4_DEPOOL=1 KT_MXFP4_NZ_CHUNK=32 KT_DYNAMIC_RESIDENT=1 \
#     MEM_FRACTION=0.72 NPU_DEVICE_ID=<空卡> PORT=8200 CHUNKED_PREFILL_SIZE=65408 \
#       bash tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/serve.log
#
# 用法：
#   PORT=8200 bash tools/p27_longseq_test.sh                    # 默认跑 32k + 64k 两档
#   TARGET_TOKENS_LIST="32000" PORT=8200 bash tools/p27_longseq_test.sh   # 只跑 32k
# env：
#   TARGET_TOKENS_LIST  默认 "32000 64000"（空格分隔多档）
#   MAX_NEW             默认 1000（ignore_eos 跑满）
#   CHARS_PER_TOK       默认 4.6（英文实测 ~4.69 char/token；偏小→实际 token 略小于 target，留 context 余量）
#   SKIP_PREFILL_PROBE=1  跳过每档 n=1 prefill 探针（省一次该长度的 prefill，但拿不到 prefill/decode 拆分）
#   PORT(8200) / HOST(127.0.0.1)
set -euo pipefail

PORT="${PORT:-8200}"
HOST="${HOST:-127.0.0.1}"
MAX_NEW="${MAX_NEW:-1000}"
CHARS_PER_TOK="${CHARS_PER_TOK:-4.6}"
TARGET_TOKENS_LIST="${TARGET_TOKENS_LIST:-32000 64000}"
SKIP_PREFILL_PROBE="${SKIP_PREFILL_PROBE:-0}"
PY="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
CONTEXT_LIMIT=65536

if ! curl -sf -m5 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "[longseq][ERROR] 服务没起或不健康：http://${HOST}:${PORT}/health（先按脚本头部命令起服务）" >&2
  exit 1
fi

SUMMARY_ARGS=()
for TGT in ${TARGET_TOKENS_LIST}; do
  if (( TGT + MAX_NEW > CONTEXT_LIMIT )); then
    echo "[longseq][WARN] 档 ${TGT}: prompt + 输出(${MAX_NEW}) 可能 > context-length ${CONTEXT_LIMIT}；输出会被截或报错。建议 target ≤ $((CONTEXT_LIMIT - MAX_NEW))。"
  fi
  PF="/tmp/longseq_prompt_${TGT}.txt"; MA="/tmp/longseq_mA_${TGT}.json"; MC="/tmp/longseq_mC_${TGT}.json"
  RQ1="/tmp/longseq_req1_${TGT}.json"; RQ2="/tmp/longseq_req2_${TGT}.json"

  # ---- 构 prompt + 两个请求体 JSON 直接写文件（curl -d @file，避开 Argument list too long）----
  "$PY" - "$TGT" "$PF" "$CHARS_PER_TOK" "$MAX_NEW" "$RQ1" "$RQ2" <<'PYEOF'
import sys, json
target = int(sys.argv[1]); out = sys.argv[2]; cpt = float(sys.argv[3])
maxn = int(sys.argv[4]); rq1 = sys.argv[5]; rq2 = sys.argv[6]
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
chars_needed = int(target * cpt)
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
json.dump({"text": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}, open(rq1, "w"))
json.dump({"text": prompt, "sampling_params": {"max_new_tokens": maxn, "temperature": 0, "ignore_eos": True}}, open(rq2, "w"))
print(f"[longseq] 档 {target}: built ~{clen} chars, {n} sentences", file=sys.stderr)
PYEOF

  if [[ "${SKIP_PREFILL_PROBE}" != "1" ]]; then
    echo "[longseq] 档 ${TGT}: prefill 探针 (max_new_tokens=1) ..."
    curl -s -m 3600 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' \
      -d @"${RQ1}" -o "${MA}" -w "  http %{http_code}  wall %{time_total}s\n"
  fi
  echo "[longseq] 档 ${TGT}: 全量生成 (max_new_tokens=${MAX_NEW}, ignore_eos) ..."
  curl -s -m 3600 -X POST "http://${HOST}:${PORT}/generate" -H 'Content-Type: application/json' \
    -d @"${RQ2}" -o "${MC}" -w "  http %{http_code}  wall %{time_total}s\n"

  SUMMARY_ARGS+=("${TGT}" "${MA}" "${MC}")
done

echo
SKIP_PREFILL_PROBE="${SKIP_PREFILL_PROBE}" "$PY" - "${SUMMARY_ARGS[@]}" <<'PYEOF'
import sys, os, json
args = sys.argv[1:]
skip = os.environ.get("SKIP_PREFILL_PROBE") == "1"
print("=" * 92)
print(f"{'目标':>7} {'prompt_tok':>10} {'compl':>6} {'prefill(s)':>10} {'pf tok/s':>9} "
      f"{'decode ms/tok':>13} {'dec tok/s':>9} {'e2e(s)':>8}")
print("-" * 92)
last_mc = None
for i in range(0, len(args), 3):
    tgt, ma, mc = args[i], args[i + 1], args[i + 2]
    c = json.load(open(mc)); m = c.get("meta_info", {}); last_mc = c
    pt = m.get("prompt_tokens"); nc = m.get("completion_tokens"); e3 = m.get("e2e_latency")
    pf = pftps = decms = dectps = None
    if not skip and os.path.exists(ma):
        am = json.load(open(ma)).get("meta_info", {}); e1 = am.get("e2e_latency")
        if e1 and pt and nc:
            pf = e1; pftps = pt / e1
            dec = (e3 - e1) / max(nc - 1, 1); decms = dec * 1000; dectps = 1 / dec
    g = lambda x, fmt: (fmt % x) if x is not None else "-"
    print(f"{tgt:>7} {pt:>10} {nc:>6} {g(pf,'%.1f'):>10} {g(pftps,'%.0f'):>9} "
          f"{g(decms,'%.0f'):>13} {g(dectps,'%.2f'):>9} {g(e3,'%.1f'):>8}")
print("=" * 92)
print("decode = 长上下文每 token 时间（短上下文基线 ~62ms/16tps；KV 越长 attention 增量越大）")
if last_mc is not None:
    print("末档 answer head:", repr(last_mc.get("text", "")[:220]))
PYEOF
