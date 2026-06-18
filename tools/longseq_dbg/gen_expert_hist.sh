#!/usr/bin/env bash
# 复现 prefill 专家激活直方图(handoff §3.5.C-bis / expert_distribution_analysis.md)。
#
# 产物:$HIST_PATH(默认 tools/longseq_dbg/expert_hist.pt),torch.save 的 dict,schema 见
#       expert_distribution_analysis.md §2。解析/分析用 analyze_expert_hist.py。
#
# 用法(卡号/端口/长度可覆盖):
#   NPU=5 PORT=8013 bash tools/longseq_dbg/gen_expert_hist.sh
#
# 注意:直方图当前**跨请求累加**(全局)。测代表性分布前别混入退化数据(全同 filler token
# 会让路由退化→只激活 ~44 专家)。本脚本只发**真实多样文本**(repo 文档+源码),冷专家=0。
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NPU="${NPU:-5}"; PORT="${PORT:-8013}"
HIST_PATH="${HIST_PATH:-$REPO/tools/longseq_dbg/expert_hist.pt}"
LOG="${LOG:-$REPO/tools/longseq_dbg/hist_server.log}"
REPEAT="${REPEAT:-4}"   # 发几次真实文本(累加;越多越代表性)

echo "[hist] 拉起 32-expert 生产配置(NPU=$NPU PORT=$PORT),直方图 -> $HIST_PATH"
KT_PREFILL_EXPERT_HIST=1 KT_PREFILL_EXPERT_HIST_PATH="$HIST_PATH" \
  CHUNKED_PREFILL_SIZE=32768 PORT="$PORT" ASCEND_RT_VISIBLE_DEVICES="$NPU" \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  nohup "$REPO/tools/p27_launch_ds4flash_npu.sh" > "$LOG" 2>&1 &
echo "[hist] launcher pid=$! ; 等 ready(~2.5min)..."
until grep -qE 'ready to roll|Scheduler hit an exception|107030' "$LOG"; do sleep 5; done
grep -qE 'ready to roll' "$LOG" || { echo "[hist] 启动失败,见 $LOG"; exit 1; }

# 真实多样文本(repo 文档+sglang 模型源码),~60k 字符 → ~18k token/请求
python3 - <<'PY'
import glob, json
t=[]
for p in glob.glob('doc/zh/**/*.md', recursive=True)+glob.glob('third_party/sglang/python/sglang/srt/models/*.py'):
    try: t.append(open(p,encoding='utf-8',errors='ignore').read())
    except: pass
blob=("\n\n".join(t)*2)[:60000]
json.dump({"text":blob,"sampling_params":{"max_new_tokens":1,"temperature":0}}, open('/tmp/realtext_body.json','w'))
print("[hist] prompt chars", len(blob))
PY
for i in $(seq 1 "$REPEAT"); do
  echo "[hist] 真实文本请求 $i/$REPEAT ..."
  curl -sS -o /dev/null -w "[hist] req$i HTTP=%{http_code} %{time_total}s\n" --max-time 900 \
    -X POST "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
    --data-binary @/tmp/realtext_body.json || true
done
echo "[hist] dump 摘要:"; grep 'KT_PREFILL_EXPERT_HIST' "$LOG" | tail -1
echo "[hist] 分析:python3 tools/longseq_dbg/analyze_expert_hist.py $HIST_PATH 32"
