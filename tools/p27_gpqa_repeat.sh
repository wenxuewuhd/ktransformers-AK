#!/usr/bin/env bash
# GPQA-diamond 精度回归:同配置重复跑 N 遍,逐轮打分 + 累计均值,末尾给 mean/min/max/sd。
#
# 为什么要重复跑:temp=1 下单跑的采样噪声很大(198 题的二项 SE ≈ ±3.3pp),**单次结果不能下结论**。
# 本仓实测:单卡 910C 的 GPQA-off 真实中心 ~68-69%,10× 重复 mean 68.99% / SD 1.3pp。
# 想追一个 <5pp 的"回归"必须先重复 10 次看均值,否则你追的只是噪声。详见 accuracy_report.md。
#
# 前提:服务已起(脚本会等 /health=200)。
#
# 用法:
#   bash tools/p27_gpqa_repeat.sh                       # 默认 10 轮,端口 8020
#   REPEATS=3 PORT=8020 bash tools/p27_gpqa_repeat.sh   # 只跑 3 轮
#   OUT_PREFIX=eval_gpqa_myrun bash tools/p27_gpqa_repeat.sh
#
# env(全部可覆盖):
#   REPEATS(10) / PORT(8020) / HOST(127.0.0.1)
#   MODEL_PATH  被测模型路径(必须与服务端一致)
#   OUT_DIR     输出根目录(默认仓库根)
#   OUT_PREFIX  每轮输出目录前缀(默认 eval_gpqa_R)
#   PY / EVALSCOPE  解释器与 evalscope 可执行(默认从 PATH 找)
set -u

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
: "${REPEATS:=10}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8020}"
: "${MODEL_PATH:=/mnt/workspace/models/DeepSeek-V4-Flash-W8A8}"
: "${OUT_DIR:=$REPO_ROOT}"
: "${OUT_PREFIX:=eval_gpqa_R}"
: "${PY:=$(command -v python3 || echo python3)}"
: "${EVALSCOPE:=$(command -v evalscope || echo evalscope)}"

API="http://${HOST}:${PORT}/v1/chat/completions"
# temp=1 是 GPQA-off 的标准设置;thinking/high_effort=false = 非思考档
GEN='{"temperature":1,"top_p":1,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}'
export no_proxy="${HOST},localhost" NO_PROXY="${HOST},localhost"

command -v "$EVALSCOPE" >/dev/null 2>&1 || { echo "[gpqa] 找不到 evalscope,请 pip install evalscope 或设 EVALSCOPE=" >&2; exit 1; }

echo "[gpqa] 等 ${API%/v1*}/health 就绪(最多 20 分钟)…"
for _ in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" = "200" ] && { echo "[gpqa] 服务就绪"; break; }
  sleep 5
done

declare -a SCORES
for i in $(seq 1 "$REPEATS"); do
  WD="$OUT_DIR/${OUT_PREFIX}${i}"
  rm -rf "$WD"
  echo "===================== R$i START $(date +%H:%M:%S) ====================="
  "$EVALSCOPE" eval --model "$MODEL_PATH" \
    --api-url "$API" --api-key EMPTY \
    --eval-type openai_api --datasets gpqa_diamond \
    --generation-config "$GEN" --eval-batch-size 1 --repeats 1 \
    --work-dir "$WD" 2>&1 | tee "$WD.log"

  RPT="$(find "$WD" -name gpqa_diamond.json -path '*reports*' 2>/dev/null | tail -1)"
  S="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('score','?'))" "$RPT" 2>/dev/null || echo '?')"
  SCORES[i]="$S"
  echo "===================== R$i DONE  $(date +%H:%M:%S)  score=$S ====================="
  "$PY" - "${SCORES[@]}" <<'PYEOF'
import sys
xs=[float(x) for x in sys.argv[1:] if x not in ('','?')]
if xs: print("  >> 累计(n=%d): mean=%.4f min=%.4f max=%.4f" % (len(xs), sum(xs)/len(xs), min(xs), max(xs)))
PYEOF
done

echo "======================== 汇总 ========================"
for i in $(seq 1 "$REPEATS"); do echo "R$i = ${SCORES[i]:-?}"; done
"$PY" - "${SCORES[@]}" <<'PYEOF'
import sys, statistics as st
xs=[float(x) for x in sys.argv[1:] if x not in ('','?')]
if xs:
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    print("n=%d  mean=%.4f  min=%.4f  max=%.4f  sd=%.4f" % (len(xs), sum(xs)/len(xs), min(xs), max(xs), sd))
    print("(参考:本仓单卡 910C 实测中心 ~68-69%%,10× mean 68.99%% / SD 1.3pp;"
          "单跑 SE ±3.3pp,别拿单次结果下结论)")
PYEOF
