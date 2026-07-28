#!/usr/bin/env bash
# GPQA-Diamond thinking-OFF × 5 轮，打已拉起的单卡服务，逐轮 + 汇总 mean/min/max/SD，并归档。
# 前提: 先 bash script/dsv4_single_npu/1_serve.sh 把服务拉起（/health=200）。
# 用法:
#   bash script/dsv4_single_npu/2_gpqa_5x.sh                 # 5 轮，端口 8020
#   REPEATS=5 PORT=8020 bash script/dsv4_single_npu/2_gpqa_5x.sh
#   LIMIT=5 bash script/dsv4_single_npu/2_gpqa_5x.sh         # 先冒烟：每轮只 5 题验证链路
#
# 说明:
#   * thinking OFF = chat_template_kwargs{thinking:false,high_effort:false}（p27_gpqa_repeat.sh 里已设）；
#   * 打 /v1/chat/completions（不是 /generate——chat 模板才注入 non-thinking 的 </think>）；
#   * --eval-batch-size 1（服务 --max-running-requests 1，并发会撞 NPU runtime 崩）；
#   * 数据集 gpqa_diamond 首次从 ModelScope 自动拉（需外网）；
#   * 单轮 198 题串行 ~70 分钟（空载）；5 轮 ~6 小时。别在跑评测时测吞吐（host DDR 争抢会假性掉速）。
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$REPO"

export PY="${PY:-/usr/local/python3.11.14/bin/python3.11}"
export PATH="$(dirname "$PY"):$PATH"
PORT="${PORT:-8020}"
HOST="${HOST:-127.0.0.1}"
REPEATS="${REPEATS:-5}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8}"
STAMP="$("$PY" -c 'import time;print(time.strftime("%Y%m%d_%H%M%S"))')"
OUT_PREFIX="${OUT_PREFIX:-eval_gpqa_R}"
ARCH="${ARCH:-/workspace/models/eval_archive/gpqa_off_5x_${STAMP}}"
# 日志同时落到仓库内持久目录（可被 review/分析）；控制台也实时上屏（前台）。
LOGDIR="${LOGDIR:-$REPO/logs/dsv4_single_npu}"
mkdir -p "$LOGDIR"
CONSOLE="${LOGDIR}/gpqa_5x_${STAMP}.log"

command -v evalscope >/dev/null 2>&1 || "$PY" -m pip install -q evalscope

# 服务健康检查
if [ "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; then
  echo "!! 服务未就绪 @ ${HOST}:${PORT}。先跑 script/dsv4_single_npu/1_serve.sh" >&2; exit 1
fi
mkdir -p "$ARCH"
echo "==== GPQA-off × ${REPEATS}（前台运行，实时上屏）===="
echo "   控制台日志（持久，可分析）: ${CONSOLE}"
echo "   产物归档: ${ARCH}"

# 冒烟档（LIMIT=N）：p27_gpqa_repeat.sh 的 evalscope 命令行写死、不认 --limit，故冒烟直接在这里跑
# 一发带 --limit 的 evalscope，只为验证链路（服务/数据集/chat 接口/打分）通不通，不做统计。
if [ -n "${LIMIT:-}" ]; then
  echo "   [冒烟] 单发 ${LIMIT} 题验证链路（不做 5 轮统计）"
  export no_proxy="${HOST},localhost" NO_PROXY="${HOST},localhost"
  GEN='{"temperature":1,"top_p":1,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}'
  evalscope eval --model "$MODEL_PATH" \
    --api-url "http://${HOST}:${PORT}/v1/chat/completions" --api-key EMPTY \
    --eval-type openai_api --datasets gpqa_diamond \
    --generation-config "$GEN" --eval-batch-size 1 --repeats 1 --limit "$LIMIT" \
    --work-dir "${ARCH}/smoke" 2>&1 | tee "$CONSOLE"
  cp -f "$CONSOLE" "${ARCH}/master_console.log"
  echo "==== 冒烟完成，链路 OK。去掉 LIMIT 再跑正式 ${REPEATS} 轮。 ===="
  exit 0
fi

# 正式档：直接复用仓库官方重复脚本；它内部已固化 GEN(thinking off)/chat 接口/batch=1/逐轮打分+汇总。
# 前台运行：tee 同时上屏 + 落到持久日志（你能盯，我能事后读 ${CONSOLE} 分析）。
# ★逐题产物写进本次带时间戳的归档目录（OUT_DIR=$ARCH），每次运行独立目录，永不覆盖历史。
REPEATS="$REPEATS" PORT="$PORT" HOST="$HOST" MODEL_PATH="$MODEL_PATH" \
OUT_DIR="$ARCH" OUT_PREFIX="$OUT_PREFIX" PY="$PY" \
EVALSCOPE="$(command -v evalscope)" \
  bash tools/p27_gpqa_repeat.sh 2>&1 | tee "$CONSOLE"
cp -f "$CONSOLE" "${ARCH}/master_console.log"

# 收尾归档
cp -f /tmp/kt_mxfp4_serve.log "${ARCH}/serve.log" 2>/dev/null || true
{
  echo "# GPQA-off × ${REPEATS}  @ ${STAMP}"
  echo "model=${MODEL_PATH}  port=${PORT}  batch=1  thinking=OFF  dataset=gpqa_diamond"
  echo
  for i in $(seq 1 "$REPEATS"); do
    R=$(find "${ARCH}/${OUT_PREFIX}${i}" -name gpqa_diamond.json -path '*reports*' 2>/dev/null | tail -1)
    S=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['score'])" "$R" 2>/dev/null || echo '?')
    echo "R${i} = ${S}"
  done
} > "${ARCH}/RESULTS.txt"
echo "==== 完成。全部产物在: ${ARCH}/（逐题在 ${OUT_PREFIX}{1..${REPEATS}}/，不覆盖历史）===="
cat "${ARCH}/RESULTS.txt"
