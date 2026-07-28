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

# 解析 python3.11:PY / PYTHON_BIN 覆盖 > PATH > 常见安装位置
_resolve_py() {
  for v in "${PY:-}" "${PYTHON_BIN:-}"; do
    [ -n "$v" ] && [ -x "$v" ] && { echo "$v"; return; }
  done
  local p
  for c in python3.11 /opt/buildtools/Python-3.11.4/bin/python3.11 \
           /usr/local/python3.11.14/bin/python3.11 /usr/bin/python3.11; do
    p="$(command -v "$c" 2>/dev/null || true)"
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return; }
    [ -x "$c" ] && { echo "$c"; return; }
  done
}
export PY="$(_resolve_py)"
[ -z "$PY" ] && { echo "找不到 python3.11，请设 PYTHON_BIN=/path/to/python3.11"; exit 1; }
export PATH="$(dirname "$PY"):$PATH"
PORT="${PORT:-8020}"
HOST="${HOST:-127.0.0.1}"
REPEATS="${REPEATS:-5}"
# MODEL_PATH 默认:自动探测(本盒 /mnt/workspace、旧镜像 /workspace),显式 env 仍优先。
if [[ -z "${MODEL_PATH:-}" ]]; then
  for _c in /mnt/workspace/models/DeepSeek-V4-Flash-W8A8 \
            /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8; do
    [[ -d "$_c" ]] && { MODEL_PATH="$_c"; break; }
  done
  MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8}"
fi
STAMP="$("$PY" -c 'import time;print(time.strftime("%Y%m%d_%H%M%S"))')"
OUT_PREFIX="${OUT_PREFIX:-eval_gpqa_R}"
# 归档输出目录:选一个可写的 base(本盒 /mnt/workspace、旧镜像 /workspace,都不可写则落仓库 logs/)。
if [[ -z "${ARCH:-}" ]]; then
  for _b in /mnt/workspace/models/eval_archive /workspace/models/eval_archive \
            "$REPO/logs/dsv4_single_npu/eval_archive"; do
    _d="$(dirname "$_b")"
    [[ -d "$_d" && -w "$_d" ]] && { ARCH="$_b/gpqa_off_5x_${STAMP}"; break; }
  done
  ARCH="${ARCH:-$REPO/logs/dsv4_single_npu/eval_archive/gpqa_off_5x_${STAMP}}"
fi
# 日志同时落到仓库内持久目录（可被 review/分析）；控制台也实时上屏（前台）。
LOGDIR="${LOGDIR:-$REPO/logs/dsv4_single_npu}"
mkdir -p "$LOGDIR"
CONSOLE="${LOGDIR}/gpqa_5x_${STAMP}.log"

# ★ evalscope 版本必须锁死 —— 这不是洁癖，是踩过的坑（2026-07-28）：
#   本行原为 `command -v evalscope || pip install -q evalscope`（不锁版本）。GPQA 的
#   198 题里有 15 题选项含方括号（IUPAC 命名如 spiro[4.5]decan、benzo[1,2-c:...]，
#   以及量子态记号）。evalscope **1.8.1** 的 benchmarks/gpqa/gpqa_adapter.py 在
#   _process_input() 的 preprocess() 里做 `re.sub(r'\[.*?\]', '', text)`，把这些方括号
#   内容整段删掉 —— spiro[4.5]decan-6-ol 变成 spirodecan-6-ol、[1,1'-biphenyl]-4-ol
#   变成 -4-ol，选项与题干对不上。1.9.x 已移除该清洗。
#   实测（910B，同一模型同一服务）：1.8.1 六轮 67.85%±1.83 vs 1.9.1 九轮 70.88%±0.87，
#   差 3.03pp，置换检验 p=0.0006；差异全部来自那 15 题（41.3% vs 61.1%），
#   其余 183 题只差 1.6pp（噪声）。910C 独立排查得到同一结论（同样 15 题、同样 idx24）。
#   ⇒ 不锁版本会让「跑分」随 PyPI 上的最新版漂移，把 harness 差异误读成模型/硬件回归。
EVALSCOPE_VERSION="${EVALSCOPE_VERSION:-1.9.1}"
# 预装了 <1.9.0 的旧版也强制升级到锁定版，避免 1.8.x 静默复现
# （根因另见 REPORT_910b_vs_910c_accuracy.md）。
_ev="$("$PY" -m pip show evalscope 2>/dev/null | awk '/^Version:/{print $2}')"
if [ -z "$_ev" ] || [ "$(printf '1.9.0\n%s\n' "$_ev" | sort -V | head -1)" != "1.9.0" ]; then
  echo "   [evalscope] ${_ev:-未装} < 1.9.0，安装 evalscope==${EVALSCOPE_VERSION}（1.8.1 会污染 GPQA 精度）"
  "$PY" -m pip install -q "evalscope==${EVALSCOPE_VERSION}"
fi
# 无论用的是哪一份（PATH 上的 / venv 里的 / 调用方用 EVALSCOPE= 指定的），都必须落盘，
# 否则事后无法判断这一跑到底用了哪个 harness（这正是本次排查最耗时的一环）。
_ES_BIN="${EVALSCOPE:-$(command -v evalscope)}"
_ES_VER="$("$_ES_BIN" --version 2>/dev/null | tr -d '\n' || echo unknown)"
echo "==== [evalscope] bin=${_ES_BIN}"
echo "==== [evalscope] version=${_ES_VER}  (期望 ${EVALSCOPE_VERSION}；>=1.9 才修好 GPQA 方括号清洗)"
case "$_ES_VER" in
  *1.8.*) echo "!! [evalscope] 警告：1.8.x 会破坏 15/198 道 GPQA 题的选项，跑分系统性低约 3pp。" >&2
          echo "!! 除非你明确要复现历史口径，否则请升级：$PY -m pip install 'evalscope==${EVALSCOPE_VERSION}'" >&2 ;;
esac

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
