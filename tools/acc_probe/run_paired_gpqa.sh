#!/usr/bin/env bash
# 配对种子 temp=1 GPQA:每个 seed 一轮 —— 重启服务(--random-seed S)→ 跑 1 遍 → SIGTERM 收。
# 两台机器用同一组 SEEDS,bs=1 串行 + 每轮重启 → 同机可复现;若跨机 RNG 流一致则构成配对样本。
# 用法(910C):NPU_DEVICE_ID=1 PORT=8021 SEEDS="101 102 103 104 105" bash tools/acc_probe/run_paired_gpqa.sh
# 用法(910B 容器):NPU_DEVICE_ID=0 PORT=8020 SEEDS="101 102 103 104 105" bash tools/acc_probe/run_paired_gpqa.sh
set -uo pipefail
# REPO 可用 env 覆盖(910B 上脚本放独立目录、指向容器主干只读使用)
REPO="${REPO:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"
cd "$REPO"
SEEDS="${SEEDS:-101 102 103 104 105}"
PORT="${PORT:-8021}"
NPU_DEVICE_ID="${NPU_DEVICE_ID:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
for BASE in /mnt/workspace/models/eval_archive /workspace/models/eval_archive "$REPO/logs/dsv4_single_npu/eval_archive"; do
  [[ -d "$(dirname "$BASE")" && -w "$(dirname "$BASE")" ]] && { ROOT="$BASE/gpqa_paired_${STAMP}"; break; }
done
mkdir -p "$ROOT"
export LOGDIR="${LOGDIR:-$ROOT/logs}"; mkdir -p "$LOGDIR"   # 日志不落 REPO(910B 主干只读纪律)
echo "==== paired GPQA  seeds=[$SEEDS]  port=$PORT card=$NPU_DEVICE_ID  root=$ROOT ===="
for S in $SEEDS; do
  echo "---- seed $S: launch ----"
  NPU_DEVICE_ID="$NPU_DEVICE_ID" PORT="$PORT" DETACH=1 KT_PREFILL_STREAM=0 \
    EXTRA_FLAGS="--random-seed $S" bash script/dsv4_single_npu/1_serve.sh || true
  # DETACH 的 pgrep 有误报,自己等 health
  ok=0
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' "http://127.0.0.1:${PORT}/health")" = 200 ] && { ok=1; break; }
    sleep 10
  done
  [ "$ok" = 1 ] || { echo "!! seed $S: server not healthy, skip"; continue; }
  echo "---- seed $S: eval (REPEATS=1) ----"
  REPEATS=1 PORT="$PORT" ARCH="$ROOT/seed_$S" OUT_PREFIX="eval_seed${S}_R" \
    bash script/dsv4_single_npu/2_gpqa_5x.sh
  echo "---- seed $S: stop ----"
  P=$(pgrep -f "sglang.launch_server.*--port ${PORT}" | head -1)
  [ -n "$P" ] && kill -TERM "$P"
  for i in $(seq 1 30); do pgrep -f "sglang.launch_server.*--port ${PORT}" >/dev/null || break; sleep 5; done
  sleep 10
done
echo "==== all seeds done ===="
for S in $SEEDS; do
  R=$(find "$ROOT/seed_$S" -name gpqa_diamond.json -path '*reports*' 2>/dev/null | tail -1)
  [ -n "$R" ] && python3 -c "import json,sys;print('seed $S =', json.load(open(sys.argv[1]))['score'])" "$R" 2>/dev/null
done | tee "$ROOT/RESULTS.txt"
