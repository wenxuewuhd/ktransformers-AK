#!/usr/bin/env bash
# 单卡 910B DeepSeek-V4-Flash 拉起服务（MXFP4 主线，no-arg=最优全量）。
# 用法:
#   NPU_DEVICE_ID=<空闲卡> bash script/dsv4_single_npu/1_serve.sh          # 前台跑，Ctrl-C 优雅停
#   NPU_DEVICE_ID=<空闲卡> DETACH=1 bash script/dsv4_single_npu/1_serve.sh # 后台跑（setsid 脱离）
#   (不传则默认卡 2；PORT 默认 8020)
#
# 这层壳只做手册要求、但容易漏的事，然后调用 tools/p27_launch_ds4flash_npu.sh：
#   1) 挑卡前打印 npu-smi，提醒你别撞到别人的卡（0.85×64≈55GB HBM 才够）；
#   2) 拉起前跑「三合一硬闸门」——防坑⑲(kt_kernel 被陈旧目录遮蔽→静默算错) / 坑⑳(.so 选错 ABI)；
#   3) 前台跑（默认，日志同时上屏+落盘可监控）或 setsid 后台（DETACH=1，规避坑⑭父进程回收）。
# 日志始终落盘到 logs/dsv4_single_npu/（持久，可事后分析）。
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
cd "$REPO"

PY="${PYTHON_BIN:-/usr/local/python3.11.14/bin/python3.11}"
NPU_DEVICE_ID="${NPU_DEVICE_ID:-2}"
PORT="${PORT:-8020}"
LOGDIR="${LOGDIR:-$REPO/logs/dsv4_single_npu}"
mkdir -p "$LOGDIR"
LOG="${LOG:-$LOGDIR/serve_$("$PY" -c 'import time;print(time.strftime("%Y%m%d_%H%M%S"))').log}"
DETACH="${DETACH:-0}"

echo "==== [1/4] 当前 NPU 占用（0.85×64≈55GB 空闲才放得下；别撞别人的卡）===="
npu-smi info | grep -E "65536" | nl -w2 -s': ' || true
echo "   -> 本次将用 NPU_DEVICE_ID=${NPU_DEVICE_ID}（改：NPU_DEVICE_ID=<卡> 重跑）"

echo "==== [2/4] 端口检查 ===="
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "!! 端口 ${PORT} 已被占用，换 PORT=<其他端口> 重跑" >&2; exit 1
fi
echo "   -> 端口 ${PORT} 空闲"

echo "==== [3/4] 硬闸门：kt_kernel 未被遮蔽 + e5f53ad 修复在位 + MXFP4=39 + .so ABI 匹配 ===="
# kt_kernel 必须是软链（被弄成真实目录副本会遮蔽 python/ 源码、悄悄回退竞态修复）
if [ ! -L kt-kernel/kt_kernel ]; then
  echo "   kt-kernel/kt_kernel 不是软链，重建 -> python"
  rm -rf kt-kernel/kt_kernel; ln -sfn python kt-kernel/kt_kernel
fi
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - <<'PYGATE'
import os, inspect, kt_kernel, kt_kernel.kt_kernel_ext as ext, kt_kernel.experts_base as m
from kt_kernel.utils.loader import GGMLQuantizationType as G
real = os.path.realpath(m.__file__)
assert real.endswith("/kt-kernel/python/experts_base.py"), f"kt_kernel 被陈旧副本遮蔽: {real}"
assert "Wait unconditionally (bypass already did)." in inspect.getsource(m), "缺 e5f53ad 异步竞态修复"
assert int(G.MXFP4) == 39, G.MXFP4
print("   -> PASS  ext=%s  MXFP4=39  e5f53ad 在位" % os.path.basename(os.path.realpath(ext.__file__)))
PYGATE

echo "==== [4/4] 拉起服务（日志落盘: ${LOG}）===="
rm -f "$LOG"

if [ "$DETACH" = "1" ]; then
  # 后台模式：setsid 脱离会话（规避坑⑭），日志只落盘
  setsid nohup env NPU_DEVICE_ID="$NPU_DEVICE_ID" PORT="$PORT" \
    bash tools/p27_launch_ds4flash_npu.sh > "$LOG" 2>&1 < /dev/null &
  echo "   已后台拉起（DETACH=1）。等待 /health（加载约 2–4 分钟）…"
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "==== 就绪 ✓  health 200 @ 127.0.0.1:${PORT}（约 $((i*10))s）===="
      echo "   服务日志: ${LOG}"
      echo "   下一步:  PORT=${PORT} bash script/dsv4_single_npu/2_gpqa_5x.sh"
      echo "   收服务:  kill -INT \$(ss -ltnp 2>/dev/null | grep :${PORT} | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)"
      exit 0
    fi
    if ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
      echo "!! 进程已退出，见日志尾部：" >&2; tail -30 "$LOG" >&2; exit 1
    fi
    sleep 10
  done
  echo "!! 20 分钟内未就绪，见 ${LOG}" >&2; tail -30 "$LOG" >&2; exit 1
fi

# 前台模式（默认）：日志同时上屏 + 落盘，你能实时监控；Ctrl-C 会优雅停服（sglang 捕获 SIGINT 释放 HBM）。
echo "   前台运行中——加载约 2–4 分钟，出现 'The server is fired up and ready to roll!' 即就绪。"
echo "   就绪后另开一个终端跑评测:  PORT=${PORT} bash script/dsv4_single_npu/2_gpqa_5x.sh"
echo "   停服: 在本终端按 Ctrl-C（优雅释放 HBM，别用 pkill -f sglang）。"
echo "   ---------------------------------------------------------------"
exec env NPU_DEVICE_ID="$NPU_DEVICE_ID" PORT="$PORT" \
  bash tools/p27_launch_ds4flash_npu.sh 2>&1 | tee "$LOG"
