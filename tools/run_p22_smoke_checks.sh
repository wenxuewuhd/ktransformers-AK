#!/usr/bin/env bash
# P2.2 前置三项检查（kt_accel → kt_ep_wrapper 导入 → phase12，可选）。
# 必须通过本仓库 third_party/sglang/python 注入 SGLang；勿依赖环境里另一份
# （例如 /sgl-workspace/sglang 或 pip 全局安装的 sglang）。
# 用法：在仓库根目录执行
#   bash tools/run_p22_smoke_checks.sh
# 环境变量：
#   SGLANG_PYROOT  默认 third_party/sglang/python，用于 PYTHONPATH

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${SGLANG_PYROOT:-$ROOT/third_party/sglang/python}"
PY="${PYTHON:-/usr/local/python3.11.14/bin/python3}"

echo "== 1/3 kt_accel_stream_smoke =="
"$PY" "$ROOT/tools/kt_accel_stream_smoke.py"

echo "== 2/3 import kt_ep_wrapper =="
"$PY" -c "from sglang.srt.layers.moe import kt_ep_wrapper as k; print('ok', hasattr(k, 'resolve_kt_weight_path_for_layer'))"

echo "== 3/3 phase12 (若存在 GGUF) =="
GGUF="${P12_GGUF:-/workspace/models/cache/dsv4_layer3.gguf}"
LAYER="${P12_LAYER:-3}"
if [[ -f "$GGUF" ]]; then
  "$PY" "$ROOT/tools/phase12_llamafile_moe_smoke.py" --gguf "$GGUF" --layer-idx "$LAYER"
else
  echo "skip: no file $GGUF (set P12_GGUF / P12_LAYER to override)"
fi

echo "== all checks done =="
