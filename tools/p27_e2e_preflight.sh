#!/usr/bin/env bash
# P2.7 / 任务 3：整网 e2e 启动前预检（GGUF 层文件 + kt_kernel_ext 路径）。
#
# 用法：
#   bash tools/p27_e2e_preflight.sh
#   GGUF_DIR=/path/cache GGUF_SUFFIX=_q8_0 bash tools/p27_e2e_preflight.sh
#
# 退出码：0=通过，1=有缺失/路径不对。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=tools/p27_ensure_kt_kernel.sh
source "${SCRIPT_DIR}/p27_ensure_kt_kernel.sh"
p27_ensure_kt_kernel "$REPO"
GGUF_DIR="${GGUF_DIR:-/workspace/models/cache}"
GGUF_PREFIX="${GGUF_PREFIX:-dsv4_layer}"
# 批量 convert 默认 q8_0 后缀为空 → dsv4_layer3.gguf；若你用手动命名 dsv4_layer3_q8_0.gguf 则设 GGUF_SUFFIX=_q8_0
GGUF_SUFFIX="${GGUF_SUFFIX:-}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-42}"
MIN_GIB="${MIN_GIB:-6}"  # Q8_0 单层约 6.85 GiB；BF16 请设 MIN_GIB=12

PYBIN="${PYTHON_BIN:-${PYBIN:-/usr/local/python3.11.14/bin/python3.11}}"

echo "[preflight] REPO=$REPO"
echo "[preflight] GGUF: ${GGUF_DIR}/${GGUF_PREFIX}{L}${GGUF_SUFFIX}.gguf  layers ${LAYER_START}-${LAYER_END}"

missing=0
for L in $(seq "$LAYER_START" "$LAYER_END"); do
  f="${GGUF_DIR}/${GGUF_PREFIX}${L}${GGUF_SUFFIX}.gguf"
  if [[ ! -f "$f" ]]; then
    echo "[preflight] MISSING $f"
    missing=$((missing + 1))
    continue
  fi
  sz=$(stat -c%s "$f")
  min_bytes=$((MIN_GIB * 1024 * 1024 * 1024))
  if (( sz < min_bytes )); then
    echo "[preflight] TOO_SMALL $f ($(numfmt --to=iec "$sz" 2>/dev/null || echo "${sz}B") < ${MIN_GIB}GiB)"
    missing=$((missing + 1))
  fi
done
if (( missing > 0 )); then
  echo "[preflight] FAIL: ${missing} layer file(s) missing or too small."
  echo "[preflight] 批量转换: $PYBIN $REPO/tools/batch_convert_w8a8_layers_mp.py --input ... --output-dir $GGUF_DIR --layer-start $LAYER_START --layer-end $LAYER_END --quant q8_0 --skip-existing"
  exit 1
fi
echo "[preflight] OK: all $((LAYER_END - LAYER_START + 1)) GGUF files present (>= ${MIN_GIB} GiB each)."

echo "[preflight] kt_kernel_ext:"
"$PYBIN" -c "from kt_kernel import kt_kernel_ext; print('  ', kt_kernel_ext.__file__)"
so_path=$("$PYBIN" -c "from kt_kernel import kt_kernel_ext; print(kt_kernel_ext.__file__)")
if [[ "$so_path" == *"${REPO}/kt-kernel/python/"* || "$so_path" == *"${REPO}/kt-kernel/kt_kernel/"* ]]; then
  echo "[preflight] OK: kt_kernel_ext 在仓库 kt-kernel 包内"
else
  echo "[preflight] WARN: .so 不在 ${REPO}/kt-kernel/{python,kt_kernel}/ 下（见手册 §2.4）"
fi
build_so="/tmp/kt_kernel_build/kt_kernel_ext.cpython-"*"-linux-gnu.so"
if compgen -G "$build_so" >/dev/null 2>&1; then
  bso=$(ls -t $build_so 2>/dev/null | head -1)
  if [[ -f "$bso" && "$so_path" -ef "$bso" ]]; then
    echo "[preflight] OK: 已加载 /tmp/kt_kernel_build 同 inode（或已 cp 到 python/）"
  elif [[ -f "$bso" ]]; then
    echo "[preflight] HINT: build 目录有更新 .so，请执行:"
    echo "  cp -f $bso ${REPO}/kt-kernel/python/"
  fi
fi
echo "[preflight] PASS"
