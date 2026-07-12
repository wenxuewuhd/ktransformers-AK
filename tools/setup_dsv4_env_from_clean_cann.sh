#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash 单卡 910C(A3) —— 从干净 CANN 9.0.0 镜像搭建
#   自定义算子(custom_ops / customize / custom_transformer 三件套)
#   + sglang 相关依赖 + sgl_kernel_npu + kt-kernel
# =============================================================================
# 这是「环境/依赖 bring-up」脚本(不是模型代码 patch)。它把本次在新 910C(A3)
# 裸机上从 0 打通的确切步骤固化下来,每一步的坑都已内建修复。
#
# 用法:
#   1) 先按需改下面的「可配置变量」(路径/Python/CANN 版本)。
#   2) 逐阶段跑:  bash setup_dsv4_env_from_clean_cann.sh <phase>
#      phase ∈ {all, prereq, torch, triton, sglang_deps,
#               vendor_customize, custom_ops, vendor_transformer,
#               sgl_kernel_npu, kt_kernel, verify}
#      不带参数 = all(按顺序全跑)。
#
# 约定:配套文档见
#   doc/zh/dsv4_single_npu/从干净CANN9.0.0镜像复现_环境与自定义算子.md
# =============================================================================
set -euo pipefail

# ----------------------------- 可配置变量 ------------------------------------
: "${PYTHON_BIN:=/opt/buildtools/Python-3.11.4/bin/python3.11}"   # 你的 py3.11
: "${CANN_HOME:=$HOME/Ascend/cann-9.0.0}"                         # 干净镜像里的 CANN 9.0.0
: "${GITCODE:=/mnt/workspace/gitCode}"                            # 各仓库根目录
: "${REPO:=$GITCODE/ktransformers-AK}"                            # 本仓(含 kt-kernel + sglang 子模块)
: "${CC_BIN:=/usr/bin/gcc-13}"                                    # ARM bf16/i8mm + gnu++20 需 gcc>=13
: "${CXX_BIN:=/usr/bin/g++-13}"
: "${SOC:=ascend910_93}"                                          # A3(910C)= ascend910_93
: "${JOBS:=16}"

# 三方仓库(缺失则 clone)
: "${OPS_TF_REPO:=$GITCODE/cann/ops-transformer}"                 # NSA 算子(用 master 分支)
: "${OPS_TF_WORKTREE:=$GITCODE/cann/ops-transformer-master}"      # master 的干净 worktree
: "${RECIPES_REPO:=$GITCODE/cann/cann-recipes-infer}"             # customize vendor + custom_ops binding
: "${SGLKNPU_REPO:=$GITCODE/sgl-kernel-npu}"                      # sgl_kernel_npu / deep_ep / attentions

: "${OPS_TF_URL:=https://gitcode.com/cann/ops-transformer.git}"
: "${RECIPES_URL:=https://gitcode.com/cann/cann-recipes-infer.git}"
: "${SGLKNPU_URL:=https://github.com/sgl-project/sgl-kernel-npu.git}"
: "${SGLKNPU_TAG:=2026.6.2}"

# ★钉版本(可复现):三个三方仓都不跟随移动分支。ops-transformer 的 NSA 算子只在 master,
#   但 master 会持续漂;下面钉的是**本项目实测编译+跑通的 commit**。想升级请改这里并重测。
: "${OPS_TF_COMMIT:=dd9f31f34}"     # ops-transformer master 上实测可用的 commit
: "${RECIPES_COMMIT:=c5cc95e}"      # cann-recipes-infer(当时的 origin/master)

# pip 约束:锁死 torch 全家,任何依赖都不许动它们
TORCH_LOCK="$(dirname "$(readlink -f "$0")")/dsv4_torch_lock.txt"

PIP="$PYTHON_BIN -m pip"
export ASCEND_HOME_PATH="$CANN_HOME"

log(){ echo -e "\n\033[1;36m[setup][$(date +%H:%M:%S)] $*\033[0m"; }
die(){ echo -e "\033[1;31m[setup][FATAL] $*\033[0m" >&2; exit 1; }

# ----------------------------- 0. 前置检查 -----------------------------------
phase_prereq(){
  log "phase prereq: 工具链/权限检查"
  umask 0022                     # ★坑:umask 0002 → 产物 group-writable → CANN msopgen 安全校验 abort
  [ -x "$PYTHON_BIN" ] || die "PYTHON_BIN 不存在: $PYTHON_BIN"
  [ -d "$CANN_HOME" ]  || die "CANN_HOME 不存在: $CANN_HOME"
  # ★坑:默认 gcc-9 编不过 -march=...+bf16+i8mm 和 -std=gnu++20 → 需 gcc>=13
  if [ ! -x "$CC_BIN" ]; then
    echo "  [warn] $CC_BIN 不存在。Ubuntu/Debian: apt-get install -y gcc-13 g++-13"
    echo "         或改用 gcc-9 但 kt-kernel 关 ARM 扩展(脚本 kt_kernel 阶段已默认关)。"
  fi
  "$PYTHON_BIN" -c 'import sys;assert sys.version_info[:2]==(3,11),"need py3.11"' \
    || die "Python 必须是 3.11(torch/torch_npu/自定义算子 wheel 都是 cp311)"
  echo "  CANN=$CANN_HOME  PY=$PYTHON_BIN  CC=$CC_BIN  SOC=$SOC"
  source "$CANN_HOME/set_env.sh"
  echo "  ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
}

# ----------------------------- 1. torch 全家 ---------------------------------
phase_torch(){
  log "phase torch: 确认/安装 torch 2.8 + torch_npu 2.8.0.post4"
  # 干净 CANN 镜像通常自带 torch/torch_npu,但版本未必对。先校验,不对再装。
  if "$PYTHON_BIN" - <<'PY'
import importlib.metadata as m, sys
want={"torch":"2.8.0","torch_npu":"2.8.0.post4"}
ok=True
for k,v in want.items():
    try:
        got=m.version(k)
    except Exception:
        print(f"  missing {k}"); ok=False; continue
    if not got.startswith(v): print(f"  {k}={got} != {v}"); ok=False
sys.exit(0 if ok else 1)
PY
  then echo "  torch/torch_npu 版本 OK"; return; fi
  echo "  [!] torch 版本不符。按你镜像的源安装(下面是参考,可能需按内网源改):"
  cat <<EOF
    $PIP install torch==2.8.0 torchvision==0.23.0 torchaudio==2.11.0 \\
        --index-url https://download.pytorch.org/whl/cpu
    $PIP install torch_npu==2.8.0.post4    # Ascend 源 / 内网 pypi
EOF
  die "请先把 torch 全家装到 torch-lock.txt 里的版本,再重跑本阶段"
}

# ----------------------------- 2. triton-ascend -------------------------------
phase_triton(){
  log "phase triton: triton-ascend 3.2.1.dev(配 CANN 9.0.0)"
  # ★坑:triton-ascend==3.2.0 在 import 时就编 npu_utils.cpp,用到 CANN9.0.0 没有的
  #      RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE → import 直接失败。必须用 nightly 3.2.1.dev。
  $PIP install "triton-ascend==3.2.1.dev20260530" \
      --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly \
      --trusted-host mirrors.huaweicloud.com
  "$PYTHON_BIN" -c "import triton;print('  triton import OK')"
}

# ----------------------------- 3. sglang base deps ---------------------------
phase_sglang_deps(){
  log "phase sglang_deps: 装 sglang(dsv4 fork)的 base 依赖(不含 torch/自定义算子)"
  local reqs="$(dirname "$(readlink -f "$0")")/dsv4_sglang_base_reqs.txt"
  [ -f "$reqs" ] || die "缺 $reqs(随脚本一起提供)"
  $PIP install -r "$reqs" -c "$TORCH_LOCK"
  # kt-kernel 的纯 py 依赖
  $PIP install safetensors gguf -c "$TORCH_LOCK"
}

# ----------------------------- 4. customize vendor ---------------------------
# cann-recipes-infer 的融合算子:HcPre/HcPost/RmsNormDynamicQuant/
# InplacePartialRotaryMul/SwigluClipQuant/MoeGatingTopKHash/... → vendor "customize"
phase_vendor_customize(){
  log "phase vendor_customize: 编 + 装 customize vendor(cann-recipes-infer @ $RECIPES_COMMIT)"
  [ -d "$RECIPES_REPO" ] || git clone "$RECIPES_URL" "$RECIPES_REPO"
  # ★钉版本:不跟随 origin/master
  git -C "$RECIPES_REPO" fetch --quiet origin || true
  git -C "$RECIPES_REPO" checkout --quiet --detach "$RECIPES_COMMIT" \
    || die "cann-recipes-infer checkout $RECIPES_COMMIT 失败(工作树脏?先 git -C $RECIPES_REPO status)"
  source "$CANN_HOME/set_env.sh"; umask 0022
  cd "$RECIPES_REPO/ops/ascendc"
  chmod -R go-w .                                  # ★坑:防 msopgen 安全校验 abort
  bash build.sh -c "$SOC"                          # 默认编全部算子;A3=ascend910_93
  local run; run=$(ls output/CANN-custom_ops-*-linux.*.run | head -1)
  [ -n "$run" ] || die "customize .run 未生成,看 build 日志"
  chmod +x "$run"
  "./$run" --quiet --install-path="$CANN_HOME/opp"
  echo "  installed vendor: $CANN_HOME/opp/vendors/customize"
}

# ----------------------------- 5. custom_ops torch binding -------------------
# torch.ops.custom.* 的 python 绑定(把上面 vendor 的 aclnn 算子暴露给 torch)
phase_custom_ops(){
  log "phase custom_ops: 编 + 装 custom_ops torch binding(cann-recipes-infer)"
  [ -d "$RECIPES_REPO" ] || die "先跑 vendor_customize(会 clone cann-recipes-infer)"
  source "$CANN_HOME/set_env.sh"
  cd "$RECIPES_REPO/ops/ascendc/torch_ops_extension"
  USE_NINJA=1 bash build_and_install.sh            # build_ext + bdist_wheel + pip install -I
  "$PYTHON_BIN" -c "import custom_ops;print('  custom_ops import OK')"
}

# ----------------------------- 6. custom_transformer vendor ------------------
# NSA/DSA 算子:compressor / sparse_attn_sharedkv / quant_lightning_indexer(+metadata)
# ★这些在 ops-transformer 的 9.0.0 分支被删了,只在 master 的 experimental/attention/。
# ★vendor 命名怪癖:ops-transformer 会给 vendor 名自动追加 "_transformer",
#   所以传 --vendor_name=custom,最终得到 vendor "custom_transformer"。
phase_vendor_transformer(){
  log "phase vendor_transformer: 编 + 装 custom_transformer vendor(ops-transformer @ $OPS_TF_COMMIT)"
  [ -d "$OPS_TF_REPO" ] || git clone "$OPS_TF_URL" "$OPS_TF_REPO"
  # 用干净 worktree 编(master 工作树若脏会污染产物)。★钉版本:worktree 直接 detach 到实测 commit,
  #   不跟 master(NSA 算子只在 master,但 master 会持续漂,跟着走会编出没测过的算子)。
  if [ ! -d "$OPS_TF_WORKTREE" ]; then
    git -C "$OPS_TF_REPO" fetch origin master
    git -C "$OPS_TF_REPO" worktree add --detach "$OPS_TF_WORKTREE" "$OPS_TF_COMMIT" \
      || die "ops-transformer worktree @ $OPS_TF_COMMIT 创建失败"
  fi
  source "$CANN_HOME/set_env.sh"; umask 0022
  cd "$OPS_TF_WORKTREE"; chmod -R go-w .
  bash build.sh --pkg --experimental --soc="$SOC" --vendor_name=custom \
    --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata,compressor,quant_lightning_indexer,quant_lightning_indexer_metadata \
    --cann_3rd_lib_path="$OPS_TF_REPO/third_party" -j"$JOBS"
  local run="build/cann-ops-transformer-custom_linux-aarch64.run"
  [ -f "$run" ] || die "custom_transformer .run 未生成: $run"
  bash "$run" --quiet --install-path="$CANN_HOME/opp"
  echo "  installed vendor: $CANN_HOME/opp/vendors/custom_transformer"
}

# ----------------------------- 7. sgl_kernel_npu -----------------------------
# sgl_kernel_npu / deep_ep / attentions / torch_memory_saver(NPU attention/MoE 内核)
phase_sgl_kernel_npu(){
  log "phase sgl_kernel_npu: 源码编 sgl_kernel_npu 全家(tag $SGLKNPU_TAG)"
  if [ ! -d "$SGLKNPU_REPO" ]; then
    git clone "$SGLKNPU_URL" "$SGLKNPU_REPO"
    git -C "$SGLKNPU_REPO" checkout "$SGLKNPU_TAG"
    git -C "$SGLKNPU_REPO" submodule update --init --recursive
  fi
  source "$CANN_HOME/set_env.sh"; umask 0022
  cd "$SGLKNPU_REPO"; chmod -R go-w .
  # ★坑2:csrc/attentions/csrc/CMakeLists.txt 的 PTAExtensionOPS 缺 -ldl →
  #        undefined reference dlopen/dlsym。补 ${CMAKE_DL_LIBS}。
  local cm="csrc/attentions/csrc/CMakeLists.txt"
  if [ -f "$cm" ] && ! grep -q "CMAKE_DL_LIBS" "$cm"; then
    echo "  [patch] 给 $cm 的 target_link_libraries(PTAExtensionOPS ...) 补 \${CMAKE_DL_LIBS}"
    sed -i 's/\(target_link_libraries(PTAExtensionOPS[^\n]*\)/\1 ${CMAKE_DL_LIBS}/' "$cm" || true
    grep -q "CMAKE_DL_LIBS" "$cm" || echo "  [warn] 自动补 -ldl 失败,请手动加(见文档)"
  fi
  # ★坑3:deep_ep vendor 装成只读,二次 build 会在 rm uninstall.sh 处 Permission denied
  chmod -R u+w python/deep_ep/deep_ep/vendors 2>/dev/null || true
  rm -rf python/deep_ep/deep_ep/vendors/hwcomputing 2>/dev/null || true
  # ★坑4:别 rm csrc/attentions/build/(那是 TRACKED 源码目录,不是产物)
  bash build.sh                                    # 默认 SOC=Ascend910_9382=A3
  $PIP install output/{sgl_kernel_npu,deep_ep,attentions,torch_memory_saver}*.whl -c "$TORCH_LOCK"
}

# ----------------------------- 8. kt-kernel ----------------------------------
phase_kt_kernel(){
  log "phase kt_kernel: 编 kt-kernel(CPU MoE / MXFP4 NEON)"
  cd "$REPO/kt-kernel"
  rm -rf build/temp.linux-aarch64-cpython-311
  # ★坑:ARM SVE=ON 会让 MXFP4 CPU MoE 走 SVE 分支报 "llamafile not supported"(moe.hpp:73/77)
  #      → 必须 SVE/BF16/I8MM 全关(走验证过的 NEON armv8.2+fp16+dotprod 路径)。
  # ★坑:CMakeLists 原本强制覆盖成 /usr/bin/gcc,已改成优先认 CC/CXX(见本仓 commit)。
  CC="$CC_BIN" CXX="$CXX_BIN" CPUINFER_USE_ASCEND_NPU=1 \
    CPUINFER_ARM_SVE=OFF CPUINFER_ARM_BF16=OFF CPUINFER_ARM_I8MM=OFF \
    "$PYTHON_BIN" setup.py build_ext --inplace
  ls python/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so \
    && echo "  kt_kernel_ext.so 生成 OK"
}

# ----------------------------- 9. 验证 import gate ---------------------------
phase_verify(){
  log "phase verify: import gate"
  source "$CANN_HOME/set_env.sh"
  source "$CANN_HOME/opp/vendors/custom_transformer/bin/set_env.bash"
  source "$CANN_HOME/opp/vendors/customize/bin/set_env.bash"
  PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel/python:${PYTHONPATH:-}" \
  "$PYTHON_BIN" - <<'PY'
import torch, torch_npu; print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
import triton; print("triton OK")
import kt_kernel; print("kt_kernel OK")
import sgl_kernel_npu; print("sgl_kernel_npu OK")
import custom_ops  # noqa
# 关键:三件套 vendor + binding 都在 → torch.ops.custom.* 可解析
for op in ("compressor","npu_sparse_attn_sharedkv","npu_quant_lightning_indexer","npu_moe_gating_top_k"):
    assert hasattr(torch.ops.custom, op), f"missing torch.ops.custom.{op}"
print("torch.ops.custom.* 全部就位 OK")
PY
  echo "  == 环境 bring-up 完成。拉服务见 tools/p27_launch_ds4flash_npu.sh =="
}

# ----------------------------- 调度 ------------------------------------------
phase="${1:-all}"
case "$phase" in
  prereq) phase_prereq;;
  torch) phase_prereq; phase_torch;;
  triton) phase_prereq; phase_triton;;
  sglang_deps) phase_prereq; phase_sglang_deps;;
  vendor_customize) phase_prereq; phase_vendor_customize;;
  custom_ops) phase_prereq; phase_custom_ops;;
  vendor_transformer) phase_prereq; phase_vendor_transformer;;
  sgl_kernel_npu) phase_prereq; phase_sgl_kernel_npu;;
  kt_kernel) phase_prereq; phase_kt_kernel;;
  verify) phase_verify;;
  all)
    phase_prereq
    phase_torch
    phase_triton
    phase_sglang_deps
    phase_vendor_customize
    phase_custom_ops
    phase_vendor_transformer
    phase_sgl_kernel_npu
    phase_kt_kernel
    phase_verify
    ;;
  *) die "未知 phase: $phase(见脚本头 usage)";;
esac
log "phase '$phase' 完成 ✅"
