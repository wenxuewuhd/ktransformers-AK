# 从干净 CANN 9.0.0 镜像复现：DeepSeek-V4-Flash 单卡 910C(A3) 环境 + 自定义算子

> 目标：在一台**只装了 CANN 9.0.0 的干净镜像**上,从 0 把 DSV4-Flash 单卡 W8A8 服务需要的
> **自定义算子(三件套)+ sglang 依赖 + sgl_kernel_npu + kt-kernel** 全部装好,跑通 import gate。
> 本文 = 本次在新 910C(A3) 裸机实测跑通的**确切步骤 + 每一步的坑**,配套脚本一键可跑:
>
> - 脚本:[`tools/setup_dsv4_env_from_clean_cann.sh`](../../../tools/setup_dsv4_env_from_clean_cann.sh)
> - 依赖清单:[`tools/dsv4_sglang_base_reqs.txt`](../../../tools/dsv4_sglang_base_reqs.txt) · [`tools/dsv4_torch_lock.txt`](../../../tools/dsv4_torch_lock.txt)
> - 装完拉服务:[`tools/p27_launch_ds4flash_npu.sh`](../../../tools/p27_launch_ds4flash_npu.sh)(见文末)
> - 相关:memory `dsv4-910c-launch-blockers` / `sgl-kernel-npu-source-build-pitfalls` / `kt-kernel-gcc13-new-host-build`;数值正确性见 [`A3_W8A8_数值对齐调查.md`](A3_W8A8_数值对齐调查.md)。

---

## 0. 前提与硬件

| 项 | 值 |
|---|---|
| 硬件 | 单机 2× Ascend **910C(A3)**(phy-id 2/3 = 容器内逻辑 0/1) |
| CANN | **9.0.0**,装在 `$HOME/Ascend/cann-9.0.0`(本机 `/home/developer/Ascend/cann-9.0.0`) |
| Python | **3.11**(本机 `/opt/buildtools/Python-3.11.4/bin/python3.11`)——torch/torch_npu/自定义算子 wheel 全是 cp311 |
| 编译器 | **gcc-13 / g++-13**(默认 gcc-9 编不过 `+bf16/+i8mm` 和 `-std=gnu++20`) |
| torch | **2.8.0(+cpu)** + **torch_npu 2.8.0.post4**(别跟官方 Ascend 文档升到 torch 2.10——本 fork + kt_kernel_ext.so 是 torch-2.8 ABI) |

**四个仓库**(脚本缺失会自动 clone):

| 用途 | 仓库 / 分支 | 说明 |
|---|---|---|
| 主仓(kt-kernel + sglang 子模块) | `ktransformers-AK`;sglang 子模块 = `wenxuewuhd/sglang-dsv4 @ dsv4_release`(`4ea20e5d`) | NPU 主干 + CPU MoE 封装 |
| NSA/DSA 算子 → **custom_transformer** vendor | `gitcode.com/cann/ops-transformer` **master** | compressor / sparse_attn_sharedkv / quant_lightning_indexer(9.0.0 分支已删,只在 master) |
| 融合算子 → **customize** vendor + **custom_ops** binding | `gitcode.com/cann/cann-recipes-infer` | HcPre/HcPost/RmsNormDynamicQuant/… + `torch.ops.custom.*` 绑定 |
| NPU 内核 | `github.com/sgl-project/sgl-kernel-npu` **tag 2026.6.2** | sgl_kernel_npu / deep_ep / attentions / torch_memory_saver |

> **一句话架构**：三件套 = ① `customize` vendor(aclnn 融合算子.so)+ ② `custom_transformer` vendor(NSA 算子.so)+ ③ `custom_ops` python 绑定(把①②的 aclnn 暴露成 `torch.ops.custom.*`)。缺任一个,forward 就会在 `aclnnXxx not in libopapi.so` 或 `torch.ops.custom.xxx 不存在` 处崩。

---

## 1. 一键跑法

```bash
cd /mnt/workspace/gitCode/ktransformers-AK
# 按你的机器改路径(下面是本机默认值,已内置,通常无需改)
export PYTHON_BIN=/opt/buildtools/Python-3.11.4/bin/python3.11
export CANN_HOME=$HOME/Ascend/cann-9.0.0
export GITCODE=/mnt/workspace/gitCode
export CC_BIN=/usr/bin/gcc-13 CXX_BIN=/usr/bin/g++-13

bash tools/setup_dsv4_env_from_clean_cann.sh all      # 全量;或逐阶段跑(下表)
```

**分阶段**(排障时逐个跑):

| phase | 做什么 | 关键坑 |
|---|---|---|
| `prereq` | 工具链/权限/版本检查,`umask 0022` | umask 0002 → msopgen 安全 abort |
| `torch` | 校验 torch 2.8 / torch_npu 2.8.0.post4,不对则提示装 | 别升 torch 2.10 |
| `triton` | `triton-ascend==3.2.1.dev20260530`(华为 nightly) | 3.2.0 import 即崩(CANN9.0.0 缺符号) |
| `sglang_deps` | `pip install -r dsv4_sglang_base_reqs.txt -c dsv4_torch_lock.txt` + safetensors/gguf | torch-lock 锁死 torch |
| `vendor_customize` | 编+装 **customize** vendor(`bash build.sh -c ascend910_93` → `.run`) | `chmod -R go-w` |
| `custom_ops` | 编+装 **custom_ops** torch 绑定(`build_and_install.sh`) | — |
| `vendor_transformer` | 编+装 **custom_transformer** vendor(ops-transformer master) | vendor 名要传 `--vendor_name=custom`;9.0.0 分支没这些算子 |
| `sgl_kernel_npu` | 源码编 sgl_kernel_npu 全家(tag 2026.6.2) | 缺 `-ldl`;deep_ep 只读重装;别删 tracked 的 `attentions/build/` |
| `kt_kernel` | 编 kt-kernel(gcc-13 + ARM 扩展全关) | SVE=ON → MXFP4 MoE `llamafile not supported` |
| `verify` | import gate + `torch.ops.custom.*` 就位检查 | — |

---

## 2. 各步骤细节与坑(逐条对应脚本)

### 2.1 前置(prereq)
- **`umask 0022`**：CANN 的 `msopgen` 会拒绝 group/other 可写的中间文件(`should not be written by user group or others, which will cause security risks`)→ 整个 build abort。脚本每个编译阶段都先 `umask 0022` + `chmod -R go-w .`。
- **gcc-13**：本 CPU `/proc/cpuinfo` 有 `bf16/i8mm/sve`,kt-kernel 的 setup.py 会自动开 `-march=armv8.2-a+...+bf16+i8mm`,gcc-9 不认;`CMAKE_CXX_STANDARD 20 → -std=gnu++20` gcc-9 也不认(只认 `gnu++2a`)。Ubuntu：`apt-get install -y gcc-13 g++-13`。

### 2.2 torch / triton
- torch 全家用 `dsv4_torch_lock.txt` 锁死。干净 CANN 镜像常自带 torch,但版本未必对——脚本先校验,不符会打印参考安装命令让你按内网源装,**不会**静默升级。
- **triton-ascend 必须 3.2.1.dev**：`3.2.0` 在 `import triton` 时就编 `npu_utils.cpp`,用到 CANN 9.0.0 头文件里没有的 `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` → import 直接失败。nightly 源：`https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly`。

### 2.3 customize vendor(cann-recipes-infer)
```bash
cd cann-recipes-infer/ops/ascendc
source $CANN_HOME/set_env.sh; umask 0022; chmod -R go-w .
bash build.sh -c ascend910_93            # A3;默认编全部融合算子
./output/CANN-custom_ops-*-linux.aarch64.run --quiet --install-path=$CANN_HOME/opp
```
→ 装到 `$CANN_HOME/opp/vendors/customize`(含 `libcust_opapi.so`、`op_api/op_impl/op_proto`)。

### 2.4 custom_ops torch 绑定(cann-recipes-infer)
```bash
cd cann-recipes-infer/ops/ascendc/torch_ops_extension
USE_NINJA=1 bash build_and_install.sh    # setup.py build_ext + bdist_wheel + pip install -I
```
→ 装出 `custom_ops` wheel(本机 `custom_ops 1.0`)。它注册 `torch.ops.custom.*` 的 python binding,底层 aclnn 由上面两个 vendor 提供。

### 2.5 custom_transformer vendor（ops-transformer master）★最关键
NSA/DSA 三个算子在 **9.0.0 分支被删了**,只在 **master** 的 `experimental/attention/`。用干净 worktree 编,避免脏工作树污染：
```bash
git -C ops-transformer worktree add ../ops-transformer-master master
cd ops-transformer-master
source $CANN_HOME/set_env.sh; umask 0022; chmod -R go-w .
bash build.sh --pkg --experimental --soc=ascend910_93 --vendor_name=custom \
  --ops=sparse_attn_sharedkv,sparse_attn_sharedkv_metadata,compressor,quant_lightning_indexer,quant_lightning_indexer_metadata \
  --cann_3rd_lib_path=<ops-transformer>/third_party -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=$CANN_HOME/opp
```
- **vendor 命名怪癖**：ops-transformer 会给 vendor 名自动追加 `_transformer`,所以传 `--vendor_name=custom`,最终得到 vendor **`custom_transformer`**。
- `--cann_3rd_lib_path` 指向**主仓**(非 worktree)的 `third_party`(worktree 里没有)。

### 2.6 sgl_kernel_npu（tag 2026.6.2）
```bash
git clone https://github.com/sgl-project/sgl-kernel-npu && cd sgl-kernel-npu
git checkout 2026.6.2 && git submodule update --init --recursive
source $CANN_HOME/set_env.sh; umask 0022; chmod -R go-w .
bash build.sh                            # 默认 SOC=Ascend910_9382=A3
pip install output/{sgl_kernel_npu,deep_ep,attentions,torch_memory_saver}*.whl -c dsv4_torch_lock.txt
```
四个坑(脚本都已内建处理)：
1. **缺 `-ldl`**：`csrc/attentions/csrc/CMakeLists.txt` 的 `target_link_libraries(PTAExtensionOPS ...)` 没链 dl → `undefined reference to dlopen/dlsym`。补 `${CMAKE_DL_LIBS}`。(build.sh 是 `set -e`,任一子步骤挂 → output/ 里没 wheel。)
2. **umask**(同 2.1)。
3. **deep_ep vendor 只读**：`.run` 把 `python/deep_ep/deep_ep/vendors/hwcomputing/` 装成只读,二次 build 在 `rm uninstall.sh` 处 `Permission denied`。重跑前：`chmod -R u+w python/deep_ep/deep_ep/vendors && rm -rf .../hwcomputing`。
4. **别删 `csrc/attentions/build/`**：它是 **tracked 源码目录**(装内层 build.sh + kernel 脚本),不是产物。误删用 `git checkout -- csrc/attentions/build/` 恢复。只清 `csrc/build_out`、`output/`、vendor 目录。

### 2.7 kt-kernel（本仓内）
```bash
cd ktransformers-AK/kt-kernel && rm -rf build/temp.linux-aarch64-cpython-311
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CPUINFER_USE_ASCEND_NPU=1 \
  CPUINFER_ARM_SVE=OFF CPUINFER_ARM_BF16=OFF CPUINFER_ARM_I8MM=OFF \
  python3.11 setup.py build_ext --inplace
pip install safetensors gguf
```
- **ARM 扩展必须全关**：`SVE=ON` 会让 MXFP4 CPU MoE 走 SVE 分支,在 `operators/llamafile/moe.hpp:73/77` 报 `llamafile not supported` → decode 时 C++ `std::runtime_error` → 调度器 SIGABRT。关掉走验证过的 NEON `armv8.2-a+fp16+dotprod` 路径。
- **CMakeLists 认 CC/CXX**：原本强制覆盖成 `/usr/bin/gcc`;本仓已改成优先认 `$ENV{CC}/$ENV{CXX}`(`kt-kernel/CMakeLists.txt`,可移植、值得提交)。

### 2.8 运行期 vendor env（拉服务前 source）
```bash
source $CANN_HOME/set_env.sh
source $CANN_HOME/opp/vendors/custom_transformer/bin/set_env.bash
source $CANN_HOME/opp/vendors/customize/bin/set_env.bash
```
(`p27_launch_ds4flash_npu.sh` 已内置这些 source。)

---

## 3. 验证 import gate
```bash
bash tools/setup_dsv4_env_from_clean_cann.sh verify
```
通过标准(脚本会断言)：
```
torch 2.8.0+cpu  torch_npu 2.8.0.post4
triton OK / kt_kernel OK / sgl_kernel_npu OK / custom_ops OK
torch.ops.custom.* 全部就位 OK   # compressor / npu_sparse_attn_sharedkv / npu_quant_lightning_indexer / npu_moe_gating_top_k
```

---

## 4. 装完拉服务(冒烟)
```bash
cd /mnt/workspace/gitCode/ktransformers-AK
# ★两个默认路径脚本里写死成 /workspace,本机在 /mnt/workspace,必须显式给;
# ★CANN 9.0.0 走公开 single-state compressor,必须 export KT_NSA_COMPRESSOR_MODE=single
#   (默认是 split=旧 8.5.0 私有算子;不设会崩在 compressor 参数不匹配)。
NPU_DEVICE_ID=1 PORT=8020 SKIP_WARMUP=1 KT_STREAM_WARMUP=0 \
  KT_NSA_COMPRESSOR_MODE=single \
  MODEL_PATH='/mnt/workspace/models/DeepSeek-V4-Flash-W8A8' \
  KT_GGUF_TEMPLATE='/mnt/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  EXTRA_FLAGS="--disable-cuda-graph" \
  bash tools/p27_launch_ds4flash_npu.sh          # eager;权重加载 ~600s

curl --noproxy '*' http://127.0.0.1:8020/generate -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}'
# -> " Paris."        ;  "15 * 17 = " -> 首 token = 255(数值正确性详见 A3_W8A8_数值对齐调查.md)
```
> 注：本机测服务要 `curl --noproxy '*'`（`http_proxy=127.0.0.1:7890` 会把 localhost 拦成 502）。`NPU_DEVICE_ID` 是容器内逻辑号(0/1),不是 npu-smi 的 phy-id(2/3)。

---

## 5. 与已有 patch 包的关系
- 本文 = **环境/依赖 bring-up**(算子仓库都是**原样编译安装,C++ 源码零改动**;唯一的源码级修补是 sgl-kernel-npu 的 `-ldl` 和 kt-kernel 的 CMakeLists 认 CC/CXX)。
- **模型代码改动**(sglang fork 相对开源基线的工程 patch)见 `tools/kt_dsv4_npu_patches/`。
- **NSA 数值对齐**(单 state_cache 交织池 / fp32 norm-rope 等最小必要 3 文件)见 `A3_W8A8_数值对齐调查.md`。
