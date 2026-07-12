# 从干净 CANN 9.0.0 镜像复现：DeepSeek-V4-Flash 单卡 910C(A3) 环境 + 自定义算子

> 目标：在一台**只装了 CANN 9.0.0 的干净镜像**上,从 0 把 DSV4-Flash 单卡 W8A8 服务跑起来。
> 本文 = 在新 910C(A3) 裸机实测跑通的**确切步骤 + 每一步的坑**。
>
> **⚠️ 完整链路 = 四段,缺一段服务起不来:**
>
> | # | 段 | 覆盖在哪 | 一键脚本 |
> |---|---|---|---|
> | 1 | **算子 + 依赖**(三件套 vendor / custom_ops / sgl_kernel_npu / kt-kernel / sglang deps) | §1–§2(本文主体) | `setup_dsv4_env_from_clean_cann.sh` |
> | 2 | **depool 的 AscendC MXFP4 算子**(`KT_MXFP4_DEPOOL=1` 是 launcher **默认**,所以**默认必需**) | **§2.9**(新增) | 首次用到时自动 bisheng 编译 |
> | 3 | **模型权重 + MXFP4 GGUF**(W8A8 ckpt + 43 层 GGUF;不备好 launcher 直接找不到权重) | **§3**(新增,链到转换指南) | `batch_convert_mxfp4_layers_mp.py` |
> | 4 | 拉服务 + 冒烟 | §5 | `p27_launch_ds4flash_npu.sh` |
>
> 早期版本只写了第 1 段 —— **开源用户照着装完算子仍跑不起来**,故补齐 2/3 段。
>
> - 脚本:[`tools/setup_dsv4_env_from_clean_cann.sh`](../../../tools/setup_dsv4_env_from_clean_cann.sh)
> - 依赖清单:[`tools/dsv4_sglang_base_reqs.txt`](../../../tools/dsv4_sglang_base_reqs.txt) · [`tools/dsv4_torch_lock.txt`](../../../tools/dsv4_torch_lock.txt)
> - 权重/GGUF:[`mxfp4_gguf_conversion.md`](mxfp4_gguf_conversion.md)(转换 + 三级校验,面向开源用户)
> - 装完拉服务:[`tools/p27_launch_ds4flash_npu.sh`](../../../tools/p27_launch_ds4flash_npu.sh)(见 §5)
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

> ⚠️ **脚本只覆盖上表(= 第 1 段)。** 跑完 `verify` 通过 ≠ 能起服务,还差:
> **§2.9 depool 的 AscendC MXFP4 算子**(默认路径必需)+ **§3 模型权重与 MXFP4 GGUF**。

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

### 2.9 depool 的 AscendC MXFP4 算子 ★默认必需(setup 脚本不覆盖)

`KT_MXFP4_DEPOOL` 在 launcher 里**默认 =1**,而 depool 需要一个 **AscendC device kernel**
(MXFP4 dequant + ND→NZ,device→device,ctypes 调)。**`setup_dsv4_env_from_clean_cann.sh` 不编它**,
必须单独处理,否则第一次长 prefill 就崩在找不到 `libmxfp4fused.so`。

- **源码已入库**:[`tools/ascendc_mxfp4/`](../../../tools/ascendc_mxfp4/)(`mxfp4_{fused,dq}_kernel.cpp` + host launcher `mxfp4_fused_op.py` + 自验)。
- **`.so` 被 gitignore(不入库)→ 换机/首跑必须现编。**
- **方式 1(推荐,零操作)**:depool 首次用到算子时**自动 bisheng 编译并缓存**(源码比 `.so` 新会自动重编)。只需启动时让 `KT_MXFP4_OP_DIR` 指向本仓的 `tools/ascendc_mxfp4`(launcher 已默认)。
- **方式 2(手动确认编译链)**:
  ```bash
  cd tools/ascendc_mxfp4
  CANN=$CANN_HOME; TK=$CANN/aarch64-linux/tikcpp     # ★用 9.0.0 的 CANN_HOME(总纲 §4.6 里写的 8.5.0 路径已过时)
  bisheng -x asc --cce-aicore-arch=dav-c220 -O2 -std=c++17 -fPIC -shared \
    -I$TK/tikcfw -I$TK/tikcfw/impl -I$TK/tikcfw/interface -I$TK/tikcfw/lib \
    -I$CANN/aarch64-linux/include \
    mxfp4_fused_kernel.cpp -o libmxfp4fused.so \
    -L$CANN/aarch64-linux/lib64 -lruntime -lascendcl    # 无输出即成功
  ASCEND_RT_VISIBLE_DEVICES=<空卡> python3 test_fused_e2e.py   # 算子自验
  ```
- **不想用 depool**:显式 `KT_MXFP4_DEPOOL=0`,则不需要此算子(但会退回 277GB 常驻 W8A8 池,内存代价大)。详见总纲 §4.6。

---

## 3. 模型权重与 MXFP4 GGUF ★算子装完 ≠ 能跑

launcher 需要**两份权重**,都不在仓库里,必须自备:

| 权重 | 用途 | 从哪来 |
|---|---|---|
| **W8A8 safetensors**(`MODEL_PATH`) | NPU 侧(attention / 常驻专家 / embed / lm_head) | 官方 DeepSeek-V4-Flash checkpoint 经 **modelslim W8A8 量化** |
| **43 层 MXFP4 GGUF**(`KT_GGUF_TEMPLATE`) | CPU offload MoE(kt-kernel)+ depool 流式 prefill | 由**官方原生 MXFP4 专家权重**无损 bit-repack |

> ### ⚠️ 硬约束:两份权重的「量化基底」必须一致(坑⑰)
> CPU 侧 GGUF 与 NPU 侧 W8A8 **必须来自同一个量化基底**(quarot 旋转)。若你**自己**用 modelslim 量化 W8A8、
> 而 GGUF 来自官方原生 MXFP4(未旋转),两边基底不一致 → **输出乱码**(且不报错,极难查)。
> 详见 [`modelslim_quarot_basis_gguf_pitfall.md`](modelslim_quarot_basis_gguf_pitfall.md)。
> **最稳妥:两份权重同源**(要么都用官方发布的,要么自量化时保证 GGUF 也走同一基底)。

**MXFP4 GGUF 转换 → 见 [`mxfp4_gguf_conversion.md`](mxfp4_gguf_conversion.md)**(独立完整指南:转换 + 三级校验 + kernel 数值对账)。要点:

```bash
# 依赖:llama.cpp 子模块(公开 tag b3173) + patch 0001(NumPy2) / 0002(MXFP4 类型)——见总纲 §4.2
python3 tools/batch_convert_mxfp4_layers_mp.py --out-dir /path/to/cache ...   # 全量 43 层,多进程
python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache ...               # L1 齐全+尺寸 / L2 sha256 / L3 bit-exact
```
- 全程**无损 bit-repack**(不是再量化):GGUF 反量化与官方 checkpoint 反量化**逐元素 bit-exact**;
- 校验清单 [`tools/mxfp4_gguf_sha256.txt`](../../../tools/mxfp4_gguf_sha256.txt) 可直接比对;
- 换新机/改 kernel 后建议再跑一次 kernel 数值对账(cosine ≥ 0.999)。

> **路径坑**:launcher 里 `KT_GGUF_TEMPLATE` 默认写死 `/workspace/models/cache/...`;若你的 GGUF 不在那,**必须显式 export**(`{layer_idx}` 是占位符,别用 `${...:-}` 包,bash 会把第一个 `}` 当结束符吃掉)。

---

## 4. 验证 import gate
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

## 5. 装完拉服务(冒烟)
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

## 6. 与已有 patch 包的关系
- 本文 = **环境/依赖 bring-up**(算子仓库都是**原样编译安装,C++ 源码零改动**;唯一的源码级修补是 sgl-kernel-npu 的 `-ldl` 和 kt-kernel 的 CMakeLists 认 CC/CXX)。
- **模型代码改动**(sglang fork 相对开源基线的工程 patch)见 `tools/kt_dsv4_npu_patches/`。
- **NSA 数值对齐**(单 state_cache 交织池 / fp32 norm-rope 等最小必要 3 文件)见 `A3_W8A8_数值对齐调查.md`。
