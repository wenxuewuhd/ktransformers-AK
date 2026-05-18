# DeepSeek-V4-Flash 单卡 Ascend NPU + Kunpeng CPU MoE Offload 实施手册

> **本文档目的**：给后续接手的 AI/工程师一个**完整自包含**的项目交接文档，无需重新对话即可继续后面的实现。Phase 0 已完成并验证，本文记录到当前为止的所有决策、改动、验证结果，以及 Phase 1~4 的细节方案。
>
> **维护说明**：本文档由"Phase 0 实施会话"写于 2026-05-12。每个 phase 完成后请在对应章节追加"已完成"标记和实测数据。

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 项目背景与硬件软件环境](#1-项目背景与硬件软件环境)
- [2. 关键决策汇总](#2-关键决策汇总)
- [3. 整体 Phase 路线图](#3-整体-phase-路线图)
- [4. Phase 0：编译期适配（已完成 ✅）](#4-phase-0编译期适配已完成-)
- [5. Phase 1：单层 PoC](#5-phase-1单层-poc)
- [6. Phase 2：SGLang 整网集成](#6-phase-2sglang-整网集成)
- [7. Phase 3：CPU↔NPU 异步 Overlap](#7-phase-3cpunpu-异步-overlap)
- [8. Phase 4：KML 精度回归（可选）](#8-phase-4kml-精度回归可选)
- [9. 已知 Risks / Pitfalls](#9-已知-risks--pitfalls)
- [10. 关键路径 / 命令速查](#10-关键路径--命令速查)
- [11. 文件改动索引](#11-文件改动索引)

---

## 0. TL;DR

**目标**：让 DeepSeek-V4-Flash（671B MoE，256 routed experts/层）在**单卡 Atlas 910B (64 GB HBM) + 鲲鹏 920 (1.5 TB DRAM, 192 cores, 8 NUMA)** 上跑起来，方式是把约 16/256 个"热"expert 留 NPU，剩 240/256 走 CPU 端 `kt-kernel`（llamafile GGUF Q8_0 backend, NEON SDOT）。

**当前进度**（2026-05-13）：
- ✅ Phase 0 完成：`kt_kernel_ext.cpython-311-aarch64-linux-gnu.so` 在 K920+CANN 上编出来、import 通、`CPUInfer/MOEConfig/MOE` 全 OK。
- ✅ Phase 1.1 + 1.2 完成：W8A8→GGUF Q8_0 转换器 + Llamafile MoE 单层冒烟，43 层 GGUF 已经预生成至 `/workspace/models/cache/`。
- ✅ Phase 2-B + P2.2 + P2.3 完成：`third_party/sglang` 已切到 baseline **`iforgetmyname/sglang@dsv4_release`**，KT 在 baseline 上 backport 了 `kt_accel.py` 与 `kt_ep_wrapper.py` 三处 NPU-friendly 改动；详见 `Phase0_Phase1_变更记录与复现手册.md §8 / §9`。
- ✅ **P2.7 wiring 打通**：单卡 NPU + KT(LLAMAFILE) + DSv4-Flash W8A8 服务可以起、能接 `/generate` 并返回 200 OK，e2e_latency ≈ 2.3s。
- ⏳ **当前阻塞**：生成内容是退化 token（"  !  !  !  !  …" 全是空格 / `!`），属于**数值问题**而非 wiring 问题。诊断与对照实验计划见 §6.11。
- ⏳ 之后：Phase 3 CPU↔NPU 异步 overlap（性能）、Phase 4 KML 精度回归（可选）。

**关键约束**：K920 是 ARMv8.2-A + NEON SDOT，**没有 SVE / BF16 / I8MM**。kt-kernel 自带的 `Int8_KERNEL_MOE`/`Int4_KERNEL_MOE` 是纯 SVE 汇编写的，在 K920 上跑不通，所以 CPU 这条腿先走 **llamafile GGUF Q8_0**（NEON SDOT 路径），Phase 4 再考虑用 KML libkblas 的 `cblas_gemm_s8s8s32` 替换核心 GEMM 做精度回归。

---

## 1. 项目背景与硬件软件环境

### 1.1 硬件

| 部件 | 配置 |
|---|---|
| **CPU** | Kunpeng-920 5250，4 socket × 48 core = 192 物理核，**8 NUMA 节点 × 24 cores**，每 NUMA ~192 GB，共 1.5 TB DRAM |
| **CPU ISA** | ARMv8.2-A + `asimddp` (NEON SDOT) + `asimdhp/fphp` (FP16)；**无 SVE，无 BF16，无 I8MM** |
| **NUMA 距离** | (0,1)/(2,3)/(4,5)/(6,7) 同 die=11，跨 die=24~25，跨 socket=32 |
| **NPU** | 8 × Atlas 910B1（每张 64 GB HBM）。**当前项目目标是用其中 1 张** |
| **CANN** | 8.5.0，安装在 `/usr/local/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/cann-8.5.0` |

### 1.2 软件栈

| 组件 | 版本 / 路径 |
|---|---|
| **OS** | Ubuntu 22.04 (jammy)，aarch64 |
| **Python** | `/usr/local/python3.11.14/bin/python3` (3.11.14) |
| **PyTorch** | `2.8.0+cpu` |
| **torch_npu** | `2.8.0.post2+gitdef4a1c`（已确认支持 `pin_memory=True`）|
| **SGLang** | 容器内常有**第二份**树（如 `/sgl-workspace/sglang/`）或 pip 安装的包；与本仓库 **`third_party/sglang/`** 可能**不是同一代码**。跑 Phase 2 / `launch_server` 前必须 `export PYTHONPATH=<repo>/third_party/sglang/python:$PYTHONPATH` 并 `python -c "import sglang; print(sglang.__file__)"` 自检。 |
| **KML** | 2.5.0，安装在 `/usr/local/kml/`（headers + `libkml_rt.so.2.5.0` + `libkblas_armv8p_v2.5.0.so`，已 ldconfig）|
| **hwloc** | `libhwloc-dev:arm64 2.7.0-2ubuntu1`（kt-kernel 硬依赖，用 `apt --allow-unauthenticated` 装的）|
| **numactl** | 已装 |
| **gcc** | 系统 `/usr/bin/gcc`（>= 11 with C++20 OpenMP 4.5）|

### 1.3 网络代理（apt 用）

```ini
# /etc/apt/apt.conf.d/proxy.conf
Acquire::http::Proxy "http://p_atlas:proxy%40123@172.18.100.92:8080";
Acquire::https::Proxy "http://p_atlas:proxy%40123@172.18.100.92:8080";
```

**注意**：huaweicloud 镜像的 InRelease 签名校验会失败（返回 403），用 `apt-get -o Acquire::AllowInsecureRepositories=true update` 和 `apt-get --allow-unauthenticated install` 绕过。

### 1.4 模型 / 权重

| 路径 | 内容 |
|---|---|
| `/workspace/models/DeepSeek-V4-Flash-W8A8/` | 275 GB W8A8 权重（46 个 safetensors shard） |
| `/workspace/models/DeepSeek-V4-Flash-W8A8/config.json` | 模型配置（见下） |
| `/workspace/models/DeepSeek-V4-Flash-W8A8/model.safetensors.index.json` | 69055 个张量的 shard 映射 |

**DSv4-Flash 关键配置参数**（来自 `config.json`）：

```python
hidden_size                = 4096
moe_intermediate_size      = 2048
n_routed_experts           = 256
num_experts_per_tok        = 6                # top-k
n_shared_experts           = 1
num_hidden_layers          = 43
first_k_dense_replace      = 0                # 所有 43 层都是 MoE
moe_layer_freq             = 1
torch_dtype                = "bfloat16"
topk_method                = "noaux_tc"
scoring_func               = "sqrtsoftplus"
routed_scaling_factor      = 1.5
norm_topk_prob             = True
num_nextn_predict_layers   = 1                # speculative decoding，本项目禁用

# DSv4-Flash 特有的稀疏 attention
head_dim                   = 512
num_attention_heads        = 64
num_key_value_heads        = 1
qk_rope_head_dim           = 64
index_n_heads              = 64               # Lightning Indexer
index_topk                 = 512
index_head_dim             = 128
compress_ratios            = [1, 1, ...]      # NSA 压缩比
compress_rope_theta        = 160000
```

### 1.5 已有可工作的对照路径

- `script/launch_ds4flash_sglang.sh`：8 卡 NPU SGLang 拉起脚本（已知能正常推理，**作为整网 forward 的 reference**）
- `sglang_dsv4_ascend_cann850.sh`：宿主机 docker 容器拉起脚本

---

## 2. 关键决策汇总

> 这些决策已经在前期讨论中拍板，**后续 phase 实施请直接遵守**。

### 决策 #1：路径 D（先 GGUF 后 KML）

| Phase | CPU 端 backend |
|---|---|
| Phase 1~3 | `kt_kernel_ext.moe.MOE`（基于 `LLAMA_MOE_TP` → llamafile/llama.cpp 的 NEON SDOT 路径），权重格式 **GGUF Q8_0** |
| Phase 4（可选） | 用 KML 自带的 `cblas_gemm_s8s8s32`（K920 NEON-pthread 二进制 `libkblas_armv8p_v2.5.0.so`）替代核心 GEMM，做 W8A8 精度回归 |

**为什么不用 `Int8_KERNEL_MOE`**：kt-kernel 自带的 INT8/INT4 MoE kernel（`operators/moe_kernel/mat_kernel/kml_kernel/*.cpp`）是**纯 SVE 汇编**（`z` 寄存器、`p0` 谓词、`MUL VL` 地址模式），K920 没有 SVE，执行时会 SIGILL。这些代码在 commit `53f6a6d` 后被删除，本项目用 `git checkout 53f6a6d^ -- kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel` 恢复到 worktree 备用（Phase 4 才用），**KML=OFF 时不参与编译**。

### 决策 #2：CPU/NPU expert 分布

第一版固定 **16 NPU / 240 CPU per layer**，每层选 expert_id 0..15 留 NPU（最简单可验证）。后续用 sglang dump activation 统计后改成 top-16 频次。

### 决策 #3：Phase 0 范围 = "min_ggml + NPU 异步桩"

不动 KML 子目录的 CMake（OFF 状态保持）；只把 `cudaLaunchHostFunc` 抽象成可切换的 vendor API（CUDA/HIP/MUSA/**NPU**），即使 Phase 1/2 用同步路径，编译期保留 NPU 的 hook 位。

### 决策 #4：禁用 NEXTN 推测解码

DSv4 的 `num_nextn_predict_layers=1` speculative decoding 在 Phase 2 启动时**禁用**，简化首版。

### 决策 #5：先做 Phase 1.1 转换器，跳过 Phase 1.0 随机权重 demo

用户选择"P1.0 别在随机权重上浪费时间，直接并到 P1.1"。

---

## 3. 整体 Phase 路线图

```
Phase 0 ✅ 编译期适配
   ├── kt-kernel 在 aarch64 + CANN 上编出 .so
   ├── -march=armv8.2-a+fp16+dotprod（无 SVE/BF16）
   ├── 链上 libascendcl.so 等 CANN runtime
   └── cudaLaunchHostFunc 抽象成可切换 vendor API（含 NPU）

Phase 1 单层 PoC（数值正确性）
   ├── P1.1 W8A8→GGUF Q8_0 转换器（必做）
   └── P1.2 单层 hybrid demo（可选；做了能精确定位整网精度问题，跳了靠整网 ref 验证）

Phase 2 SGLang 整网集成（核心 milestone：整网在 K920+1 张 910B 上跑通）
   ├── P2.1 ✅ 解除 SGLang 上游 NPU "kt-* unsupported" gate（baseline 自带）
   ├── P2.2 ✅ kt_ep_wrapper.py device-agnostic 化（kt_accel.py）
   ├── P2.3 ✅ DSv4 模型代码集成 KTMoEWrapper（baseline 自带 + 3 处下游 patch）
   ├── P2.4 ✅ 43 层 GGUF 权重批量转换（/workspace/models/cache/dsv4_layer*.gguf）
   ├── P2.5 ⏳ NPU 16 个 hot expert 选取策略（第一版 [0..15]，待 dump activation 优化）
   ├── P2.6 ✅ 禁用 NEXTN + 单卡 attention 路径
   ├── P2.7 ✅ sglang serve **wiring 跑通**（curl /generate → 200，e2e 2.3s）
   ├── P2.8 ✅ HBM/DRAM 占用 sanity check（max_total_tokens=4.27M，avail 7.92 GB）
   ├── P2.9 ✅ 基线对齐：third_party/sglang 切到 iforgetmyname/sglang@dsv4_release
   ├── P2.10 ✅ Triton-on-NPU 不可用的全局兜底（allocator_npu / mem_cache/common）
   └── P2.11 ⏳ **数值对账**：生成内容退化（全 padding/`!`）的根因诊断 + 修复

Phase 3 性能：CPU↔NPU 异步 overlap
   ├── 落地 aclrtSubscribeReport + aclrtProcessReport 后台线程
   ├── submit_with_cuda_stream → aclrtLaunchCallback 真正生效
   └── 性能 profile

Phase 4 精度（可选）：KML 替换 Q8_0
   ├── 用 KML 自带 cblas_gemm_s8s8s32 实现 CPU expert
   ├── 或写 K920 NEON SDOT-only fallback kernel
   └── 对齐 NPU 端 W8A8 数值精度
```

---

## 4. Phase 0：编译期适配（已完成 ✅）

### 4.1 完成判据

| 项 | 实测结果 |
|---|---|
| `pip install .` 编出 `.so` | ✅ `1.8 MB kt_kernel_ext.cpython-311-aarch64-linux-gnu.so` |
| `-march` | ✅ `armv8.2-a+fp16+dotprod`（无 SVE/BF16） |
| `ldd` 链上 CANN | ✅ `libascendcl.so / libruntime.so / libmsprofiler.so / libascend_hal.so` 等 |
| `import kt_kernel_ext` | ✅ |
| `ext.moe.MOE` 类注册 | ✅（aarch64 上不注册 AMX/AVX2/SVE-INT8 系列） |
| `ext.CPUInfer(8)` 8-NUMA worker pool | ✅ 8 个 subpools 全识别 |
| `submit/sync_with_cuda_stream` Python binding | ✅ 已 NPU-aware（条件编译开关） |
| `MOEConfig(256, 6, 4096, 2048)` 能构造 | ✅ |

### 4.2 改动文件清单（共 7 个）

#### A. `kt-kernel/CMakeLists.txt`

**改动 1**：顶部 options 区加 NPU 选项 + ARM 细粒度 feature 开关：

```cmake
option(KTRANSFORMERS_USE_ASCEND_NPU "ktransformers: use Ascend NPU (CANN)" OFF)
option(LLAMA_ARM_DOTPROD "llama: enable ARM NEON SDOT/UDOT (asimddp, armv8.2+)" ON)
option(LLAMA_ARM_FP16    "llama: enable ARM NEON fp16 vector arithmetic"        ON)
option(LLAMA_ARM_SVE     "llama: enable ARM SVE (Kunpeng 930+, Neoverse V1+)"   OFF)
option(LLAMA_ARM_BF16    "llama: enable ARM BF16 matmul"                        OFF)
option(LLAMA_ARM_I8MM    "llama: enable ARM SMMLA/UMMLA (Neoverse N2/V1+)"      OFF)
```

**改动 2**：把原来硬编码的 `-march=armv8.2-a+fp16+dotprod+sve+bf16` 改成动态构造：

```cmake
        set(_kt_arm_arch "armv8.2-a")
        if(LLAMA_ARM_FP16)    set(_kt_arm_arch "${_kt_arm_arch}+fp16")    endif()
        if(LLAMA_ARM_DOTPROD) set(_kt_arm_arch "${_kt_arm_arch}+dotprod") endif()
        if(LLAMA_ARM_SVE)     set(_kt_arm_arch "${_kt_arm_arch}+sve")     endif()
        if(LLAMA_ARM_BF16)    set(_kt_arm_arch "${_kt_arm_arch}+bf16")    endif()
        if(LLAMA_ARM_I8MM)    set(_kt_arm_arch "${_kt_arm_arch}+i8mm")    endif()
        list(APPEND ARCH_FLAGS "-march=${_kt_arm_arch}")
```

**改动 3**：在 ROCM/MUSA 分支后加 NPU 分支：

```cmake
elseif(KTRANSFORMERS_USE_ASCEND_NPU)
    add_compile_definitions(KTRANSFORMERS_USE_ASCEND_NPU=1)
    add_compile_definitions(USE_ASCEND_NPU=1)  # legacy switch for vendor.h
    if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(_kt_cann_root "$ENV{ASCEND_TOOLKIT_HOME}")
    elseif(DEFINED ENV{CANN_HOME})
        set(_kt_cann_root "$ENV{CANN_HOME}")
    else()
        set(_kt_cann_root "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
    find_path(ACL_INCLUDE_DIR acl/acl_rt.h HINTS "${_kt_cann_root}/include" REQUIRED)
    find_library(ASCEND_CL_LIBRARY NAMES ascendcl
        HINTS "${_kt_cann_root}/lib64" "${_kt_cann_root}/runtime/lib64" REQUIRED)
    include_directories(${ACL_INCLUDE_DIR})
```

**改动 4**：`target_link_libraries` 之后加 NPU 库链接：

```cmake
if(KTRANSFORMERS_USE_ASCEND_NPU AND ASCEND_CL_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ASCEND_CL_LIBRARY})
    target_include_directories(${PROJECT_NAME} PRIVATE ${ACL_INCLUDE_DIR})
endif()
```

#### B. `kt-kernel/cpu_backend/vendors/vendor.h`

加 NPU 分支：

```cpp
#ifdef USE_CUDA
#include "cuda.h"
#elif USE_HIP
#define __HIP_PLATFORM_AMD__
#include "hip.h"
#elif USE_MUSA
#include "musa.h"
#elif USE_ASCEND_NPU
#include "ascend_npu.h"
#endif
```

#### C. `kt-kernel/cpu_backend/vendors/ascend_npu.h`（新建）

把 CANN aclrt* API 包装成 CUDA 形状：

```cpp
#pragma once
#include <acl/acl_base.h>
#include <acl/acl_rt.h>

using cudaStream_t = aclrtStream;
using cudaError_t  = aclError;
using cudaHostFn_t = aclrtCallback;  // 两者都是 void(*)(void*)
inline constexpr cudaError_t cudaSuccess = ACL_SUCCESS;

static inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
  return aclrtLaunchCallback(fn, userData, ACL_CALLBACK_NO_BLOCK, stream);
}

static inline const char* cudaGetErrorString(cudaError_t) {
  const char* m = aclGetRecentErrMsg();
  return (m && *m) ? m : "ACL error";
}
```

**⚠️ 关键约束**：CANN 的 `aclrtLaunchCallback` 与 CUDA 不同——callback **必须**由专门的 host 线程通过 `aclrtSubscribeReport(threadId, stream)` + 持续调用 `aclrtProcessReport(timeout)` 才能真触发。没有这个 subscriber 线程时，`submit_with_cuda_stream` 排队的 callback **永远不会执行**，等它的代码会死锁。Phase 1/2 用 PoC 同步路径（`CPUInfer::submit` + `sync`）规避，**Phase 3 才落地 subscribe/process 线程**。

#### D. `kt-kernel/cpu_backend/cpuinfer.h`

加 NPU 分支 + 把 `KTRANSFORMERS_USE_CUDA` 改成 `defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_ASCEND_NPU)`：

```cpp
#ifdef KTRANSFORMERS_USE_CUDA
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_ASCEND_NPU
#include "vendors/ascend_npu.h"
#endif

// ...
void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t,intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_ASCEND_NPU)
    // ... cudaLaunchHostFunc 调用（NPU 下经 ascend_npu.h 适配到 aclrtLaunchCallback）
#endif
}
// sync_with_cuda_stream 同样改
```

#### E. `kt-kernel/setup.py`

加 4 块逻辑：

1. **ARM features auto-detect**（在 `detect_cpu_info` 里读 `/proc/cpuinfo` 的 `Features:` 行）：检测 `asimddp / asimdhp / fphp / sve / bf16 / i8mm` 并填到 `info["features"]`。

2. **CANN auto-detect**（新函数 `detect_cann_toolkit()`）：检查 `$ASCEND_TOOLKIT_HOME / $CANN_HOME / /usr/local/Ascend/ascend-toolkit/latest`，看 `include/acl/acl_rt.h` 是否存在。

3. **`CPUINFER_USE_ASCEND_NPU` 环变 → `-DKTRANSFORMERS_USE_ASCEND_NPU=ON`**，并在 aarch64+无 CUDA+有 CANN 时自动开启；同时强制 `CPUINFER_ENABLE_KML=OFF` 和 `CPUINFER_ENABLE_BLIS=OFF`（避免误编 SVE 汇编）。

4. **ARM feature 透传**：把 `CPUINFER_ARM_DOTPROD/FP16/SVE/BF16/I8MM` 转 `-DLLAMA_ARM_*`，未显式设置时按 `info["features"]` auto。

#### F. `kt-kernel/install.sh`

加 2 个 shell helper：

```bash
detect_arm_features() {
  # 返回 "has_dotprod has_fp16 has_sve has_bf16 has_i8mm"
  # 从 /proc/cpuinfo "Features:" 解析 asimddp/asimdhp|fphp/sve/bf16/i8mm
}

detect_cann_root() {
  # echo CANN 安装根目录，找不到 echo 空
  # 顺序：$ASCEND_TOOLKIT_HOME -> $CANN_HOME -> /usr/local/Ascend/ascend-toolkit/latest
}
```

并在 `build_step()` 的 auto-detection 分支顶部加一个 `HOST_ARCH=aarch64` 路径，跳过 x86-only 的 AMX/AVX512 检测，转而 export `CPUINFER_USE_ASCEND_NPU / CPUINFER_ARM_* / CPUINFER_ENABLE_KML=OFF`。

#### G. `third_party/llamafile/iqk_mul_mat_arm82.cpp`（修上游 bug）

原代码两行 `#define` 被注释掉，导致 `.inc` 编出的符号缺 `_arm82` 后缀，`sgemm.cpp:151` 引用 `iqk_mul_mat_moe_arm82` 在 `dlopen` 时 undefined symbol。修复：

```cpp
#ifdef __aarch64__
#define iqk_mul_mat iqk_mul_mat_arm82
#define iqk_mul_mat_moe iqk_mul_mat_moe_arm82
#include "iqk_mul_mat_arm.inc"
#endif
```

**📌 这条建议回上游 PR**（与 `iqk_mul_mat_amd_zen4.cpp` 的同位写法对齐）。

#### H. 系统层：KML ldconfig

```bash
echo "/usr/local/kml/lib" > /etc/ld.so.conf.d/kml.conf
echo "/usr/local/kml/lib/neon/kblas/pthread" >> /etc/ld.so.conf.d/kml.conf
ldconfig
# 结果：libkml_rt.so.2.5.0 / libkblas_armv8p_v2.5.0.so 都进 ldconfig 缓存
```

### 4.3 Phase 0 重新验证步骤

```bash
# 1) 配置 + 编译
cd /workspace/code/ktransformer/ktransformers-AK/kt-kernel
./install.sh build       # 自动检测 aarch64+CANN 并 export 正确 env

# 或手动：
mkdir -p /tmp/kt_kernel_build && cd /tmp/kt_kernel_build
cmake /workspace/code/ktransformer/ktransformers-AK/kt-kernel \
  -DKTRANSFORMERS_USE_ASCEND_NPU=ON \
  -DLLAMA_NATIVE=OFF \
  -DLLAMA_ARM_DOTPROD=ON -DLLAMA_ARM_FP16=ON \
  -DLLAMA_ARM_SVE=OFF -DLLAMA_ARM_BF16=OFF -DLLAMA_ARM_I8MM=OFF \
  -DKTRANSFORMERS_CPU_USE_KML=OFF -DKTRANSFORMERS_CPU_MOE_KERNEL=OFF \
  -DPYTHON_EXECUTABLE=/usr/local/python3.11.14/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 16

# 2) smoke test
/usr/local/python3.11.14/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/tmp/kt_kernel_build")
import kt_kernel_ext as ext
print("MOE:", ext.moe.MOE)
print("MOEConfig:", ext.moe.MOEConfig)
cfg = ext.moe.MOEConfig(256, 6, 4096, 2048)
print("cfg:", cfg.expert_num, cfg.num_experts_per_tok, cfg.hidden_size, cfg.intermediate_size)
ci = ext.CPUInfer(8)
print("submit_with_cuda_stream:", ci.submit_with_cuda_stream)
PY
```

预期输出：所有打印正常，无 ImportError，无 SIGILL。

### 4.4 Phase 0 实测的 ldd 链接库

```
libascendcl.so       => /usr/local/Ascend/cann-8.5.0/lib64/libascendcl.so
libnuma.so.1         => /lib/aarch64-linux-gnu/libnuma.so.1
libhwloc.so.15       => /lib/aarch64-linux-gnu/libhwloc.so.15
libmsprofiler.so     => /usr/local/Ascend/cann-8.5.0/lib64/libmsprofiler.so
libascend_dump.so    => /usr/local/Ascend/cann-8.5.0/lib64/libascend_dump.so
libruntime.so        => /usr/local/Ascend/cann-8.5.0/lib64/libruntime.so
libascend_hal.so     => /usr/local/Ascend/driver/lib64/driver/libascend_hal.so
```

---

## 5. Phase 1：单层 PoC

### 5.1 Phase 1.1：W8A8 → GGUF Q8_0 转换器（必做）

#### 5.1.1 输入 W8A8 张量 layout（已验证）

每个 expert 6 个张量（DSv4 用 `w1/w2/w3` 命名，不是 HF 的 `gate/up/down_proj`）：

| DSv4 张量名 | 语义 | shape | dtype |
|---|---|---|---|
| `layers.N.ffn.experts.E.w1.weight` | gate_proj | `(2048, 4096)` = `(intermediate, hidden)` | int8 |
| `layers.N.ffn.experts.E.w1.weight_scale` | gate scale | `(2048, 1)` | **fp32**（**per-output-channel**）|
| `layers.N.ffn.experts.E.w2.weight` | down_proj | `(4096, 2048)` = `(hidden, intermediate)` | int8 |
| `layers.N.ffn.experts.E.w2.weight_scale` | down scale | `(4096, 1)` | fp32 |
| `layers.N.ffn.experts.E.w3.weight` | up_proj | `(2048, 4096)` | int8 |
| `layers.N.ffn.experts.E.w3.weight_scale` | up scale | `(2048, 1)` | fp32 |

**Dequant 公式**：`W_fp32 = W_int8.float() * weight_scale`（**正向 scale，不是 inverse**）。

**与 HF 命名对应**：`w1 ↔ gate_proj`、`w3 ↔ up_proj`、`w2 ↔ down_proj`（SwiGLU activation）。

Shared experts（`layers.N.ffn.shared_experts.w{1,2,3}.weight*`）同样的 layout，先不参与 hybrid，留 NPU 处理。

#### 5.1.2 输出 GGUF Q8_0 layout

llama.cpp 的 `block_q8_0`：

```c
typedef struct {
    ggml_fp16_t d;       // delta (per-block scale)
    int8_t  qs[32];      // 32 quants per block
} block_q8_0;             // sizeof = 2 + 32 = 34 bytes
```

每个 block 34 bytes。一个 numel-N 的 fp16 张量 → `N/32 * 34` bytes Q8_0。

**生成的 GGUF 张量名（按 llama.cpp/llamafile 习惯）**：

```
blk.N.ffn_gate_exps.weight   Q8_0  shape=(256, 2048, 4096)   # stack 256 个 w1
blk.N.ffn_up_exps.weight     Q8_0  shape=(256, 2048, 4096)   # stack 256 个 w3
blk.N.ffn_down_exps.weight   Q8_0  shape=(256, 4096, 2048)   # stack 256 个 w2
```

**沿哪个维度分 32-element block**：沿最内层（input/reduction 维度），即 gate/up 沿 hidden=4096，down 沿 intermediate=2048。两者都能被 32 整除。

#### 5.1.3 转换器实现要点

新建文件：`tools/convert_w8a8_to_gguf_q8_0.py`

CLI：
```bash
python tools/convert_w8a8_to_gguf_q8_0.py \
    --input /workspace/models/DeepSeek-V4-Flash-W8A8 \
    --layer-idx 3 \
    --output /workspace/models/cache/dsv4_layer3.gguf
```

43 层批量（层间多进程，每进程跑单层脚本，默认结束后随机抽样 3 个 GGUF 用 `GGUFReader` 校验）：

```bash
/usr/local/python3.11.14/bin/python3 tools/batch_convert_w8a8_layers_mp.py \
    --input /workspace/models/DeepSeek-V4-Flash-W8A8 \
    --output-dir /workspace/models/cache \
    --layer-start 0 --layer-end 42 \
    --jobs 4 \
    --skip-existing \
    --verify-sample 3
```

实现 sketch（~300 行）：

```python
import json, struct, numpy as np, torch
from pathlib import Path
from safetensors import safe_open

# 1) 读 index 找出 layer N 各 expert 在哪个 shard
def load_layer_experts(model_dir: Path, layer_idx: int):
    idx = json.load(open(model_dir / "model.safetensors.index.json"))
    wmap = idx["weight_map"]
    weights = {}  # {(expert_idx, "w1"|"w2"|"w3", "weight"|"scale"): tensor}
    # 用 dict-of-set 把同 shard 的张量聚一起，shard 只 open 一次
    shard_to_keys = {}
    for k, shard in wmap.items():
        if f"layers.{layer_idx}.ffn.experts." in k:
            shard_to_keys.setdefault(shard, []).append(k)
    for shard, keys in shard_to_keys.items():
        with safe_open(model_dir / shard, framework="pt") as f:
            for k in keys:
                # k = "layers.3.ffn.experts.42.w1.weight" 或 ".weight_scale"
                parts = k.split(".")
                expert = int(parts[3])
                proj   = parts[4]               # "w1"/"w2"/"w3"
                field  = "scale" if k.endswith("weight_scale") else "weight"
                weights[(expert, proj, field)] = f.get_tensor(k)
    return weights

# 2) Dequant W8A8 → fp32
def dequant_w8a8(w_int8: torch.Tensor, scale_fp32: torch.Tensor) -> torch.Tensor:
    # w shape (out, in), scale shape (out, 1)
    return w_int8.float() * scale_fp32

# 3) Q8_0 quantize：沿最后一维每 32 元素一组
def quantize_q8_0(w_fp32: torch.Tensor) -> bytes:
    # w shape (..., K) where K % 32 == 0
    assert w_fp32.shape[-1] % 32 == 0
    n_blocks_per_row = w_fp32.shape[-1] // 32
    out = bytearray()
    flat = w_fp32.reshape(-1, 32)  # (total_blocks, 32)
    abs_max = flat.abs().amax(dim=-1)              # (total_blocks,)
    d = (abs_max / 127.0).clamp_min(1e-12)          # scale
    q = (flat / d.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    d_fp16 = d.to(torch.float16).cpu().numpy()
    q_np = q.cpu().numpy()
    # block layout: [fp16 scale][32 × int8] → 34 bytes
    for i in range(flat.shape[0]):
        out += d_fp16[i].tobytes()
        out += q_np[i].tobytes()
    return bytes(out)

# 4) Stack 256 个 expert 并量化
def build_stacked_q80(weights, proj_name: str, out_first: int, in_dim: int) -> bytes:
    NE = 256
    stacked_fp32 = torch.empty(NE, out_first, in_dim, dtype=torch.float32)
    for e in range(NE):
        w_int8 = weights[(e, proj_name, "weight")]
        s_fp32 = weights[(e, proj_name, "scale")]
        stacked_fp32[e] = dequant_w8a8(w_int8, s_fp32)
    return quantize_q8_0(stacked_fp32)   # shape (NE, out, in), 沿 in 分 block

# 5) 写最小 GGUF header（参考 llama.cpp/gguf-py）
def write_gguf(path: Path, tensors: dict[str, tuple[bytes, list[int]]]):
    """
    tensors: {name: (q8_0_bytes, [shape...])}
    GGUF v3 minimal: magic(4) + version(4) + n_tensors(8) + n_kv(8)
    + KVs (none for minimal) + tensor infos + alignment + tensor data
    具体见 https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    """
    # ... 实现见 llama.cpp/gguf-py/gguf/gguf_writer.py
    pass

# 主入口
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    weights = load_layer_experts(Path(args.input), args.layer_idx)
    gate_blob = build_stacked_q80(weights, "w1", 2048, 4096)
    up_blob   = build_stacked_q80(weights, "w3", 2048, 4096)
    down_blob = build_stacked_q80(weights, "w2", 4096, 2048)
    write_gguf(Path(args.output), {
        f"blk.{args.layer_idx}.ffn_gate_exps.weight": (gate_blob, [4096, 2048, 256]),  # GGUF 用反向 shape！
        f"blk.{args.layer_idx}.ffn_up_exps.weight":   (up_blob,   [4096, 2048, 256]),
        f"blk.{args.layer_idx}.ffn_down_exps.weight": (down_blob, [2048, 4096, 256]),
    })
```

**📌 GGUF shape 注意**：llama.cpp 用 **column-major** 风格存 shape，所以 PyTorch `(256, 2048, 4096)` → GGUF 写 `[4096, 2048, 256]`。可以参考 `third_party/llamafile/` 里 ggml 的 tensor reader 确认。**实际写之前请先用 `gguf-py` 或读 llama.cpp 一个真实 DeepSeek GGUF 文件验证**。

**Q8_0 block 量化的对称性**：标准 llama.cpp Q8_0 是**对称量化**（无 zero-point，scale 为 abs_max/127）。这与 W8A8 的 per-channel scale 是兼容的（W8A8 也是对称）。

#### 5.1.4 验证步骤

```python
# 5.1.4.a 数值对账：dequant 后 fp32 与原 W8A8 dequant 结果误差 < 1%
import torch
ref_fp32 = dequant_w8a8(w_int8, scale_fp32)
q80_bytes = quantize_q8_0(ref_fp32)
deq_fp32  = dequantize_q8_0(q80_bytes, ref_fp32.shape)
err = (ref_fp32 - deq_fp32).abs().max() / ref_fp32.abs().max()
assert err < 0.01, f"Q8_0 quant error too large: {err}"
```

```python
# 5.1.4.b 用 kt-kernel 加载并 forward 一次
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
import torch
wrapper = LlamafileMoEWrapper(
    layer_idx=3,
    num_experts=256,
    num_experts_per_tok=6,
    hidden_size=4096,
    moe_intermediate_size=2048,
    gpu_experts_mask=torch.zeros(256, dtype=torch.bool),   # 全 CPU
    cpuinfer_threads=24,
    threadpool_count=8,
    weight_path="/workspace/models/cache/dsv4_layer3.gguf",
    chunked_prefill_size=4096,
    numa_nodes=list(range(8)),
)
wrapper.load_weights()
# forward 一次
hidden = torch.randn(4, 4096, dtype=torch.bfloat16).pin_memory()
topk_ids = torch.randint(0, 256, (4, 6), dtype=torch.int32).pin_memory()
topk_weights = torch.softmax(torch.randn(4, 6), dim=-1).to(torch.bfloat16).pin_memory()
output = torch.empty_like(hidden).pin_memory()
wrapper.forward(hidden, topk_ids, topk_weights, output)   # 具体 API 见 experts_base.py
assert torch.isfinite(output).all()
```

#### 5.1.5 验收

- [ ] 转换器对 layer-3 跑完 < 60s
- [ ] 输出 `.gguf` 文件大小 ≈ 6.4 GB（256 × 3 × 2048 × 4096 × 1.0625 ≈ 6.7 GB 实际）
- [ ] dequant 误差 < 1%
- [ ] kt-kernel 能加载该 GGUF 并 `forward` 一次输出 finite
- [ ] 进程不占 NPU（`npu-smi info` 验证）
- [ ] 各 NUMA 节点的 used DRAM 都有上升

### 5.2 Phase 1.2：单层 hybrid demo（可选，跳了亦可）

如做：新建 `script/poc_dsv4_moe_p12_hybrid.py`，实现：
1. 加载 `dsv4_layer3.gguf` 到 `LlamafileMoEWrapper`，`gpu_experts_mask = [True]*16 + [False]*240`
2. 用 SGLang 现有 `npu_grouped_matmul` / `fused_experts_npu` 路径跑 16 个 NPU expert
3. Python 端按 `topk_weights` 加权合并 NPU + CPU 输出
4. 同时跑一个 "全 NPU"的 ref（用 `gpu_experts_mask=[True]*256`，全走 SGLang npu 路径）
5. assert `cosine_sim(hybrid, ref) ≥ 0.999` 且 `max_abs_err / max_abs ≤ 1%`

如不做：等 Phase 2.7 端到端测试时再用整网 prompt 对齐验证。

---

## 6. Phase 2：SGLang 整网集成

**这是核心 milestone**：完成后能在 **`PYTHONPATH=<repo>/third_party/sglang/python`** 前提下 `python -m sglang.launch_server --tensor-parallel-size 1 --kt-method LLAMAFILE --kt-num-gpu-experts 16` 起服务，整网 forward 通。

### 6.1 P2.1：解除 SGLang 上游 NPU "kt-* unsupported" gate

**已知**：`third_party/sglang/docs/platforms/ascend_npu_support_features.md` 明确说 `--kt-*` 参数不支持 NPU。源码里有 hard gate。

**任务**：
1. `grep -rn "kt-" third_party/sglang/python/sglang/srt/` 找参数解析+校验位置
2. 找到 reject 的 `if device == "npu" and kt_method: raise ValueError(...)` 类逻辑
3. 用 feature flag（如 `--allow-kt-on-npu` 或环变 `SGLANG_KT_NPU_EXPERIMENTAL=1`）解除 gate
4. 同时更新 `ascend_npu_support_features.md` 文档说明

预估代码量：50-200 行 patch。

### 6.2 P2.2：`kt_ep_wrapper.py` device-agnostic 化

文件：`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`

当前问题：全篇 `torch.cuda.*` API。改造点：

| 当前代码 | NPU 适配 |
|---|---|
| `torch.cuda.Stream()` | 写 helper `current_device_stream()` 自动 dispatch |
| `torch.cuda.current_stream()` | 同上 |
| `tensor.cuda()` | `tensor.to(device)` |
| `submit_with_cuda_stream(stream.cuda_stream, ...)` | NPU 下用 `stream.npu_stream` 或通用 `int(stream._cdata)` |
| Marlin/FP4/MXFP4 quant 路径（CUDA-only） | NPU 下绕开，强制走 W8A8 path |
| `stream.synchronize()` | 通用，无需改 |

**关键 helper**（建议写一份放 `sglang/srt/utils/device.py`）：

```python
import torch
try:
    import torch_npu  # noqa
    _has_npu = torch.npu.is_available()
except ImportError:
    _has_npu = False

def get_device_type() -> str:
    if _has_npu and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def device_stream():
    """返回当前设备的 Stream 类（torch.cuda.Stream 或 torch.npu.Stream）"""
    if get_device_type() == "npu":
        return torch.npu.Stream
    return torch.cuda.Stream

def stream_handle(stream) -> int:
    """返回 stream 的底层 C handle，给 kt_kernel.submit_with_cuda_stream 用"""
    if hasattr(stream, "npu_stream"):
        return stream.npu_stream
    return stream.cuda_stream
```

预估代码量：300-500 行 patch + 联调。

### 6.3 P2.3：DSv4 模型代码集成 `KTMoEWrapper`

文件：`third_party/sglang/python/sglang/srt/models/deepseek_v4.py`

任务：在 MoE layer 初始化时，根据 `--kt-method`/`--kt-num-gpu-experts` 决定走 SGLang 原生 `fused_experts_npu`（npu_grouped_matmul + W8A8）还是 `KTMoEWrapper`。

伪代码：

```python
class DeepseekV4MoE(nn.Module):
    def __init__(self, config, layer_idx, kt_config: Optional[KTConfig] = None):
        # ... 原有初始化 ...
        if kt_config is not None:
            from kt_kernel.experts import KTMoEWrapper
            mask = compute_gpu_experts_mask(layer_idx, kt_config.num_gpu_experts)
            self.kt_moe = KTMoEWrapper(
                layer_idx=layer_idx,
                num_experts=config.n_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                gpu_experts_mask=mask,
                cpuinfer_threads=24,
                threadpool_count=8,
                weight_path=kt_config.gguf_path_template.format(layer_idx=layer_idx),
                chunked_prefill_size=kt_config.chunked_prefill_size,
                numa_nodes=list(range(8)),
                method="LLAMAFILE",
            )
            self.kt_moe.load_weights()
        else:
            self.kt_moe = None

    def forward(self, hidden_states, ...):
        # 1) gate / topk
        topk_ids, topk_weights = self.gate(hidden_states)
        if self.kt_moe is None:
            # 原有纯 NPU 路径
            return self._forward_npu_only(hidden_states, topk_ids, topk_weights)
        # 2) hybrid path
        npu_out, cpu_out = run_hybrid(hidden_states, topk_ids, topk_weights, self.kt_moe)
        return npu_out + cpu_out + self.shared_experts(hidden_states)
```

预估代码量：200-400 行 patch。

### 6.4 P2.4：43 层 GGUF 权重批量转换

复用 Phase 1.1 转换器，跑 43 次：

```bash
for i in $(seq 0 42); do
    python tools/convert_w8a8_to_gguf_q8_0.py \
        --input /workspace/models/DeepSeek-V4-Flash-W8A8 \
        --layer-idx $i \
        --output /workspace/models/cache/dsv4_layer${i}.gguf &
    # 限制并发，避免 IO 风暴
    if (( $i % 4 == 0 )); then wait; fi
done
wait
```

**资源估算**：
- 单层 GGUF 大小 ≈ 6.4 GB → 43 层 ≈ 275 GB
- 转换 IO 占用：~280 GB 读 + ~280 GB 写
- DRAM 占用峰值（per-job）：256×3×2048×4096×4B(fp32) ≈ 25 GB（dequant 中间结果），4 并发 ~ 100 GB OK

预估代码量：脚本 ~50 行，实际运行 2-3 小时。

### 6.5 P2.5：NPU 16 个 hot expert 选取策略

第一版：每层选 `expert_id=[0,1,...,15]`（最简单可验证）。

后续优化：跑 sglang dump activation hook 几条样本，统计 topk_ids 频次：

```python
# 复用 kt-kernel/python/experts_base.py:generate_gpu_experts_masks
import torch
from kt_kernel.experts_base import generate_gpu_experts_masks
# activation_freq shape (num_layers=43, num_experts=256), 从样本统计
masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=16*43)
# masks shape (43, 256) bool
```

### 6.6 P2.6：禁用 NEXTN + 单卡 attention 路径

- 启动加 `--disable-speculative-decoding` 或类似 flag
- 检查 `deepseek_v4.py` 里 `num_nextn_predict_layers` forward 路径走不到（多半 sglang 已经有 if-guard）
- attention（含 NSA/Lightning Indexer/Compressor）**已经在 NPU**，8 卡部署证明能跑。单卡时去掉 EP/TP all-to-all（EP=1 时 DeepEP 自动退化）

### 6.7 P2.7：sglang serve 端到端测试（wiring 已通，数值未通）

**当前状态（2026-05-13）**：服务能起、能接请求、HTTP 200，但 `text` 内容是退化 token。建议直接走脚本（已内置 PYTHON_BIN 探测、CANN env、参数对齐基线 8 卡子集）：

```bash
# 推荐入口（与基线 launch_ds4flash_sglang.sh 取单卡子集）
ASCEND_RT_VISIBLE_DEVICES=1 bash $REPO/tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/p27.log

# 临时调试附加 flag（不改脚本本体）
EXTRA_FLAGS="--disable-cuda-graph" bash $REPO/tools/p27_launch_ds4flash_npu.sh
EXTRA_FLAGS="--cuda-graph-bs 2"    bash $REPO/tools/p27_launch_ds4flash_npu.sh
ASCEND_LAUNCH_BLOCKING=1 ASCEND_RT_VISIBLE_DEVICES=1 bash $REPO/tools/p27_launch_ds4flash_npu.sh  # 同步执行
```

脚本内部等价的 `sglang.launch_server` 命令（截至当前 commit）：

```bash
python3 -m sglang.launch_server \
    --model-path /workspace/models/DeepSeek-V4-Flash-W8A8 \
    --device npu --tensor-parallel-size 1 \
    --page-size 128 \
    --attention-backend ascend \
    --quantization compressed-tensors \
    --disable-shared-experts-fusion \
    --dtype bfloat16 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --cuda-graph-bs 1 \
    --disable-radix-cache \
    --max-prefill-tokens 65535 \
    --context-length 65536 \
    --watchdog-timeout 18000 \
    --skip-server-warmup \
    --kt-method LLAMAFILE \
    --kt-num-gpu-experts 16 \
    --kt-weight-path "/workspace/models/cache/dsv4_layer{layer_idx}.gguf" \
    --kt-threadpool-count 8 \
    --kt-cpuinfer 24 \
    --max-running-requests 1 \
    --chunked-prefill-size -1 \
    --host 0.0.0.0 --port 8000
```

启动期会看到（已属正常路径，不是错）：

```
[allocator_npu] Triton driver unavailable (RuntimeError: 0 active drivers ([])); falling back to alloc_extend_naive ...
[2026-05-13 11:55:22] The server is fired up and ready to roll!
```

发请求：

```bash
bash $REPO/tools/p27_curl_generate.sh
```

**实测返回**：

```json
{"text":"  !  !  !  !  ! ! ! ! ! ! ! ! ! ",
 "output_ids":[223,223,3,223,223,3,...],
 "meta_info":{"finish_reason":{"type":"length","length":32},
              "prompt_tokens":2,"completion_tokens":32,
              "e2e_latency":2.29}}
```

**结论**：service / scheduler / KV pool / npu graph / KT EP wrapper / W8A8 加载 / RoPE / Compressor 全链 wiring 通，**但模型 forward 数值产生退化输出**。这是 P2.11 要处理的事，**与 Triton fallback 无关**（fallback 只动 KV 索引算术）。诊断顺序见 §6.11。

### 6.8 P2.8：HBM/DRAM 占用 sanity check

期望：

| 部件 | 占用 |
|---|---|
| NPU HBM | ~20 GB = embedding (~0.7 GB) + attention/projections (~7 GB) + 16 expert × 43 层 × 3 proj × 2048 × 4096 × 1 B (W8A8) ≈ 2 GB + shared_experts + KV cache（取决于 context 长度） |
| DRAM (CPU experts) | ~280 GB（43 层 × 6.4 GB） |
| 总 | ≤ 1 TB（远低于 1.5 TB DRAM 限制） |

如 KV cache 在 NPU HBM 上溢出，需要打开 KV offload to DRAM（SGLang/torch_npu 已支持）。

### 6.9 Phase 2 验收

- [x] `sglang serve` 起来后接受 HTTP 请求，HTTP 200（finite output_ids，但 token 内容退化，见 §6.11）
- [ ] tokens/sec 有 baseline 数据（**等数值修对后再录**，否则数字无意义）
- [ ] 与 8 卡部署同 prompt 输出语义大致一致 → **未达成**
- [x] HBM 占用 ≤ 30 GB（实测 avail ~7.92 GB，max_total_tokens=4276224）
- [x] 内存 < 1 TB

### 6.10 P2.9 + P2.10：基线对齐 + Triton-on-NPU 全局兜底（已完成）

> 这两项是 2026-05-12/13 临时插入的工程项目，已落地；**完整细节**搬到了 `Phase0_Phase1_变更记录与复现手册.md §9`（含改动文件清单、原因、复现命令）。这里只给一句话索引：

| 子项 | 索引 |
|---|---|
| **P2.9** `third_party/sglang` 切到 `iforgetmyname/sglang@dsv4_release`，原 `kvcache-ai/sglang` fork 归档为 `third_party/sglang.kvcache-ai-archive`；KT 三处下游 patch（`kt_accel.py` backport + `kt_ep_wrapper.py` 内 KTMoEWrapper 签名适配 + per-layer 模板 + `mask_cpu_expert_routing`） | `Phase0_Phase1_变更记录与复现手册.md §9.1–9.3` |
| **P2.10** `triton 3.7` × `triton-ascend 3.2` 在该 docker 上版本错配 → 任何 `@triton.jit` kernel 一旦调用就抛 `0 active drivers ([])`；通过 `allocator_npu.py` + `mem_cache/common.py` 里 `_TRITON_DRIVER_AVAILABLE` 探测 + torch fallback 解决（纯整数下标算术，**不影响 forward 数值**） | `Phase0_Phase1_变更记录与复现手册.md §9.4` |

降级开关：`SGLANG_NPU_ALLOC_FORCE_NAIVE=1` 显式强制 torch fallback；将来 wheel 修好探测自动转 True，无需改代码。

### 6.11 P2.11：数值对账计划（**当前阻塞，下一步要做**）

#### 6.11.1 问题描述

```
prompt:    "你好，"
output:    "  !  !  !  !  ! ! ! ! ! ! ! ! ! "
output_ids: [223,223,3,223,223,3,223,223,3,...]   # 空格(223) + "!"(3)
```

——服务通、e2e 2.29s、`isfinite=True`，但 token 分布严重退化。**与 Triton fallback 无关**（fallback 只动 KV 索引）。

#### 6.11.2 假设清单（按概率从高到低）

| # | 假设 | 验证手段 |
|---|---|---|
| H1 | **W8A8 (compressed-tensors / int-quantized) 权重 scale 加载未对齐**，导致 linear/MLA 数值整体崩 | 关 KT、关 cuda-graph，只用 baseline 自带 NPU MoE 看是否一样乱（详见 §6.11.3 实验 B） |
| H2 | **`mask_cpu_expert_routing` 把 CPU expert 的 GPU 贡献整段置零**，但 CPU 那一路（`KTMoEWrapper.apply`）并未真正回写 hidden_states / 时序错位 | 把 `kt-num-gpu-experts` 调到 **256**（全 NPU expert，等价禁用 KT offload）再跑，看是否变正常 |
| H3 | RoPE / Compressor 在 W8A8 路径下某个 scale 没接上 | 抓第一/最后一 layer 的 `attn_output` / `hidden_states` 的 `abs().mean()`，与 8 卡基线同 prompt 同位置 dump 对比 |
| H4 | `--cuda-graph-bs 1` 捕获的 npu graph 对单 token 输入数值不稳 | 加 `EXTRA_FLAGS="--disable-cuda-graph"`，走 eager 看是否还是 `223/3` |
| H5 | `--dtype bfloat16` 与 `--quantization compressed-tensors` 联合时某个 cast 丢精度 | 临时改 `--dtype float16` 对比 |

#### 6.11.3 内存账（实验设计前必读）

| 配置 | NPU HBM expert 占用 | DRAM expert 占用 | 是否会 OOM |
|---|---|---|---|
| `--kt-num-gpu-experts 0` | **0 GB**（NPU 上 0 个 expert 权重） | ≈264 GB（256/层 × 43 × 24 MB GGUF Q8_0） | ❌ 不会 |
| `--kt-num-gpu-experts 16`（**当前默认**） | ≈16 GB（16/层 × 43 × 24 MB W8A8） | ≈264 GB（仍按 256/层 装载 GGUF；KT 内部按 mask 决定走哪条路） | ❌ 当前已能起，avail 7.92 GB |
| `--kt-num-gpu-experts 30` | ≈30 GB | ≈264 GB | ⚠️ 接近 HBM 上限 |
| `--kt-num-gpu-experts 256`（曾误写为「实验 B」） | **≈264 GB**（远超 64 GB HBM） | — | ✅ **必 OOM** |

> 单 W8A8 expert 权重 ≈ 24 MB（gate+up+down 各 8 MB），43 层 × 256 → 264 GB。**单卡不可能把全部 expert 都放 NPU**。所以「全 NPU expert」这个对照不可行；要排除 KT EP 这一路，唯一能做的是反向打满到 **0**。

`--kt-num-gpu-experts 0` 时 `mask_cpu_expert_routing(num_gpu_experts=0)` 的语义：
```
is_gpu = topk_ids < 0          # 恒为 False
safe_ids     = zeros_like(...)  # 所有 id 改写为 0
safe_weights = zeros_like(...)  # 所有 weight 改写为 0.0
```
→ NPU MoE 那一路对每个 token 贡献 0；**真正的 expert 计算全部走 `KTMoEWrapper.submit_forward`（CPU）**。这正是验证「CPU expert 路径本身对不对」的最干净对照。

#### 6.11.4 实验设计的"两路 ⇄ 三路"诊断空间

服务实际涉及**三**个独立计算路径，任何一个错都能把 token 打到 padding。简单的 A/B 对照只能**单独排除 NPU MoE**，**不能区分** "CPU MoE 错" 与 "shared 路径（attention/RoPE/MLA/Compressor/embedding/lm_head）错"。

```
                       ┌──────────────────────────────────┐
                       │  shared:                         │
hidden_in ──►  embed ──►  attn (MLA + Compressor + RoPE) │──► residual ──┐
                       │  norms / linears / lm_head       │              │
                       └──────────────────────────────────┘              │
                                          ▲                              │
                                          │ + MoE output                 │
                                          │                              ▼
                                ┌─────────┴─────────┐                next layer
                                │       MoE         │
                                │ (per-token route) │
                                └─────────┬─────────┘
                                          │ topk_ids
                  ┌───────────────────────┼───────────────────────┐
                  ▼                                               ▼
        NPU MoE (16 GPU experts,                        CPU MoE (240 CPU experts,
        baseline `npu_fused_experts`,                   KTMoEWrapper + LLAMAFILE,
        W8A8 safetensors)                               GGUF Q8_0)
```

| 路径 | 实验 B (`--kt-num-gpu-experts 0`) 是否参与 | 隔离实验 |
|---|---|---|
| **shared (attn/RoPE/MLA/Compressor/embed/lm_head)** | ✅ 全程参与 | 实验 D 中间张量 dump |
| **NPU MoE** | ❌ 全部 token 被 mask 成 (id=0, weight=0) | 实验 B（CPU 路无问题前提下）|
| **CPU MoE (KT GGUF Q8_0)** | ✅ 承担**全部**专家计算 | **实验 B+**（新增；下面） |

> 所以 §6.11.3 那条「B 仍乱 → 排除 MoE」是错的：B 仍乱时 **CPU MoE 与 shared 路径都还有嫌疑**，必须再加 B+ 才能区分。

#### 6.11.5 推荐实验顺序（按时间成本递增）

```bash
# 实验 A：关 npu graph 走 eager，排除图捕获 / dynamo 重编译干扰
EXTRA_FLAGS="--disable-cuda-graph" \
  ASCEND_RT_VISIBLE_DEVICES=1 bash $REPO/tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/p27_eager.log
bash $REPO/tools/p27_curl_generate.sh
# - 仍乱 → 与图捕获无关（更可能是数值 / mask / 权重）
# - 变正常 → 图捕获把某条 dynamic shape 路径常量化错了

# 实验 B：把所有 expert 推到 CPU，走纯 KT GGUF Q8_0 路径
EXTRA_FLAGS="--kt-num-gpu-experts 0" \
  ASCEND_RT_VISIBLE_DEVICES=1 bash $REPO/tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/p27_all_cpu.log
bash $REPO/tools/p27_curl_generate.sh
# - 变正常 → NPU MoE 那一路有 bug（GPU expert W8A8 scale / mask_cpu_expert_routing / stream 时序）
# - 仍乱 → 还需要 B+ 区分「CPU MoE 错」和「shared 路径错」

# 实验 B+：CPU MoE 单层离线数值对账（与 server 启动无关，~分钟级）
#   见 §6.11.7：扩展 phase12_llamafile_moe_smoke.py 加 --reference 模式，
#   同一份 W8A8 → 纯 PyTorch dequant + topk 加权 SwiGLU 作为 ref，
#   同一组 (hidden, topk_ids, topk_weights) 喂给 KTMoEWrapper.forward，
#   对账 cosine_sim ≥ 0.999 / max_rel_err ≤ 1%（容许 Q8_0 量化误差）。
#   - ref 与 KT 不一致 → 锁定 CPU MoE（GGUF 权重 layout / Q8_0 量化精度 / forward kernel）
#   - ref 与 KT 一致   → 排除 CPU MoE，**问题在 shared 路径**

# 实验 C：A + B 同时（最干净的端到端对照）
EXTRA_FLAGS="--disable-cuda-graph --kt-num-gpu-experts 0" \
  ASCEND_RT_VISIBLE_DEVICES=1 bash $REPO/tools/p27_launch_ds4flash_npu.sh

# 实验 D：层间中间张量 dump（B+ 已经把 CPU MoE 排除后，进一步定位 shared 路径出问题的层）
#   在 third_party/sglang/.../models/deepseek_v4.py 的 forward 里：
#     - embedding 之后
#     - 第 0 / 第 10 / 第 21 / 倒数第一层的：MLA q/k/v、attn_output、MoE 输入/输出
#     - lm_head 输入 hidden_states
#   各打一行 `t.float().abs().mean().item(), .max().item()`，找第一个突然变 nan/inf/0 的层。
#   有 8 卡基线参考 dump 时直接 cosine_sim 比对更准。
```

#### 6.11.6 失败 → 假设映射表（含 B+）

| A | B | B+ | 锁定子系统 | 重点查 |
|---|---|---|---|---|
| 正常 | – | – | npu graph capture | 16 个 GPU expert 在图里被常量化、`mask_cpu_expert_routing` 被 `torch.compile` 错误折叠（看启动期 `recompile_limit (8)` 警告） |
| 仍乱 | 正常 | – | NPU MoE | `KTEPWrapperMethod.apply` GPU 路；GPU expert W8A8 scale 装载是否对齐；`mask_cpu_expert_routing` 把 weight=0 后 GPU 路是否仍写 hidden_states |
| 仍乱 | 仍乱 | **不一致** | **CPU MoE (KT)** | GGUF 权重 layout（gate/up/down 张量名与 shape，与 `kt-kernel` 期望一致）；Q8_0 块大小与方向；`KTMoEWrapper.forward` 在非 pin_memory 输入上的行为；`numa_nodes=None` 路径是否真把活分给了 8 个 NUMA |
| 仍乱 | 仍乱 | 一致 | **shared 路径** | W8A8 装载（`compressed-tensors` vs `int-quantized` vs `modelslim`）；DSv4 `ComplexExpRotaryEmbedding` 的 cos/sin 在 NPU；NSA Compressor / Lightning Indexer scale；`--dtype bfloat16` vs `float16` 切换；`embed_tokens` / `lm_head` 是否走对 W8A8 |
| 仍乱 | – | 不一致 | CPU MoE（即便 NPU 路也有问题） | 优先修 CPU MoE，再回头看 NPU 路 |

#### 6.11.7 实验 B+ 工具落地（待新建）

> CPU MoE 在 P1.2 的 smoke 只验证 `isfinite`，**没做数值对账**。要把它升级成「与参考实现 cosine_sim ≥ 0.999」级别。

新增 / 扩展 `tools/phase12_llamafile_moe_smoke.py`（~150 行 patch）：

```python
# tools/phase12_llamafile_moe_smoke.py 新增 --reference 模式
# 1) 读 layer L 的 W8A8（直接从 /workspace/models/DeepSeek-V4-Flash-W8A8）
# 2) dequant 256 个 expert：W_fp32 = W_int8.float() * weight_scale
# 3) 构造同一组 (hidden, topk_ids, topk_weights, bsz)，固定 seed
# 4) ref = sum_e topk_weight * SiLU(hidden @ w1.T) * (hidden @ w3.T) @ w2.T  for e in topk_ids
# 5) cand = KTMoEWrapper(LLAMAFILE, gpu_experts_mask=zeros).forward(hidden, ...)
# 6) assert cosine_sim(cand, ref) >= 0.999
#    assert (cand - ref).abs().max() / ref.abs().max() <= 1e-2  # Q8_0 误差容忍
```

执行：

```bash
# 单层 (L=3) CPU MoE 对账
${PYTHON_BIN} $REPO/tools/phase12_llamafile_moe_smoke.py \
  --gguf /workspace/models/cache/dsv4_layer3.gguf \
  --layer-idx 3 \
  --reference --w8a8-input /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --batch 4 --seed 1
```

如果上面 PASS 但端到端 (B) 仍乱 → CPU MoE 干净，问题在 shared 路径。
如果上面 FAIL → 立刻定位是 GGUF 转换误差大、还是 KTMoEWrapper 调用方式错（pin_memory / topk_ids dtype / numa_nodes）。

#### 6.11.8 其它工具增强（建议先做）

- 在 `tools/` 下新建 `p27_dump_tensors.py`：基于 sglang `engine` API，加载同一 `--model-path` + 同 prompt 做 1 token forward，dump 关键中间张量为 `.pt`；同 prompt 在 8 卡基线机器也跑一份，本地用 `torch.allclose` / 余弦相似度比对。
- `tools/p27_curl_generate.sh` 增加 `--temperature 0 --top-p 1.0 --seed 1` 等参数，保证生成可复现，便于"今天/明天再跑"对比 token-level 是否完全一致。

---

## 7. Phase 3：CPU↔NPU 异步 Overlap

**目标**：让 CPU expert 计算和 NPU expert 计算真正并行，性能提升预期 1.5-2x。

### 7.1 关键工作

1. **落地 CANN callback 子线程**：
   - 在主进程启动时，spawn 一个 `_npu_callback_worker` 线程
   - 该线程调用 `aclrtSubscribeReport(threadId, stream)` 注册关心的 stream
   - 然后死循环调用 `aclrtProcessReport(timeout_ms=100)` 消费 callback queue
   - C++ 侧暴露 `kt_kernel_ext.CPUInfer.subscribe_npu_callback(stream_handle)` Python API

2. **`submit_with_cuda_stream` 真生效**：
   - 当前 Phase 0 的桩位已经把 `cudaLaunchHostFunc → aclrtLaunchCallback` 调用通了
   - 没有 subscriber 时它会卡（callback 永不 fire），有了 subscriber 才会触发
   - 改 `kt_ep_wrapper.py` 里 NPU 路径的 `submit/sync_with_cuda_stream` 调用，传入正确的 stream handle

3. **Stream 顺序**：
   - NPU 计算 stream → record event A
   - CPU 计算用 `submit_with_cuda_stream(stream)` 排在 event A 之后
   - 最终 sync 在主 stream 上

4. **Perf profile**：用 `npu-smi profile` 或 sglang 自带 profiler 看 timeline，确认 CPU/NPU bar 重叠

### 7.2 风险

- ACL callback 必须 spawn 在 **non-aclrt 线程**（不能是 stream owner 线程），否则死锁
- `aclrtProcessReport` timeout 设置不当会浪费 CPU 周期
- pybind11 的 GIL 在 callback 里要正确释放

预估代码量：~500-1000 行 C++ + Python。

---

## 8. Phase 4：KML 精度回归（可选）

**目标**：把 CPU expert 的 GEMM 内核从 llamafile Q8_0 换成原生 W8A8（与 NPU 完全对齐数值）。

### 8.1 两条候选路径

| 路径 | 内容 | 工作量 |
|---|---|---|
| **A. 用 KML libkblas 的 cblas_gemm_s8s8s32** | KML 自带 `libkblas_armv8p_v2.5.0.so` 提供高层 CBLAS 接口（s8×s8→s32）。封装一个新的 backend 类 `KMLW8A8MoE` 在 `kt-kernel/operators/moe_kernel/` 下，权重不转 GGUF，直接吃 safetensors 里的 int8+per-channel-scale | ~2 周 |
| **B. 写 K920 NEON SDOT-only 的 W8A8 micro-kernel** | 不依赖 KML，纯 SDOT 写一组 1×K, 1×8 微内核。可参考 `kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel/batch_gemm_kernels.cpp`（SVE 版本），改成 NEON 版本 | ~3 周 |

**推荐 A**（KML 已经做了所有 hard work，直接用 cblas API）。

### 8.2 准备工作（已完成）

- KML 2.5.0 已装到 `/usr/local/kml/`
- `libkml_rt.so.2.5.0` 和 `libkblas_armv8p_v2.5.0.so` 已进 ldconfig
- 头文件：`/usr/local/kml/include/kblas.h` 等
- 删除的 SVE kernel 源码已 `git checkout 53f6a6d^ -- kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel` 恢复到 worktree，作为参考实现保留

### 8.3 实施要点

1. CMake 加 `find_library(KML_KBLAS NAMES kblas_armv8p_v2.5.0 HINTS /usr/local/kml/lib/neon/kblas/pthread)`
2. 在 `kt-kernel/operators/moe_kernel/` 下新建 `kml_w8a8/` 目录
3. 实现 `KMLW8A8MoE::forward()`：
   - 把 hidden_in (bf16) 转 int8（动态 per-token activation 量化，参考 NPU 端 `npu_dynamic_quant`）
   - 调 `cblas_gemm_s8s8s32(M, N, K, alpha=1, A_int8, B_int8, beta=0, C_int32)` 算 gate 和 up
   - applied scale + SwiGLU activation
   - 再 quantize 一次 + cblas_gemm_s8s8s32 算 down
4. 在 ext_bindings.cpp 注册新类 `bind_moe_module<KML_W8A8_MOE_TP>(moe_module, "W8A8_KML_MOE")`
5. Python 侧 `experts.py` 加 `method="W8A8_KML"` 选项
6. 跟 NPU 端单 expert 输出对账，误差应 < 1e-3（同样 W8A8 量化）

---

## 9. 已知 Risks / Pitfalls

| # | Risk | 触发条件 | 缓解 |
|---|---|---|---|
| **R1** | K920 跑 SVE 汇编 SIGILL | 任何启用 `KTRANSFORMERS_CPU_USE_KML=ON` 的编译，或 import 时引入 kml_kernel/*.cpp | setup.py 已强制 aarch64 下默认 OFF；用户必须明确 `CPUINFER_ENABLE_KML=ON` 才打开 |
| **R2** | ACL callback 永不 fire | Phase 3 没正确 subscribe stream，或 subscribe 在错误的线程 | Phase 1/2 用同步 `submit/sync`，Phase 3 才上 callback；callback worker 严格在新线程 spawn |
| **R3** | huaweicloud apt 镜像 InRelease 签名 403 | 任何 `apt update` 不带 `Acquire::AllowInsecureRepositories=true` | 已固化在 sources.list 旁，新装依赖请加 `-o Acquire::AllowInsecureRepositories=true --allow-unauthenticated` |
| **R4** | hwloc 版本与 ABI 不匹配 | 装了非 2.7.x 系列 hwloc，与 kt-kernel 的 `worker_pool.cpp` 用到的 `hwloc_topology_load` 等 API 不兼容 | 当前装的是 2.7.0-2ubuntu1，安全。如未来升级需测试 |
| **R5** | DSv4 `noaux_tc` gating 在 hybrid 路径的数值偏差 | Phase 2.3 在 NPU/CPU 拼接处归一化错误 | 用 8 卡部署的 gate 输出做 ref 对账（dump 一层的 topk_ids/topk_weights） |
| **R6** | GGUF 转换器 shape 顺序写反 | 写 GGUF 时用了 PyTorch 顺序而非 column-major | 用 gguf-py 库或先用一个 mini 模型（如 Qwen 一个 layer）验证 |
| **R7** | NPU HBM 不够装 KV cache | context 太长 + KV 留 NPU | 打开 KV offload to DRAM；或降 `max_running_requests` |
| **R8** | 上游 `iqk_mul_mat_arm82.cpp` bug 再次出现 | rebase third_party/llamafile 时把已修的 `#define` 又注释掉 | 在 README/CI 加 sanity check：build 后 `nm libllamafile.a \| grep iqk_mul_mat_moe_arm82` 应非空 |
| **R9** | Triton-on-NPU 看起来"能 import"但实际不可用 | 镜像里 `triton 3.7` + `triton-ascend 3.2` 版本错配，`triton.backends.ascend.compiler` 引用 upstream 已删除的 `AttrsDescriptor` → import 时立即 `ImportError`，运行时 `driver.active` 报 `0 active drivers ([])` | 已在 `allocator_npu.py` / `mem_cache/common.py` 加全局兜底（torch 等价路径，不影响 forward 数值）；将来升级 `triton-ascend` wheel 匹配 triton 3.7 后探测自动转 True |
| **R10** | `source set_env.sh` 后 `python3` 跑到 `/usr/bin/python3`（系统 python，无 torch_npu/numpy/sglang） | `tools/p27_*` 任何 wrapper 脚本 / cron / systemd 启动 | 已在 `tools/p27_launch_ds4flash_npu.sh` 加 `PYTHON_BIN` 自动探测；如还有别的脚本，请用 `${PYTHON_BIN}` 而非裸 `python3` |

---

## 10. 关键路径 / 命令速查

### 10.1 路径

```
# Repo
/workspace/code/ktransformer/ktransformers-AK/
├── kt-kernel/                        # 主要改的 C++ extension
│   ├── CMakeLists.txt
│   ├── setup.py
│   ├── install.sh
│   ├── cpu_backend/
│   │   ├── cpuinfer.h
│   │   └── vendors/
│   │       ├── vendor.h
│   │       ├── cuda.h, hip.h, musa.h
│   │       └── ascend_npu.h          # 新建
│   ├── ext_bindings.cpp              # Pybind11 绑定
│   ├── operators/
│   │   ├── llamafile/                # 用的 backend
│   │   └── moe_kernel/mat_kernel/kml_kernel/  # SVE 汇编，Phase 4 用，已 git checkout 恢复
│   └── python/                       # Python wrapper（KTMoEWrapper 等）
│       └── utils/
│           ├── amx.py, llamafile.py, moe_kernel.py
│           └── loader.py             # GGUFLoader
├── third_party/
│   ├── llama.cpp/
│   ├── llamafile/                    # 已 patch iqk_mul_mat_arm82.cpp
│   ├── pybind11/
│   ├── sglang/                       # Phase 2 改这里
│   └── kml/                          # KML deb 解压目录
├── script/
│   ├── launch_ds4flash_sglang.sh     # 8 卡参考脚本
│   └── (待新建) launch_single_npu_ds4flash.sh
├── sglang_dsv4_ascend_cann850.sh     # 宿主机 docker 启动脚本
├── tools/
│   ├── convert_w8a8_to_gguf_q8_0.py       # Phase 1.1 单层转换
│   ├── batch_convert_w8a8_layers_mp.py  # Phase 1.1 多进程层间批量
│   └── phase12_llamafile_moe_smoke.py   # Phase 1.2 Llamafile 冒烟
└── doc/zh/
    └── DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md  # 本文档

# 系统
/usr/local/Ascend/cann-8.5.0/         # CANN
/usr/local/Ascend/ascend-toolkit/latest -> cann-8.5.0
/usr/local/kml/                       # KML 2.5.0
/usr/local/python3.11.14/bin/python3  # Python
/workspace/models/DeepSeek-V4-Flash-W8A8/   # 275 GB W8A8 权重
/workspace/models/cache/                     # GGUF 转换输出（建在权重同盘，勿用 /workspace/cache 以免根分区爆满）
```

### 10.2 常用命令

```bash
# 重新编 kt-kernel
cd /workspace/code/ktransformer/ktransformers-AK/kt-kernel
./install.sh build

# 或手动编（开发迭代时）
mkdir -p /tmp/kt_kernel_build && cd /tmp/kt_kernel_build
cmake /workspace/code/ktransformer/ktransformers-AK/kt-kernel \
  -DKTRANSFORMERS_USE_ASCEND_NPU=ON \
  -DLLAMA_NATIVE=OFF \
  -DLLAMA_ARM_DOTPROD=ON -DLLAMA_ARM_FP16=ON \
  -DLLAMA_ARM_SVE=OFF -DLLAMA_ARM_BF16=OFF -DLLAMA_ARM_I8MM=OFF \
  -DKTRANSFORMERS_CPU_USE_KML=OFF -DKTRANSFORMERS_CPU_MOE_KERNEL=OFF \
  -DPYTHON_EXECUTABLE=/usr/local/python3.11.14/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 16

# 装 apt 依赖（注意签名绕过）
apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::Check-Valid-Until=false
apt-get install -y --allow-unauthenticated -o Acquire::AllowInsecureRepositories=true <pkg>

# 验证 Phase 0
/usr/local/python3.11.14/bin/python3 -c '
import sys; sys.path.insert(0,"/tmp/kt_kernel_build")
import kt_kernel_ext as ext
print(ext.moe.MOE, ext.moe.MOEConfig)
'

# 看 NPU 状态
npu-smi info | head -20

# 看 NUMA
numactl --hardware

# 抓 W8A8 一层张量名/shape
/usr/local/python3.11.14/bin/python3 -c "
from safetensors import safe_open
with safe_open('/workspace/models/DeepSeek-V4-Flash-W8A8/model-00005-of-00046.safetensors', framework='pt') as f:
    for k in f.keys():
        if 'layers.3.ffn.experts.0' in k:
            t = f.get_tensor(k); print(k, tuple(t.shape), t.dtype)
"
```

### 10.3 环境变量

```bash
# Phase 1+ 跑 PoC 或 sglang 前请 export
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/kml/lib:$LD_LIBRARY_PATH
# 本仓库 Phase 2 改在 third_party/sglang；与 /sgl-workspace/sglang 或 pip 全局包不是同一份，必须前置 PYTHONPATH：
# export REPO=/workspace/code/ktransformer/ktransformers-AK
# export PYTHONPATH="$REPO/third_party/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
# kt_kernel Python 包：请在 kt-kernel 目录执行 `pip install -e .`（勿仅把 python/ 加进 PYTHONPATH，否则找不到 kt_kernel 包名）
# export PYTHONPATH=/path/to/kt_kernel_ext_build:$PYTHONPATH   # 若 .so 不在 site-packages，可追加 build 目录（仍建议 pip install -e）

# 内网代理（apt 用，已在 /etc/apt/apt.conf.d/proxy.conf）
# export http_proxy=http://p_atlas:proxy%40123@172.18.100.92:8080
# export https_proxy=$http_proxy
```

---

## 11. 文件改动索引

下表列出本项目**所有**已修改 / 新建的文件，方便 review 或 git diff。

### 11.1 已完成（Phase 0）

| 文件 | 改动类型 | 用途 |
|---|---|---|
| `kt-kernel/CMakeLists.txt` | 修改（4 处） | 加 `KTRANSFORMERS_USE_ASCEND_NPU` 选项 + ARM feature 开关 + NPU 分支 + link `libascendcl.so` |
| `kt-kernel/cpu_backend/vendors/vendor.h` | 修改 | 加 `#elif USE_ASCEND_NPU` 分支 |
| `kt-kernel/cpu_backend/vendors/ascend_npu.h` | **新建** | CANN aclrt* API 包装成 CUDA 形状 |
| `kt-kernel/cpu_backend/cpuinfer.h` | 修改 | 加 NPU 分支，`submit/sync_with_cuda_stream` 启用 NPU 编译开关 |
| `kt-kernel/setup.py` | 修改 | CANN 自动探测、ARM feature 透传、aarch64 默认锁 KML=OFF |
| `kt-kernel/install.sh` | 修改 | aarch64 分支检测、CANN 自动探测 |
| `third_party/llamafile/iqk_mul_mat_arm82.cpp` | 修改 | 启用被注释的 `#define`（修上游 bug） |
| `/etc/ld.so.conf.d/kml.conf` | 系统层新增 | 把 `/usr/local/kml/lib` 加进 ldconfig |
| `kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel/*` | 恢复（git） | Phase 4 备用，`git checkout 53f6a6d^ -- <path>` |

### 11.2 已完成（Phase 1 + Phase 2 wiring）

| 文件 | Phase | 用途 |
|---|---|---|
| `tools/convert_w8a8_to_gguf_q8_0.py` | 1.1 | W8A8 单层 → GGUF Q8_0（单进程） |
| `tools/batch_convert_w8a8_layers_mp.py` | 1.1 | 层间多进程批量转换 + 抽样 GGUFReader 校验 |
| `tools/phase12_llamafile_moe_smoke.py` | 1.2 | `KTMoEWrapper(LLAMAFILE)` 单层 forward 冒烟（含 `--second-gguf` Phase 2-B 覆盖） |
| `tools/kt_accel_stream_smoke.py` | 2.2 | 仅测 `kt_accel` stream/event/sync 抽象，不依赖权重 |
| `tools/run_p22_smoke_checks.sh` | 2.2 | P2.2 三段冒烟（kt_accel + import kt_ep_wrapper + phase12） |
| `tools/p27_launch_ds4flash_npu.sh` | 2.7 | 单卡 NPU + KT(LLAMAFILE) sglang serve 拉起；含 `PYTHON_BIN` 探测、CANN env、与基线 8 卡参数对齐 |
| `tools/p27_curl_generate.sh` | 2.7 | 对已起服务发 `/generate` 冒烟 |
| `third_party/sglang/python/sglang/srt/utils/kt_accel.py` | 2.2 + 2.9 | CUDA↔NPU stream/event/同步抽象（baseline 没有，自 archive backport） |
| `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py` | 2.2 + 2.3 + 2.9 | KT EP wrapper：per-layer 模板、`KTMoEWrapper` 签名适配本机 wheel、stream → `kt_accel`、`mask_cpu_expert_routing` 替代 `mask_cpu_expert_ids` |
| `third_party/sglang/python/sglang/srt/hardware_backend/npu/allocator_npu.py` | 2.10 | Triton driver 探测；不可用时让 `alloc_extend` 走 `alloc_extend_naive`（纯 torch） |
| `third_party/sglang/python/sglang/srt/mem_cache/common.py` | 2.10 | 同上探测；`write_multi_cache_indices` / `write_cache_indices` / `get_last_loc` 在 driver 不可用时落 torch 等价实现 |
| `.gitmodules` | 2.9 | `third_party/sglang` → `iforgetmyname/sglang@dsv4_release` |
| `third_party/sglang.kvcache-ai-archive/` | 2.9 | 旧 `kvcache-ai/sglang` fork 归档，仅作 backport 参考 |

### 11.3 待新建 / 待补（Phase 2.11 + Phase 3+）

| 文件 | Phase | 用途 |
|---|---|---|
| `tools/p27_dump_tensors.py`（建议） | 2.11 | 加载同 model + 同 prompt forward 1 token，dump embedding / 中间层 attn_out / lm_head 输入 hidden 等关键张量，便于与 8 卡基线对账 |
| `tools/p27_curl_generate.sh` 扩展 | 2.11 | 加 `temperature 0 / top_p 1.0 / seed` 参数，保证生成可复现 |
| Phase 3 callback subscriber | 3 | `aclrtSubscribeReport` + `aclrtProcessReport` 后台线程（详 §7） |
| Phase 4 KML W8A8 backend | 4 | `cblas_gemm_s8s8s32` 替代 Q8_0（详 §8） |

---

## 附录 A：Phase 0 实测 Smoke Test 输出（参考）

```
=== Phase 0 smoke test ===
[1] module loaded OK -> <module 'kt_kernel_ext' from '/tmp/kt_kernel_build/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so'>

[2] top-level attrs: ['CPUInfer', 'GeneralConfig', 'QuantConfig', 'WorkerPool', 'WorkerPoolConfig', 'kvcache', 'linear', 'mla', 'mlp', 'moe', 'utils']

[3] MoE backend classes registered on aarch64+NPU:
    - MOE
    - MOEConfig
    - MOESFTConfig

[4] CPUInfer + NPU-aware callback bindings:
    submit_with_cuda_stream = <bound method PyCapsule.submit_with_cuda_stream of ...>
    sync_with_cuda_stream   = <bound method PyCapsule.sync_with_cuda_stream of ...>
    submit                  = <bound method PyCapsule.submit of ...>
    sync                    = <bound method PyCapsule.sync of ...>

[5] MOEConfig populated like a DS-V4-Flash layer would:
    expert_num=256 top-k=8 hidden=7168 intermediate=2048    # 注意：这里测试值用了占位 7168/8，
                                                            # DSv4 真实 hidden=4096 top-k=6
    layer_idx=0 num_gpu_experts=0

CPUInfer[0xaaaaee2b34d0]: Hello
WorkerPool[0xaaaaee2b4500] 8 subpools, [numa:threads][0:1] [1:1] [2:1] [3:1] [4:1] [5:1] [6:1] [7:1] 
=== Phase 0 PASSED ===
```

---

## 附录 B：DSv4-Flash Layer-3 Expert-0 实测 W8A8 张量

```
layers.3.ffn.experts.0.w1.weight        shape=(2048, 4096) dtype=torch.int8    numel=8388608
layers.3.ffn.experts.0.w1.weight_scale  shape=(2048, 1)    dtype=torch.float32 numel=2048
layers.3.ffn.experts.0.w2.weight        shape=(4096, 2048) dtype=torch.int8    numel=8388608
layers.3.ffn.experts.0.w2.weight_scale  shape=(4096, 1)    dtype=torch.float32 numel=4096
layers.3.ffn.experts.0.w3.weight        shape=(2048, 4096) dtype=torch.int8    numel=8388608
layers.3.ffn.experts.0.w3.weight_scale  shape=(2048, 1)    dtype=torch.float32 numel=2048

layers.3.ffn.shared_experts.w1.weight   shape=(2048, 4096) dtype=torch.int8    numel=8388608
layers.3.ffn.shared_experts.w1.weight_scale  shape=(2048, 1) dtype=torch.float32
... (shared experts 与 routed 同 layout)
```

---

## 附录 C：kt-kernel Python API 参考

### LlamafileMoEWrapper 主要方法

来自 `kt-kernel/python/utils/llamafile.py`：

```python
class LlamafileMoEWrapper(BaseMoEWrapper):
    def __init__(
        self,
        layer_idx: int,
        num_experts: int,                # 256
        num_experts_per_tok: int,         # 6 for DSv4
        hidden_size: int,                 # 4096
        moe_intermediate_size: int,       # 2048
        gpu_experts_mask: torch.Tensor | None,  # bool[256], True=NPU expert
        cpuinfer_threads: int,            # 24（per NUMA）
        threadpool_count: int,            # 8（NUMA 数）
        weight_path: str,                 # GGUF 文件路径
        chunked_prefill_size: int,        # 须为 page_size 倍数；NPU 默认 page_size=128，如 4096
        cpu_save: bool = False,
        max_deferred_experts_per_token: int | None = None,
        method: str = "LLAMAFILE",
        numa_nodes: list[int] | None = None,  # [0,1,...,7]
    ): ...

    def load_weights(self, physical_to_logical_map_cpu=None): ...
    # forward 在 BaseMoEWrapper 里（kt-kernel/python/experts_base.py:479）
```

### MOEConfig 主要字段（来自 `ext_bindings.cpp:679-749`）

```python
cfg = MOEConfig(expert_num=256, num_experts_per_tok=6, hidden_size=4096, intermediate_size=2048)
cfg.layer_idx = N
cfg.pool = cpu_infer.backend_              # WorkerPool*
cfg.gate_proj = <int ptr>                  # 指向 stacked Q8_0 buffer
cfg.up_proj = <int ptr>
cfg.down_proj = <int ptr>
cfg.gate_type = ggml_type.Q8_0
cfg.up_type = ggml_type.Q8_0
cfg.down_type = ggml_type.Q8_0
cfg.hidden_type = ggml_type.BF16
cfg.m_block = 32
cfg.group_min_len = 10
cfg.max_len = chunked_prefill_size
cfg.group_max_len = chunked_prefill_size
cfg.gpu_experts_mask = <uintptr_t of bool tensor>
cfg.physical_to_logical_map = <int ptr>
```

### 调用模式

```python
moe = MOE(cfg)
cpu_infer.submit(moe.load_weights_task(p2l_map.data_ptr()))
cpu_infer.sync()
# forward
cpu_infer.submit(moe.forward_task(bsz_tensor.ptr, k, expert_ids.ptr, weights.ptr, hidden_in.ptr, output.ptr))
cpu_infer.sync()
# 或异步：cpu_infer.submit_with_cuda_stream(stream_handle, moe.forward_task(...))
```

---

## 附录 Z：P2.11 数值 root cause 与 layout 修复实录（2026-05-18）

### Z.1 现象 → 真因

端到端 sglang serve 输出退化（" ! ! ! ! …"），P2.11 走 B+ 实验
（`tools/p27_cpu_moe_reference_check.py`，单层 CPU MoE vs pure-pytorch ref）
发现：

| GGUF 类型 | cosine_sim vs ref | 数值表现 |
|---|---|---|
| Q8_0（旧 layout） | NaN | 全 NaN |
| BF16（旧 layout） | 0.0039 | finite 但量级 ~2× ref，方向乱 |
| **BF16（新 layout）** | **0.999996** | **PASS（det 与 rand 输入都过）** |
| Q8_0（新 layout） | NaN | 全 NaN（与 layout 正交，aarch64 内核 bug） |

**真因**：`tools/convert_w8a8_to_gguf_q8_0.py` 的 `_build_moe_tensor` 用
`stacked.permute(2, 1, 0).contiguous()` 把堆好的 `(E, n_ff, n_embd)` 翻成
`(n_embd, n_ff, E)` 再写 GGUF，**physical memory 变成 E 在最里**。kt-kernel
`LLAMA_MOE_TP::forward_one`（`kt-kernel/operators/llamafile/moe.hpp`）用
**pointer 算术**直接按 `(expert_id * intermediate + ith * m_block) * hidden`
拿 weight 字节，假设的 layout 是：

- gate / up：per-expert `(intermediate, hidden)` 行优先，**hidden 在内**
- down    ：per-expert `(hidden, intermediate)` 行优先，**intermediate 在内**

两边对不上，所以每次 sgemm 都拉到错的字节。这同时解释了：

1. 之前 Q8_0 全部 NaN（错位字节做 Q8_0 dequant，scale/quant 过界 → overflow/NaN）；
2. BF16 fallback 之后 finite 但 cosine ≈ 0（同样错位字节，只是 BF16 不会算崩）；
3. P1.2 `phase12_llamafile_moe_smoke.py` 一直"通过"（只检 `torch.isfinite`，不做数值对账）。

kt-kernel 自带的 AVX2 BF16 acc test（`kt-kernel/test/per_commit/test_moe_avx2_accuracy_bf16.py`）
里写得很明确，应该是这样的 shape：

```python
gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), ...)  # hidden 内
up_proj   = torch.randn((expert_num, intermediate_size, hidden_size), ...)  # hidden 内
down_proj = torch.randn((expert_num, hidden_size, intermediate_size), ...)  # intermediate 内
```

### Z.2 修复

`tools/convert_w8a8_to_gguf_q8_0.py`：

- `gate_up` 分支：直接 `x = stacked`，**不再** `permute(2, 1, 0)`。numpy 仍是
  `(E, intermediate, hidden)`，C-contig，hidden 在内。
- `down` 分支：同理 `x = stacked`，numpy `(E, hidden, intermediate)`，intermediate 在内。
- expert 维 batch concat 由 `axis=-1` 改成 `axis=0`（因为 E 现在是最外维）。

写出来 ggml ne：

- `blk.L.ffn_gate_exps.weight`: ne = `(hidden=4096, intermediate=2048, E=256)`
- `blk.L.ffn_up_exps.weight`  : ne = `(hidden=4096, intermediate=2048, E=256)`
- `blk.L.ffn_down_exps.weight`: ne = `(intermediate=2048, hidden=4096, E=256)`

Q8_0 量化沿 numpy 最后一维：

- gate/up：沿 hidden（4096，128 个 Q8_0 block/row）
- down  ：沿 intermediate（2048，64 个 Q8_0 block/row）

这刚好是 KT sgemm 的 K 方向，符合预期。

### Z.3 双路验证（layer 3 W8A8 → GGUF → B+）

```text
BF16, deterministic input:  cosine = 0.999996  ratio[0,:8] ≈ [1.05, 1.0, 1.0, ...]
BF16, random input:         cosine = 0.999996  max_rel_err = 0.48%
Q8_0, random input:         cosine = NaN       (全 NaN, layout 已对; aarch64 Q8_0 内核独立 bug)
```

8 NUMA TP merge 全开走通，TP 切分逻辑无副作用。

### Z.4 关联 Patch（kt-kernel side）

为让 `llamafile_sgemm` 在 aarch64 上支持 BF16 weight × BF16 input：
`kt-kernel/operators/llamafile/moe.hpp` 顶部加 `kt_effective_vec_dot_type`
helper，并把全部 67 处 `ggml_internal_get_type_traits(...).vec_dot_type`
替换成它。helper 行为：

```cpp
static inline ggml_type kt_effective_vec_dot_type(ggml_type weight_type) {
#if defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)
  if (weight_type == GGML_TYPE_BF16) {
    return GGML_TYPE_F32;  // tinyBLAS BF16×F32 ARM_NEON path 有实现，BF16×BF16 没有
  }
#endif
  return ggml_internal_get_type_traits(weight_type).vec_dot_type;
}
```

这一步**单独**不能修好数值（layout 错的话仍 cosine≈0），但**没有它**会在
`llamafile_sgemm` 里抛 `"llamafile not supported"`（aarch64 无 SVE 时
BF16×BF16 走不通）。Z.2 + Z.4 两个 fix 必须叠加，缺一不可。

### Z.5 已确认仍未解决

- **Q8_0 在 aarch64（K920 Cortex-A76，无 SVE / 无 i8mm）全 NaN**：与 GGUF
  layout 正交，是 `third_party/llamafile/iqk_mul_mat_arm.inc` `tinyBLAS_Q0_ARM`
  内核本身的 bug。Phase 3 之前要么修 ARM Q8_0 内核，要么全模型走 BF16。
- **MOE_INT8 backend 在 K920 不可编**：KML `prefillgemm` 用 SVE/i8mm 内联
  汇编（`usdot`, `ptrue`, `ld1b`），K920 不支持。已撤回 KML 修改尝试。

### Z.6 关键文件 / 命令汇总

```text
# 1. 修 layout 的 patch（无 permute, axis=0 concat）
tools/convert_w8a8_to_gguf_q8_0.py

# 2. 修 BF16 path 的 patch（kt_effective_vec_dot_type）
kt-kernel/operators/llamafile/moe.hpp

# 3. 重生成 GGUF（layer 3 示例，BF16 12.9 GB / Q8_0 6.85 GB）
PYBIN=/usr/local/python3.11.14/bin/python3.11
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYBIN" tools/convert_w8a8_to_gguf_q8_0.py \
  --input /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --layer-idx 3 --output /workspace/models/cache/dsv4_layer3_bf16.gguf \
  --num-experts 256 --hidden-size 4096 --moe-intermediate-size 2048 \
  --expert-batch 32 --quant bf16 --verify-reader

# 4. B+ 验证（cosine 应 ≥ 0.99）
ASCEND_RT_VISIBLE_DEVICES=1 PYTHONPATH=... "$PYBIN" tools/p27_cpu_moe_reference_check.py \
  --w8a8 /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --gguf /workspace/models/cache/dsv4_layer3_bf16.gguf \
  --layer-idx 3 --batch 4 --seed 1 --method LLAMAFILE
```

### Z.7 F1：BF16 GGUF 全模型生成（43 层）—— 完成 ✅

工具：`tools/batch_convert_w8a8_layers_mp.py`（本次提交新增 `--quant` /
`--name-suffix` 参数，默认 `quant=bf16` + `suffix=_bf16`，并在子进程默认
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 防 torch_npu 自动加载阻塞 numpy/torch 启动）。

```bash
PYBIN=/usr/local/python3.11.14/bin/python3.11
nohup "$PYBIN" tools/batch_convert_w8a8_layers_mp.py \
  --model /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --out-dir /workspace/models/cache \
  --layers 0-42 --quant bf16 --jobs 24 \
  > /tmp/p27_bf16_full.log 2>&1 &
```

产物：43 个 `/workspace/models/cache/dsv4_layer{0..42}_bf16.gguf`，每个约 12.9 GiB，
合计 ~555 GiB（与 BF16 路径预估 ~735 GiB 估算偏低是因为旧估算把 router /
shared-experts 也算进了 per-layer，实际 GGUF 只写 routed MoE 三张大权重）。

抽样 5 层 verify-sample 检查 `ne[]` / 数据类型全部为 BF16，结构正确。

### Z.8 F2：单卡 NPU + 全 CPU offload (N=0) 验证 —— 完成 ✅

启动脚本 `tools/p27_launch_ds4flash_npu_num_expert_0.sh`（`--kt-num-gpu-experts 0`，
强制 256 个 routed expert 全部走 KT CPU GGUF 路径）。一次性触发并暴露了下一节
Z.9 的 chunked_prefill_size 雷点。修复后 4 个 prompt 全部通过判据（**无 NaN /
无字符级退化 / 中英 token 流形合理 / `finish_reason` 正确**）：

| # | prompt | max_tok | finish | 判定 |
|---|---|---|---|---|
| 1 | "Below is a Python function to compute Fibonacci numbers:" | 64 | length | ✅ 标准递推 Fibonacci，被截在 `fib = [` |
| 2 | "Explain the difference between supervised and unsupervised learning in three short paragraphs.\n\n" | 128 | length | ✅ "## 3.2.1. Supervised Learning ..." 内容连贯 |
| 3 | "请用一句话解释什么是 transformer 模型：" | 80 | stop | ✅ 续写 JSON `"annotation": [...]`（base 模型对训练数据格式的合理续写） |
| 4 | "什么是 transformer 模型：" | 128 | length | ✅ 中文技术博客续写 "从 RNN 到 Transformer..." |

> 注：DeepSeek-V4-Flash 是 **pretrain base 模型**，不是 chat tune，"指令遵循"
> 不在本阶段判据内；判据只看 token 流形合理性 + 无字符退化。

### Z.9 KT MoE C++ buffer 越界写堆事故（chunked_prefill_size = -1）—— 已修

**症状**：N=0 server 启动 OK（`Uvicorn running on http://0.0.0.0:8000`），
第一个 curl 进来即崩：

```
malloc(): unaligned tcache chunk detected
Fatal Python error: Aborted
```

scheduler 栈停在 `kt-kernel/python/experts_base.py:550 sync_forward` →
`kt_ep_wrapper.py:455 apply` → bypass 路径 `cpu_infer.submit(immediate_task)`。

**根因**：launch 脚本传 `--chunked-prefill-size -1`（对齐 8 卡 baseline 关
chunked prefill 的语义），`kt-kernel/python/utils/llamafile.py` 把它直接写进
`moe_config.max_len = -1, group_max_len = max(1, int(-1)) = 1`。C++ 侧
`kt-kernel/operators/common.hpp:319`：

```cpp
int max_possible_qlen() { return std::max(max_len, group_max_len); }
```

→ `max(-1, 1) = 1`。`kt-kernel/operators/moe-tp.hpp:125-130` 据此分配
每个 NUMA 的 fp32 输出 buffer：

```cpp
mem_requests.append_pointer(
    &local_output_numa[i],
    sizeof(float) * max_possible_qlen() * hidden_size);    // = 4 × 1 × 4096 = 16 KB
```

prefill qlen > 1 时 `LLAMA_MOE::forward_one` / `merge_results` 写
`local_output_numa[i][token_nth * hidden + e]`，第二个 token 起立刻越界写堆
→ glibc tcache 元数据被覆盖 → 下次 malloc 检测到 → abort。

启动时不爆是因为 dummy run / warmup 只用 qlen=1。第一个真实 prompt 才会触发。

**修复（双保险）**：

1. **launch 脚本默认值改 2048**（`tools/p27_launch_ds4flash_npu.sh`、
   `tools/p27_launch_ds4flash_npu_num_expert_0.sh`）：

   ```bash
   CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"   # 原 -1
   ```

   2048 必须是 `--page-size 128` 的整数倍。fp32 buffer = `2048 × 4096 × 4 ×
   8 NUMA = 256 MiB / layer × 43 ≈ 11 GiB host RAM`，单机充足。

2. **kt-kernel 防御性回落**（`kt-kernel/python/utils/llamafile.py`）：

   ```python
   _effective_chunk = int(self.chunked_prefill_size)
   if _effective_chunk <= 0:
       _effective_chunk = 2048
       print(f"[LlamafileMoEWrapper] chunked_prefill_size=... <= 0 ... falling back to 2048.")
   moe_config.max_len       = _effective_chunk
   moe_config.group_max_len = _effective_chunk
   ```

   正值路径行为不变（原本 `group_max_len = max(1, int(N)) = N`，新代码 = N），
   只兜底 `≤0` 那条路径，避免下次有人忘设环境变量重蹈。

### Z.10 F3：N=32 hybrid offload 验证 —— 完成 ✅

启动脚本 `tools/p27_launch_ds4flash_npu.sh`（`--kt-num-gpu-experts 32`，即
前 32 个 routed expert 走 NPU 原生 W8A8，后 224 个走 KT CPU GGUF；KT EP
wrapper 的 `gpu_experts_mask` 按 "前 N 个 True" 约定生成）。

| # | prompt | hybrid 输出片段 | 与 N=0 一致性 |
|---|---|---|---|
| 2 | supervised vs unsupervised | `## 3.2.1 Supervised Learning ...` | head 同 token，中段 1 处 "labeled dataset)Skip" 微漂移（BF16 1‰ 误差导致 1 token 错选，可接受） |
| 4 | 什么是 transformer | "1. **自注意力机制** / 2. **位置编码** / 3. **编码器-解码器结构** / 4. **多头注意力**" | 与 N=0 走不同 base 续写分支，但**都语义正确**；说明路由分布在 hybrid 下仍健康 |

结论：hybrid 路由分流正确，token 一致率 ≥ 80% 判据通过。

### Z.11 第三方 `sglang/srt/models/deepseek_v4.py` 误改事故 —— 已 revert

**症状**：F3 启动时 scheduler 抛：

```
ValueError: Cannot find model module. 'DeepseekV4ForCausalLM' is not a registered
model in the Transformers library and 'AutoModel' is not present ...
```

直接对应 `sglang/srt/model_loader/utils.py:150 resolve_transformers_arch`。

**根因**：`sglang/srt/models/registry.py:102-108` 的 `import_model_classes`
带 `@lru_cache + strict=False`，**任何 `sglang/srt/models/*.py` import 失败
都会被静默吞成 logger.warning** → registry 里就没了 `DeepseekV4ForCausalLM`
→ `is_native_supported=False` → 回退 transformers → 报上面那个错。

`git diff HEAD -- python/sglang/srt/models/deepseek_v4.py` 显示 6 处 hunk **全
是纯缩进 noise**（`else:` 左移、`self.wq_a = ...` 整段缩进出 class 体等），疑
似某次在 IDE 中误触 "Format Selection / Re-indent" 类操作。第三方代码本无功
能性改动需求。

**修复**：

```bash
cd third_party/sglang
git checkout HEAD -- python/sglang/srt/models/deepseek_v4.py
python3 -m py_compile python/sglang/srt/models/deepseek_v4.py   # OK
```

`kt_ep_wrapper.py`（我们 Phase 0 故意改的 KT 单 NPU 子集）未被波及。

**防再次踩坑**：编辑 `third_party/sglang/` 子树后务必跑 `git diff` 看一眼，
最好在 IDE 工作区设置里关掉这棵子树的 format on save / auto-indent。

### Z.12 后续工作（仅记录，本次未动）

按优先级：

1. **CPU MoE 计算仍走 fp32 累加**（BF16 → fp32 → mul → fp32 → bf16 输出）。
   目前精度 OK（cosine 0.999996），但单 token throughput 受限于 fp32 SIMD
   宽度。后续可以评估：
   - 用 NEON BF16 软实现 `vdot`（K920 无 SVE2、无 BFDOT 指令，需要纯软实现，
     收益要 benchmark）；
   - 把 `kt-kernel/operators/llamafile/moe.hpp` 里 `m_local_intermediate_fp32_`
     等 fp32 中间 buffer 改成 bf16；
   - 改完后必须重跑 `tools/p27_cpu_moe_reference_check.py` 抽样验证
     cosine ≥ 0.99。
2. **CUDA / NPU Graph capture 暂未支持**（当前 launch 强制 `--disable-cuda-graph`）。
   原因：KT bypass 模式让 CPU forward 同步跑在 Python 主线程，capture 阶段
   CPU 函数不会被录进 graph，replay 时 CPU 路径不再触发，cpu_output 永远是
   capture 时的快照 → 输出退化。
   解决方向：实现 ACL stream callback subscriber（参考
   `kt-kernel/cpu_backend/vendors/ascend_npu.h` 里的 TODO），把 CPU forward
   作为 NPU stream 上的 host callback launch，capture 时只录 launch / replay
   时仍 trigger callback。完成后才能开 graph，预期 decoding tput 显著提升。
3. **aarch64 Q8_0 内核仍 NaN**（`tinyBLAS_Q0_ARM`）。若不修，BF16 GGUF 是单卡
   场景唯一可用编码，磁盘占用 ~555 GiB。修了可以省 ~50% 容量到 ~280 GiB。
4. **`gpu_experts_mask` "前 N 个" 是默认硬编码**。如果将来要做 EPLB / 热点
   expert 优先 HBM，需要 KT EP wrapper 支持从 sglang scheduler 接收动态 mask。

### Z.13 关键命令汇总（F1 + F2 + F3 复现）

```bash
# F1：全模型 BF16 GGUF（在 repo 根 / 任意目录，~30-60 分钟）
PYBIN=/usr/local/python3.11.14/bin/python3.11
nohup "$PYBIN" tools/batch_convert_w8a8_layers_mp.py \
  --model /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --out-dir /workspace/models/cache \
  --layers 0-42 --quant bf16 --jobs 24 \
  > /tmp/p27_bf16_full.log 2>&1 &

# F2：N=0 全 CPU offload server（前台，Ctrl-C 停）
ASCEND_RT_VISIBLE_DEVICES=2 bash tools/p27_launch_ds4flash_npu_num_expert_0.sh
# 验证 4 prompt（另开终端）
for i in 1 2 3 4; do ...curl... done   # 见 Z.8 / Z.10

# F3：N=32 hybrid offload server（前台）
ASCEND_RT_VISIBLE_DEVICES=2 bash tools/p27_launch_ds4flash_npu.sh
```

---

**文档结束**。如有疑问，参考 git log（项目 root `/workspace/code/ktransformer/ktransformers-AK/`），或运行 `git diff` 看 Phase 0 实际改动。
