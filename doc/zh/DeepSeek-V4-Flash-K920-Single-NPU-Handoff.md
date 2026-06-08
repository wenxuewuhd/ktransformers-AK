> ⚠️ **历史文档(2026-05-12,规划期)**。现行总纲见 [doc/zh/dsv4_single_npu/DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md](dsv4_single_npu/DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md)。
> Phase 0~2 已完成、Phase 4(KML)现已"不做";部分量化结论已被取代,见总纲 §3。保留作历史/Phase 计划记录。

# DeepSeek-V4-Flash · 单卡 Ascend NPU + Kunpeng K920 CPU offload 接手文档

> 给后续 Claude Code / 协作者的完整状态快照。
>
> 时间：2026-05，仓库 commit `d7b5b49` (`0.6.2.post1`) + 一组本地未提交改动。
>
> 阅读顺序建议：先读 §1 §2 §3 建立目标和环境感，再看 §4 已完成工作，然后按 §5 路线图往下做。§6 是踩过的坑必看，§7 是关键文件位置，§8 是给 Claude Code 第一次接手时的"先读这些"清单。

---

## 1. 项目目标

把 DeepSeek-V4-Flash (DSv4-Flash, 671B-class, MoE) 跑在单张 Ascend 910B / 800T（32GB 或 64GB HBM，本机是 64GB）+ 单台 Kunpeng-920 CPU 服务器上，使用 `kt-kernel` 把绝大多数 MoE expert 放到 CPU（DRAM）上算，仅保留少量 hot expert + 主干在 NPU 上算。

参考点：

- 现有的 8 卡 SGLang 拉起脚本（容器化）：`sglang_dsv4_ascend_cann850.sh`（用户原话叫 `launch_ds4flash_sglang.sh`）。
- ktransformers 在 GPU 上已经能跑 DSv4-Flash（参见 `doc/en/DeepSeek-V4-Flash.md`）。
- 老版本 Ascend 单卡 R1 教程：`doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md`（思路对，细节过时）。
- Qwen3-MoE Ascend 教程：`doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md`（注入点 + Python wrapper 用法参考）。

最终交付（按用户要求顺序）：

1. PoC：standalone Python 脚本，跑一层（或几层）MoE，验证数值正确。
2. 整网单卡跑通：通过自定义 SGLang Adapter / Server Args 把 MoE 路由到 `kt-kernel`，整 43 层可以推理。
3. 性能优化：CPU/NPU 异步 overlap（基于 `aclrtLaunchCallback`）。
4. 精度回归：W8A8 -> KML `cblas_gemm_s8s8s32` 直算（替代 GGUF Q8_0 损失精度）。

---

## 2. 硬件 & 软件环境

### 2.1 NPU 侧

- 8x Ascend 910B1，每卡 64GB HBM。当前只用 1 张。
- CANN：`8.5.0`，路径 `/usr/local/Ascend/ascend-toolkit/latest`。
- `aclrtLaunchCallback` 已经在 header 里：`/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h:975`。
- `torch_npu`：支持 `pin_memory=True`、`torch.npu.Stream`、`torch.npu.Event`。
- 部署形态：容器内运行 SGLang。`sglang_dsv4_ascend_cann850.sh` 是宿主机启动脚本，挂载模型和代码进容器。

### 2.2 CPU 侧（关键）

- Kunpeng-920 5250，`/proc/cpuinfo` 关键 flags：`fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma dcpop asimddp asimdfhm`。
- 有：NEON、`asimddp` (SDOT, ARMv8.2 dot product)、`fphp`/`asimdhp` (FP16)。
- 没有：`sve`、`bf16`、`i8mm`、SME。-> 这条决定了 KML 的 `Int8_KERNEL_MOE` 路径不能直接用（详见 §6.2）。
- 拓扑：4 sockets x 48 cores = 192 物理核，1 thread/core，8 个 NUMA node。
- DRAM：1.5 TiB；目前空闲 ~1.4 TiB（DSv4 W8A8 总权重 ~600GB，放得下）。

### 2.3 重要库版本

| 库 | 版本 / 路径 |
| - | - |
| CANN | `8.5.0` @ `/usr/local/Ascend/ascend-toolkit/latest` |
| KML | `2.5.0` @ `/usr/local/kml`（Phase 4 用，Phase 0/1/2/3 不要链） |
| Python | `/usr/local/python3.11.14/bin/python3` (容器外宿主机) |
| torch_npu | 容器内已安装并能 pin host memory |
| hwloc | `libhwloc-dev` 已通过 `--allow-unauthenticated` 装上 |
| numactl | 已 `apt install numactl`（容器内/外都需要） |

### 2.4 模型权重

- 路径：`/workspace/models/DeepSeek-V4-Flash-W8A8`。
- 关键参数（`config.json`，已实际读过）：
  - `hidden_size = 4096`
  - `moe_intermediate_size = 2048`
  - `n_routed_experts = 256`
  - `num_experts_per_tok = 6`（topk=6，注意不是 8）
  - `n_shared_experts = 1`
  - `num_hidden_layers = 43`
  - `first_k_dense_replace`（前几层稠密）以 `config.json` 实际值为准
- 权重命名（`model.safetensors.index.json` 实测）：
  - `model.layers.N.mlp.experts.E.gate_proj.weight` / `.weight_scale`
  - `model.layers.N.mlp.experts.E.up_proj.weight` / `.weight_scale`
  - `model.layers.N.mlp.experts.E.down_proj.weight` / `.weight_scale`
  - 别名约定（在转换器里要识别）：`w1=gate_proj`、`w3=up_proj`、`w2=down_proj`。
- 量化格式（用 safetensors header + 实读 shard 确认过）：
  - `gate_proj.weight`：shape `(2048, 4096)`，dtype `int8`。
  - `gate_proj.weight_scale`：shape `(2048, 1)`，dtype `fp32`，per-output-channel。
  - `up_proj` 同 `gate_proj` 形状。
  - `down_proj.weight`：shape `(4096, 2048)`，dtype `int8`；`down_proj.weight_scale` shape `(4096, 1)` fp32。
  - 没有 per-group fp16 scales、没有 zero-point（对称量化）。

### 2.5 SGLang 已知限制

仓库内 `sglang/python/sglang/srt/server_args.py` 在 Ascend 后端会显式 `assert` 拒绝所有 `--kt-*` 参数（kt-kernel 集成路径只走 GPU 后端）。-> Phase 2 必须自己开 patch，详见 §5.3。

---

## 3. 关键技术决策（含理由）

| 决策 | 选项 | 选了 | 理由 |
| - | - | - | - |
| GPU 一侧端到端框架 | SGLang fork / 自己写一层 | SGLang fork | 已经能跑 8 卡，主干、attention、router 全部走 sglang 现成 NPU 算子，只补 MoE 路由 |
| CPU expert 权重格式（Phase 1/2/3） | W8A8 直算 / GGUF Q8_0 / FP16 | GGUF Q8_0 + LLAMAFILE backend | `kt-kernel` 默认 ARM backend，K920 上能 SDOT 加速；KML 的 W8A8 路径有 SVE 汇编，K920 没 SVE |
| CPU expert 权重格式（Phase 4） | 继续 Q8_0 / KML W8A8 直算 | KML `cblas_gemm_s8s8s32` | 避免 W8A8->Q8_0 的精度损失；KML 2.5.0 提供该 CBLAS 接口，K920 上有 NEON int8 实现（不依赖 SVE） |
| MoE 切分 | 全 CPU / 全 NPU / hybrid | hybrid：~16/256 hot expert 上 NPU，其余 240 个 CPU | 64GB HBM，主干 + KVcache + 16 hot expert + activation buffer 估算 ~50-55GB |
| Speculative decoding (NEXTN) | 启 / 关 | 第一版关 | 减少初版复杂度，SGLang NPU NEXTN 路径也有坑 |
| ARM `-march` | `armv8.2-a+fp16+dotprod` / 加 `+sve+bf16+i8mm` | `armv8.2-a+fp16+dotprod` | K920 实际能力 = 这个集合，多一个就编不出或运行非法指令 |
| 异步 overlap API | `aclrtLaunchCallback` / 手动线程池 | `aclrtLaunchCallback` | 与 CUDA `cudaLaunchHostFunc` 行为最贴近，kt-kernel 现有抽象只需做语义对齐 |
| Phase 1 是否做随机权重 PoC (P1.0) | 做 / 跳 | 跳 P1.0，直接 P1.1 | 用户明确选择 |

---

## 4. 已完成工作（Phase 0：编译期适配）

所有 Phase 0 的目标：让 `kt-kernel` 在 aarch64 + CANN 的环境里能编出来、能 import、关键 binding 能调用。

### 4.1 改动文件清单（未 commit）

```text
M  kt-kernel/CMakeLists.txt
M  kt-kernel/cpu_backend/cpuinfer.h
M  kt-kernel/cpu_backend/vendors/vendor.h
M  kt-kernel/install.sh
M  kt-kernel/setup.py
M  third_party/llamafile/iqk_mul_mat_arm82.cpp
?? kt-kernel/cpu_backend/vendors/ascend_npu.h          (新)
?? script/                                              (新, 后续 PoC 脚本放这)
?? sglang_dsv4_ascend_cann850.sh                       (容器启动脚本)
?? third_party/kml/                                    (KML 源拷贝, 暂不用)
```

另外通过 `git checkout 53f6a6d^` 恢复了 KML kernel 源（在 `kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel/` 下，大量 A），Phase 4 才参与编译。

### 4.2 各文件改动要点

#### `kt-kernel/CMakeLists.txt`

1. 加 `option(KTRANSFORMERS_USE_ASCEND_NPU "..." OFF)`。
2. 加 `LLAMA_ARM_DOTPROD` / `LLAMA_ARM_FP16` / `LLAMA_ARM_SVE` / `LLAMA_ARM_BF16` / `LLAMA_ARM_I8MM` 五个细粒度开关。
3. 修掉了原来在 L205 的硬编码 `-march=armv8.6-a`。新逻辑：

```
_kt_arm_arch = "armv8.2-a"
if LLAMA_ARM_FP16: append "+fp16"
if LLAMA_ARM_DOTPROD: append "+dotprod"
if LLAMA_ARM_BF16: append "+bf16"
if LLAMA_ARM_I8MM: append "+i8mm"
if LLAMA_ARM_SVE: 单独加 "+sve" 并提示
ARCH_FLAGS = "-march=${_kt_arm_arch}"
```

4. 加 `elseif(KTRANSFORMERS_USE_ASCEND_NPU)` 分支：
   - `find_path(ACL_INCLUDE_DIR acl/acl_rt.h HINTS $ENV{ASCEND_TOOLKIT_HOME}/include /usr/local/Ascend/.../include)`。
   - `find_library(ASCEND_CL_LIBRARY ascendcl HINTS .../lib64)`。
   - `add_definitions(-DKTRANSFORMERS_USE_ASCEND_NPU)`。
   - `target_link_libraries` 把 `${ASCEND_CL_LIBRARY}` 加进去。

#### `kt-kernel/cpu_backend/vendors/vendor.h`

加 `#elif defined(USE_ASCEND_NPU)` 分支，include `ascend_npu.h`。这是 kt-kernel 内部抽象，注意 macro 名是 `USE_ASCEND_NPU` 而不是 `KTRANSFORMERS_USE_ASCEND_NPU`，前者在源码 `#if/#elif` 内用，后者是 cmake `-D` 透传时用。

#### `kt-kernel/cpu_backend/vendors/ascend_npu.h`（新文件）

提供 CUDA 兼容 wrapper：

```cpp
#pragma once
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <cstdint>

using cudaStream_t = aclrtStream;
using cudaError_t  = aclError;
using cudaHostFn_t = aclrtCallback;
inline constexpr cudaError_t cudaSuccess = ACL_SUCCESS;

static inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream,
                                             cudaHostFn_t fn,
                                             void* userData) {
    // ACL 语义: 调用方需要在某个宿主线程上调 aclrtSubscribeReport +
    // aclrtProcessReport, callback 才会真的被触发. kt-kernel 后续在
    // CPUInfer 的 worker 线程里做这件事.
    return aclrtLaunchCallback(fn, userData, ACL_CALLBACK_NO_BLOCK, stream);
}

static inline const char* cudaGetErrorString(cudaError_t /*err*/) {
    const char* m = aclGetRecentErrMsg();
    return (m && *m) ? m : "ACL error";
}
```

#### `kt-kernel/cpu_backend/cpuinfer.h`

把原来 `#ifdef KTRANSFORMERS_USE_CUDA` 全部改成 `#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_ASCEND_NPU)`，覆盖 `submit_with_cuda_stream` 和 `sync_with_cuda_stream` 两个方法 + include 头文件分支。

#### `kt-kernel/setup.py`

1. 加 `CPUINFER_USE_ASCEND_NPU` 环境变量识别（透传成 `-DKTRANSFORMERS_USE_ASCEND_NPU=ON`）。
2. 加 `detect_cann_toolkit()`：先看 `$ASCEND_TOOLKIT_HOME`，再看 `/usr/local/Ascend/ascend-toolkit/latest`。
3. aarch64 + 找到 CANN + 没启 CUDA -> 默认自动开 NPU。
4. 加 `CPUINFER_ARM_DOTPROD` / `_FP16` / `_SVE` / `_BF16` / `_I8MM`，全部透传到 cmake。
5. aarch64 + NPU 模式下，默认 force off `CPUINFER_ENABLE_KML` 和 `CPUINFER_ENABLE_BLIS`（KML SVE 路径 K920 不能用，BLIS 用不到）。

#### `kt-kernel/install.sh`

1. 加 `detect_arm_features()`：grep `/proc/cpuinfo` 设 `KT_ARM_HAS_DOTPROD` / `_FP16` / `_SVE` / `_BF16` / `_I8MM`。
2. 加 `detect_cann_root()`：找 `$ASCEND_TOOLKIT_HOME` -> `/usr/local/Ascend/ascend-toolkit/latest`。
3. `build_step()` 增加 aarch64 分支：
   - 把上面探测到的 flag 透传成 `CPUINFER_ARM_*` 环境变量。
   - 找到 CANN 就 `export CPUINFER_USE_ASCEND_NPU=1` + `ASCEND_TOOLKIT_HOME=$cann_root`。
   - K920 上强行 `CPUINFER_ENABLE_KML=OFF`、`CPUINFER_ENABLE_BLIS=OFF`。
   - 跳过 x86 AMX/AVX512 检测。

#### `third_party/llamafile/iqk_mul_mat_arm82.cpp`

上游小 bug：里头本来就有一对 `#define iqk_mul_mat iqk_mul_mat_arm82` / `#define iqk_mul_mat_moe iqk_mul_mat_moe_arm82`，但是被注释掉了。结果 `iqk_mul_mat_arm.inc` 被 include 时发出来的符号没有 `_arm82` 后缀，而 `sgemm.cpp` 里调用又显式找 `_arm82` 后缀的 -> 链接通过、运行时 `undefined symbol: iqk_mul_mat_moe_arm82`。

修复：把这两行 `#define` 的注释去掉。

### 4.3 环境侧操作

1. `apt install -y --allow-unauthenticated -o Acquire::AllowInsecureRepositories=true libhwloc-dev`（Huawei Cloud ports 源 GPG 签名错误，绕过装上的）。
2. `apt install -y numactl`。
3. `dpkg -i kml-2.5.0.aarch64.deb` 装 KML 到 `/usr/local/kml`。
4. 写 `/etc/ld.so.conf.d/kml.conf`：

```
/usr/local/kml/lib
/usr/local/kml/lib/neon/kblas/pthread
```

然后 `ldconfig`。

### 4.4 Phase 0 验收

冒烟脚本：`/tmp/kt_kernel_phase0_smoke.py`（临时文件，可重建）。

通过点：

- `import kt_kernel_ext` 不报 undefined symbol。
- `kt_kernel_ext.moe.MOE` / `MOEConfig` / `MOESFTConfig` 都存在。
- `kt_kernel_ext.CPUInfer` 能实例化，构造时打印 NUMA worker pool 信息。
- `kt_kernel_ext.moe.MOE` binding 包含 `submit_with_cuda_stream` 和 `sync_with_cuda_stream`（这俩在 Ascend 编译路径下也开了，因为走的就是 cudaStream 兼容层）。
- `ldd build/.../kt_kernel_ext*.so` 含 `libascendcl.so`、`libnuma.so.1`、`libhwloc.so.15`。

---

## 5. 路线图（接手后按顺序做）

### 5.1 Phase 1 · 单层数值正确性 PoC

> 已和用户确认：跳过 P1.0（随机权重），直接做 P1.1。

#### P1.1 · W8A8 -> GGUF Q8_0 离线转换器（下一步要做的）

要新建两个文件：

```
tools/_w8a8_dequant.py            # helper: int8 + fp32 per-channel scale -> fp16/fp32 dense
tools/convert_w8a8_to_gguf_q8_0.py
```

输入：`/workspace/models/DeepSeek-V4-Flash-W8A8` 整套 safetensors。
输出：每层每 expert 的 3 个矩阵（gate / up / down）按 GGUF Q8_0 格式写到磁盘。

关键技术点：

1. W8A8 是 per-output-channel fp32 scale 对称量化 -> GGUF Q8_0 是 per-32-element fp16 scale 对称量化。无法直接 reinterpret，必须先 dequantize 到 fp16/fp32，再按 GGUF block 重量化。
2. Dequantize 公式：`x = int8 * fp32_scale[out_channel]`。
3. Re-quantize 到 Q8_0：每 32 个元素一个 block，每 block = `fp16 d` + `int8 qs[32]`，公式：
   ```
   amax = max(|x_i|), d = amax / 127.0, qs[i] = round(x_i / d) -> int8
   ```
   block 总字节数 34。整张矩阵字节数 = `numel/32 * 34`。
4. 别名映射（写转换器时一定要兼容）：
   - `w1 <-> gate_proj`
   - `w3 <-> up_proj`
   - `w2 <-> down_proj`
5. 输出布局建议：
   ```
   <out>/layer_{L}/expert_{E}/{gate|up|down}.q8_0.bin
   ```
   或者直接打成单文件 GGUF（看 `kt_kernel.KTMoEWrapper` 当前的载入接口；如果接口要 raw block，写 raw 更省事）。
6. 第一版只转 1 个 layer 的 1 个 expert，跑通端到端，然后再批量化。

#### P1.2 · Hybrid forward demo

新建 `script/poc_dsv4_moe_p11_real_weights.py`：

1. 从 HF transformers 载入 DSv4-Flash config（不载权重，只要拓扑）。
2. 在 NPU 上手搓 1 层 MoE：
   - router (`gate.weight`) 直接从 safetensors 取，跑 topk=6。
   - 4 个 hot expert 放在 NPU（用 transformers 现成的 expert forward + bmm，或者直接 dequant 到 fp16 算）。
   - 12 个 cold expert 通过 `kt_kernel.KTMoEWrapper` 走 CPU 路径。
3. 对比"全 NPU 算"和"hybrid"两条路径的输出，要求 cosine sim > 0.999 / max diff < 5e-3。
4. 同时打印两条路径的耗时（这一步先不追性能，只要能跑）。

退出标准：

- 单层 hybrid 数值对齐。
- CPUInfer + KTMoEWrapper 拿到真实 GGUF Q8_0 权重能跑。
- 这一步过了再决定 Phase 2 启动。

### 5.2 Phase 2 · SGLang 整网集成（"整网单卡跑通"）

> 用户最关心的里程碑。Phase 1 完成 != 整网跑通，必须到这里。

子任务（顺序大致独立，可并行）：

- P2.1：在 `sglang/python/sglang/srt/server_args.py` 里有条件放开对 Ascend 后端的 `--kt-*` 参数 assert。新增 `--kt-cpu-experts`, `--kt-cpu-config`, `--kt-num-cpu-workers`, `--kt-amx-method=KML_INT8|LLAMAFILE_Q8_0` 等。要做 patch 而不是 fork，方便升级。
- P2.2：写 `sglang/python/sglang/srt/models/deepseek_v4_flash_ktmoe.py`（或在现有 `deepseek_v4_flash.py` 加分支）。注入点是 MoE block 的 expert forward：当 layer L、token's expert id ∈ cpu_set 时，把对应 token 的 hidden_state 通过 `KTMoEWrapper.submit()` 投到 CPU，主干在 NPU 上继续 router/topk，结果用 `KTMoEWrapper.sync()` 拿回。
- P2.3：完成 `kt_kernel/KTMoEWrapper`（Python 侧）的 NPU stream 适配。具体来说，它现在的 `submit_with_cuda_stream` 接的是 `torch.cuda.current_stream().cuda_stream`，要改成接 `torch.npu.current_stream().npu_stream` 并 cast 成 `aclrtStream`（int 透传到 C++ 即可，C++ 侧 cast 回去）。
- P2.4：在主 host 线程跑 `aclrtSubscribeReport` + 起 worker 线程跑 `aclrtProcessReport` 循环，给 `aclrtLaunchCallback` 兜底。这是 ACL 和 CUDA 的核心语义差异，Phase 0 已经做了 stub，Phase 2 必须在 CPUInfer 启动时真正起线程。
- P2.5：定义 hot/cold 静态切分文件（YAML 或 JSON），例如：

```yaml
cpu_offload:
  layer_3: [0,1,2,...,239]    # 240 个走 CPU
  layer_4: [...]
```

hot 的 16 个 expert 走 NPU。第一版先静态全局策略，不做动态。
- P2.6：权重加载。CPU 部分通过 `KTMoEWrapper.load_q8_0(path)`；NPU 部分通过现有 transformers safetensors 路径，但只挑 hot expert 的 shard 加载，cold expert 的 tensor 直接 skip（节省 ~580GB HBM 占用，让 KVcache 有空间）。
- P2.7：跑短 prompt 验证。先跑 1 token decode，要求 logits 和 reference 8 卡服务的输出 token 一致；再跑 32 token 续写做粗糙对齐。
- P2.8：把 `sglang_dsv4_ascend_cann850.sh` 拷一份成 `sglang_dsv4_ascend_cann850_singlecard.sh`，TP=1，加上 `--kt-*` 参数，关 NEXTN。

退出标准：

- 单卡能拉起 SGLang 服务接 HTTP 请求。
- 单 prompt 推理输出和 8 卡服务在前 32 token 上完全一致（或 logits MSE < 阈值）。
- HBM 占用稳定在 ~55GB 以内。

### 5.3 Phase 3 · 异步 overlap 性能优化

要点：

1. `aclrtLaunchCallback` 已经在 Phase 0/2 接通，但默认还是"NPU 提交 -> 等 CPU 算完回来 -> 继续"。
2. Phase 3 要做的：把 hidden_state 的 NPU->CPU 拷贝、CPU expert 计算、CPU->NPU 拷贝三段流水起来。具体技术：
   - decode 时 batch 内多个 token 的 cold-expert dispatch 一次性丢给 CPU。
   - 用 `torch.npu.Event` 做依赖追踪。
   - CPU 端 worker 用 NUMA-aware thread pool，已经在 kt-kernel 里有。
3. 期望 decode latency 不被 CPU 拖到 6 倍以上（hot:cold 比例 16:240 即 ~6.7%，理想就是 CPU 那部分完全被 overlap 掉）。

### 5.4 Phase 4 · KML W8A8 直算精度回归

1. 把 `kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel/` 里的 KML 源（已经 git checkout 出来的那批 A 文件）参与编译，但只编 CBLAS 接口路径，绕开 SVE 汇编。
2. 把 KML 包成新 backend `Int8_KERNEL_MOE_KML_CBLAS`，core kernel = `cblas_gemm_s8s8s32`。
3. 权重直接吃 W8A8 safetensors（不用 GGUF 转），scale 在 GEMM 后乘回去。
4. 对比 Phase 2 (Q8_0) 和 Phase 4 (W8A8) 的 logits / perplexity 差异。

退出标准：

- 精度比 Phase 2 (Q8_0) 更接近 GPU 参考。
- 性能不输 Q8_0 路径太多（如果输太多就不切，保持 Q8_0）。

---

## 6. 已踩过的坑（必看）

### 6.1 K920 没 SVE，KML 的 `Int8_KERNEL_MOE` 不能直接用

`kt-kernel/operators/moe_kernel/.../kml_kernel/prefillgemm/` 一堆 `.S` 汇编是 SVE 的。K920 是 ARMv8.2-A，没 SVE。直接编出来跑会 `SIGILL`。

对策：

- Phase 0/1/2/3：`CPUINFER_ENABLE_KML=OFF`，走 LLAMAFILE Q8_0。
- Phase 4：只走 KML 的 CBLAS API（`cblas_gemm_s8s8s32`），它有 NEON 版本。

### 6.2 SGLang Ascend 后端硬卡 `--kt-*`

`sglang/python/sglang/srt/server_args.py` 里有：

```python
if device == "npu":
    assert all kt-* args are default, f"kt-kernel is not supported on NPU"
```

Phase 2 必须 patch 掉这段，不然连参数都过不去。

### 6.3 `iqk_mul_mat_arm82.cpp` 上游 bug

两行 `#define` 被注释，结果 ARMv8.2 路径 emit 出来的符号没后缀，链接器找不到。已修。如果将来 rebase / 同步上游记得保留这个 patch。

### 6.4 Huawei Cloud ports apt 源 GPG 签名错误

容器内 `apt update` 报 403 + 签名失败。用：

```bash
apt-get install -y --allow-unauthenticated \
  -o Acquire::AllowInsecureRepositories=true \
  libhwloc-dev numactl
```

绕过即可。

### 6.5 W8A8 -> GGUF Q8_0 的量化粒度不一致

- W8A8：fp32 scale，per output channel（一个 row 一个 scale）。
- Q8_0：fp16 scale，per 32 elements（一个 32-元 block 一个 scale）。

不能直接 reinterpret，必须 dequant->requant。第一次写转换器时单 tensor 做完后跟 dequant 后的 fp16 矩阵 cosine sim 比较，要求 > 0.9995。

### 6.6 ACL `aclrtLaunchCallback` 不会"自动"被触发

和 CUDA `cudaLaunchHostFunc` 最大的区别：CUDA 是 driver 自动调起 callback；ACL 需要：

1. 在一个专门的 host 线程里调 `aclrtSubscribeReport(thread_id, stream)` 把这个线程绑到 stream。
2. 同一个线程里循环跑 `aclrtProcessReport(timeout)`，callback 才会真的触发。
3. 结束时 `aclrtUnSubscribeReport(thread_id, stream)`。

-> kt-kernel 的 `CPUInfer` 启动时必须额外起一个 report-poller 线程。Phase 0 里 stub 已经留好，Phase 2 必须实做。

### 6.7 `torch.npu.Stream` <-> `aclrtStream`

`torch.npu.current_stream().npu_stream` 取出来已经是 `aclrtStream` 等价物（一个指针/int）。在 pybind 边界统一传 `uintptr_t`，C++ 侧 cast 回 `aclrtStream`，不要 include `torch_npu` 头文件，避免 ABI 耦合。

### 6.8 `setup.py` 和 cmake 的 macro 名分裂

外部环境变量 `CPUINFER_USE_ASCEND_NPU` -> cmake `KTRANSFORMERS_USE_ASCEND_NPU` -> C++ 源 `#if defined(KTRANSFORMERS_USE_ASCEND_NPU)`。
但是 `vendors/vendor.h` 里用的是裸名 `USE_ASCEND_NPU`，是 cmake `add_definitions(-DUSE_ASCEND_NPU)` 透传的。两个名字都要保留，删任何一个都会断。

### 6.9 容器 vs 宿主机

`sglang_dsv4_ascend_cann850.sh` 在宿主机跑，会拉容器、挂模型、再 exec 进去。所有 `apt`、`pip`、`ldconfig`、`dpkg` 操作都要在容器里做（或在镜像里做完再保存）。宿主机层做了不进容器。

---

## 7. 关键文件位置

### 7.1 已改/新增

```
kt-kernel/CMakeLists.txt
kt-kernel/cpu_backend/cpuinfer.h
kt-kernel/cpu_backend/vendors/vendor.h
kt-kernel/cpu_backend/vendors/ascend_npu.h          <- 新
kt-kernel/install.sh
kt-kernel/setup.py
third_party/llamafile/iqk_mul_mat_arm82.cpp         <- 取消 #define 注释
```

### 7.2 Phase 1.1 要新建

```
tools/_w8a8_dequant.py
tools/convert_w8a8_to_gguf_q8_0.py
script/poc_dsv4_moe_p11_real_weights.py
```

### 7.3 Phase 2 要新建/改

```
sglang/python/sglang/srt/server_args.py                       <- 放开 kt-* assert
sglang/python/sglang/srt/models/deepseek_v4_flash_ktmoe.py    <- 或在现 deepseek_v4_flash.py 加分支
config/cpu_offload_dsv4_flash.yaml                            <- 静态切分配置
sglang_dsv4_ascend_cann850_singlecard.sh                      <- 拷自现有脚本
```

### 7.4 文档

```
doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md   <- 老 R1 单卡教程, 思路参考
doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md       <- Qwen3 MoE 注入点参考
doc/en/DeepSeek-V4-Flash.md                          <- GPU 上 V4 跑法
doc/zh/DeepSeek-V4-Flash-K920-Single-NPU-Handoff.md  <- 本文件
```

### 7.5 模型 / 权重

```
/workspace/models/DeepSeek-V4-Flash-W8A8/
  config.json
  tokenizer*.json
  model.safetensors.index.json
  model-*.safetensors                  (46 个 shard)
```

### 7.6 外部依赖

```
/usr/local/Ascend/ascend-toolkit/latest/             (CANN 8.5.0)
/usr/local/kml/                                       (KML 2.5.0, Phase 4 用)
/etc/ld.so.conf.d/kml.conf                           (已写)
```

---

## 8. Claude Code 第一次接手的"先读这些"

按顺序读：

1. 本文件 §1, §2, §3, §6（建立 mental model + 避坑）。
2. `kt-kernel/setup.py` 末段 + `kt-kernel/install.sh` 末段（看构建流程怎么自动探 NPU）。
3. `kt-kernel/cpu_backend/cpuinfer.h` 的 `submit_with_cuda_stream`（看 Phase 2 要怎么对接 NPU stream）。
4. `kt-kernel/python_bindings/`（找 `KTMoEWrapper` 类，确认 Phase 1.2 / 2 调用接口签名 —— 名字可能是别的，搜 `class.*MoE.*Wrapper` / `class.*MOE.*` in `kt-kernel/`）。
5. `doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md` 看 Ascend NPU 上注入 kt-kernel MoE 的"老路子"。
6. `sglang/python/sglang/srt/models/deepseek_v4_flash.py`（或对应文件）看 GPU 上 MoE block 的实现，决定 Phase 2 注入点。
7. `/workspace/models/DeepSeek-V4-Flash-W8A8/config.json`（确认参数对得上 §2.4）。
8. `sglang_dsv4_ascend_cann850.sh`（容器启动参数）。

不需要重读：

- ARM SDOT / NEON / KML API 细节（这些在 Phase 4 才有用）。
- 老的 R1 教程（思路对、细节过时，只用来看 MoE 注入位置）。

---

## 9. 一次性验收 checklist（最终状态）

- [x] Phase 0：`pip install -e kt-kernel` 在 K920 + CANN 容器里能装；smoke 脚本通过。
- [ ] Phase 1.1：单 expert (gate/up/down 三矩阵) W8A8->Q8_0 转换数值正确。
- [ ] Phase 1.2：单 layer hybrid (4 NPU expert + 12 CPU expert) 输出 cosine sim > 0.999。
- [ ] Phase 2.4：CPUInfer 启动时起 report-poller 线程，`aclrtLaunchCallback` 真的被触发。
- [ ] Phase 2.7：单卡 SGLang 接 HTTP，单 prompt 输出和 8 卡服务前 32 token 一致。
- [ ] Phase 3：decode TPOT 比"全 NPU + 等"快 >= 1.5x（CPU 被 overlap 掉一半以上）。
- [ ] Phase 4：W8A8 直算路径上线，logits MSE 比 Q8_0 至少降 30%。

---

## 10. 用户偏好 / 协作约定

- 用户全程要求先方案、再代码。每个 phase 启动前需要简短计划 + token 估算 + 用户点头。
- 用户偏好简洁中文回复，不要绕。
- 用户的环境是真实容器，apt/pip 都受限，改动前先看路径是否在容器内。
- 用户的 token budget 紧张，写代码前一定先告知大致 token 消耗，能写文件就写文件不要堆 chat。
- 用户从 cursor + 多个模型混用，可能切到 Claude Code，所以所有状态都要落盘（git 改动 + 这种 handoff 文档），不要只活在一次对话里。

---

*最后更新：2026-05-12，Phase 0 完成，Phase 1.1 待开工。*
