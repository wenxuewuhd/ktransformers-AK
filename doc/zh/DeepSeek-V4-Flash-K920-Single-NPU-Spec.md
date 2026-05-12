# DeepSeek-V4-Flash 单卡 Ascend NPU + Kunpeng K920 CPU offload · 实施规格说明书

> **本文档面向 Claude Code（"实施方"），实施方在一台无 NPU、无 K920、无 CANN、无 DSv4 权重的开发机上盲写代码，无法在本地验证；最终代码要回到一个有 1× 910B + Kunpeng-920 + CANN 8.5.0 的容器（"验证环境"）由用户跑通。**
>
> 因此本文档的写法是规格说明（spec），不是教程：列出所有实施方"在他自己机器上查不到"的事实、接口契约、验收标准，并给出交付/调试约定。实现细节由实施方自己设计。
>
> 项目本身基于 `kvcache-ai/ktransformers`（`kt-kernel` + sglang submodule），目标版本基线：`d7b5b49` (`v0.6.2.post1`)。

---

## 0. 总览

| 项 | 值 |
| - | - |
| 模型 | DeepSeek-V4-Flash (DSv4-Flash, MoE, 43 层, 256 routed expert) |
| 量化 | W8A8（int8 权重 + fp32 per-output-channel scale，对称） |
| 验证硬件 | 1× Ascend 910B1 64GB HBM + Kunpeng-920 5250 (4S × 48C, 8 NUMA, 1.5TB DRAM) |
| CPU 后端选型 | LLAMAFILE backend + GGUF Q8_0（Phase 1-3）；KML CBLAS s8s8s32（Phase 4） |
| 框架 | SGLang fork (`ktransformers-AK/sglang` submodule，已能 8 卡跑 DSv4) |
| 主要交付 | (a) `kt-kernel` 编译期适配 (b) W8A8→Q8_0 转换器 (c) MoE Hybrid PoC (d) SGLang 单卡集成 (e) 异步 overlap (f) KML 精度回归 |

---

## 1. 验证环境上的"已知事实"（实施方可以假设这些为真）

> 实施方**不要**自己再去推测；以下都已在验证环境上实测确认过。

### 1.1 NPU 侧

- CANN root：`/usr/local/Ascend/ascend-toolkit/latest`。`CMakeLists.txt` 里通过 `find_path(... HINTS $ENV{ASCEND_TOOLKIT_HOME}/include /usr/local/Ascend/ascend-toolkit/latest/include)` 找。
- `aclrtLaunchCallback` 函数声明位置：`include/acl/acl_rt.h:975`。签名等价于 `aclError aclrtLaunchCallback(aclrtCallback fn, void* userData, aclrtCallbackBlockType blockType, aclrtStream stream)`。
- `torch_npu` 在验证容器内已装好，`torch.npu.Stream` / `torch.npu.Event` / `pin_memory=True` 都能用。
- `torch.npu.current_stream().npu_stream` 取出来是 `aclrtStream` 等价物，按 `uintptr_t` 传 pybind11 即可。**不要 include `torch_npu` C++ 头**（ABI 不稳）。

### 1.2 CPU 侧

- `lscpu`：`Kunpeng-920 5250`, 192 物理核 (4S×48C), 1 thread/core, 8 NUMA。
- `/proc/cpuinfo` flags **有**：`fp asimd asimddp asimdhp asimdrdm asimdfhm fphp jscvt fcma`。
- `/proc/cpuinfo` flags **没有**：`sve`, `sve2`, `bf16`, `i8mm`, `sme`。
- 这是 ARMv8.2-A + Dotprod + FP16，没有 SVE。**任何 SVE 汇编/intrinsic 在这台机器上都会触发 SIGILL**。
- KML 已装在 `/usr/local/kml`（Phase 4 才用）；已写 `/etc/ld.so.conf.d/kml.conf` + `ldconfig` 完毕。

### 1.3 模型权重事实（不要自己推测）

权重根目录：`/workspace/models/DeepSeek-V4-Flash-W8A8/`。包含 `config.json`、`tokenizer*`、`model.safetensors.index.json`、46 个 `model-*.safetensors`。

`config.json` 关键字段（实施方设计代码时按这些值定 shape 常量；但**代码本身要从 config 读，不要 hardcode**）：

```json
{
  "hidden_size": 4096,
  "moe_intermediate_size": 2048,
  "n_routed_experts": 256,
  "num_experts_per_tok": 6,
  "n_shared_experts": 1,
  "num_hidden_layers": 43,
  "first_k_dense_replace": <以 config.json 实际为准, 一般 1 或 3>
}
```

**权重张量命名（在 safetensors index 内已实测确认）**：

| 张量 | shape | dtype | scale 张量 | scale shape | scale dtype |
| - | - | - | - | - | - |
| `model.layers.{L}.mlp.experts.{E}.gate_proj.weight` | `(2048, 4096)` | `int8` | `....gate_proj.weight_scale` | `(2048, 1)` | `fp32` |
| `model.layers.{L}.mlp.experts.{E}.up_proj.weight`   | `(2048, 4096)` | `int8` | `....up_proj.weight_scale`   | `(2048, 1)` | `fp32` |
| `model.layers.{L}.mlp.experts.{E}.down_proj.weight` | `(4096, 2048)` | `int8` | `....down_proj.weight_scale` | `(4096, 1)` | `fp32` |

要点：

- scale 是 **per-output-channel**（一个 row 一个 scale），不是 per-group。
- scale 是 **fp32**，不是 fp16。
- 对称量化，**无 zero point**。
- 三个矩阵的别名约定（`kt-kernel` 内部命名习惯）：`w1 ↔ gate_proj`，`w3 ↔ up_proj`，`w2 ↔ down_proj`。
- Router 权重：`model.layers.{L}.mlp.gate.weight`（fp16 或 bf16，按 config 而定）。
- 共享 expert：`model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.*`。
- 前 `first_k_dense_replace` 层是稠密 MLP（没有 experts.*），实施方写 layer 遍历时必须跳过。

### 1.4 GGUF Q8_0 数据格式（kt-kernel 的 LLAMAFILE backend 输入格式）

```
block_q8_0 {
    ggml_fp16_t d;     // 2 bytes, 反量化 scale
    int8_t      qs[32]; // 32 bytes, 量化后的 int8 值
};                     // 总 34 bytes / 32 elements
```

整张矩阵字节数 = `numel / 32 * 34`，要求 `numel % 32 == 0`。DSv4-Flash 所有 expert 矩阵都满足这个对齐（2048×4096、4096×2048）。

量化算法：

```
for each 32-element block:
    amax = max(|x_i|) for i in 0..31
    d    = amax / 127.0
    qs[i] = round(x_i / d)  # clamp to [-127, 127]
```

参考实现：`third_party/llama.cpp/ggml/src/ggml-quants.c` 里的 `quantize_row_q8_0_ref`。**实施方应该用 llama.cpp 的 reference 实现**，而不是自己造轮子（避免 rounding 模式不一致）。

### 1.5 SGLang fork 关键限制

- Submodule 路径：`sglang/`。
- 文件 `sglang/python/sglang/srt/server_args.py` 在 `device == "npu"` 分支里硬 `assert` 拒绝所有 `--kt-*` 参数。Phase 2 必须在此 patch。
- DSv4-Flash 在 sglang 里的模型类：搜 `class DeepseekV4Flash` 或 `deepseek_v4_flash.py` 找。MoE block 大概率叫 `DeepseekV4FlashMoE` / `DeepseekV4FlashSparseMoeBlock`，里面有 `gate.weight` + 256 个 `experts[i]`。注入点就在这个 block 的 forward 里。
- 8 卡参考脚本：`sglang_dsv4_ascend_cann850.sh`（Docker 启动）。

### 1.6 `kt-kernel` Phase 0 已经具备的能力（实施方可以假设的接口表）

Phase 0 已经在验证环境上跑通编译 + import 冒烟。实施方写 Phase 1+ 时**可以假设以下都已存在并能工作**：

| C++/Python 符号 | 作用 | 备注 |
| - | - | - |
| `kt_kernel_ext.CPUInfer(n_workers: int)` | 创建 NUMA-aware worker pool | 构造时已支持 aarch64 |
| `kt_kernel_ext.moe.MOE` | C++ MoE backend 类 | LLAMAFILE backend，吃 GGUF Q8_0 |
| `kt_kernel_ext.moe.MOEConfig` | MoE 配置 | 字段名以源码为准，**实施方需要读 `kt-kernel/python_bindings/` 或 `kt-kernel/moe/` 确认** |
| `MOE.submit_with_cuda_stream(stream_ptr, ...)` | 异步提交 expert 计算 | 类型: `uintptr_t`，传 `torch.cuda.current_stream().cuda_stream` 或等价 |
| `MOE.sync_with_cuda_stream(stream_ptr)` | 等待该 stream 上的 MoE 任务完成 | 同上 |
| `submit_with_cuda_stream` 在 Ascend 编译路径下也启用 | C++ `#if KTRANSFORMERS_USE_CUDA \|\| KTRANSFORMERS_USE_ASCEND_NPU` | Phase 0 已改 |
| C++ macro `KTRANSFORMERS_USE_ASCEND_NPU` / `USE_ASCEND_NPU` | 两个 macro **同时存在且都需要保留**，分别给外层 CMake 和内层 vendor.h 用 | 不要去精简掉一个 |
| C++ 头 `cpu_backend/vendors/ascend_npu.h` | 提供 CUDA 兼容 wrapper：`cudaStream_t = aclrtStream`、`cudaLaunchHostFunc → aclrtLaunchCallback`、`cudaGetErrorString → aclGetRecentErrMsg` | Phase 0 已新建 |
| ARM `-march=armv8.2-a+fp16+dotprod` | CMake 默认 flag（K920 自动探测） | 别加 `+sve/+bf16/+i8mm` |
| KML / BLIS | `CPUINFER_ENABLE_KML=OFF`、`CPUINFER_ENABLE_BLIS=OFF` 在 aarch64+NPU 下默认关闭 | Phase 4 再开 |

实施方在新增代码里**调用 `submit_with_cuda_stream` 时**：参数语义就是"用这条 stream 上的 host callback 机制把 expert 任务串进 NPU 的 stream 排队"，stream 类型在 NPU 编译路径下其实是 `aclrtStream`，但 Python 侧统一传 `int`，不要做类型分支。

### 1.7 ACL Callback 语义差异（最容易踩坑的一处）

CUDA `cudaLaunchHostFunc` 是 driver 自动触发 host 函数。ACL `aclrtLaunchCallback` **不是**。要让 callback 真的被调用，必须在某个 host 线程里：

```cpp
// 这个伪代码描述行为, 实施方按实际 API 写

aclrtSubscribeReport(thread_id, stream);    // 把当前线程订阅给 stream
while (running) {
    aclrtProcessReport(timeout_ms);          // 在这个线程里处理 callback 队列
}
aclrtUnSubscribeReport(thread_id, stream);  // 收尾
```

→ Phase 2 实施方必须在 `CPUInfer` 启动时起一个 dedicated "report poller" 线程，否则提交到 NPU stream 的所有 host callback 永远不触发，表现是"程序 hang 在 sync 上但 NPU 空闲"。

参考头：`acl/acl_rt.h` 里搜 `aclrtSubscribeReport` / `aclrtProcessReport` / `aclrtUnSubscribeReport`。

---

## 2. 不可踩的红线（设计约束）

| # | 红线 | 后果 |
| - | - | - |
| R1 | 不要在 K920 编译路径上引入任何 SVE / BF16 / I8MM 指令（包括 `+sve` march、SVE intrinsic、`__bf16` 类型、`smmla` 指令） | 编译能过、运行 SIGILL |
| R2 | 不要 `#include <torch_npu/...>` 在 C++ pybind11 模块里 | torch_npu C++ ABI 不稳 |
| R3 | 不要假设 `aclrtLaunchCallback` 会自动触发 | 必须配 SubscribeReport+ProcessReport 线程 |
| R4 | 不要在 GGUF Q8_0 转换里"reinterpret" W8A8 的 int8 块 | scale 粒度完全不同（per-row vs per-32-elem），结果数值错误但不会报错 |
| R5 | 不要 commit 验证环境路径到代码里（`/workspace/...`、`/usr/local/Ascend/...`） | 用户回到容器后会撞到路径硬编码 |
| R6 | SGLang 注入不要 fork 整个 sglang 模型实现 | 升级 submodule 会破坏。**只在原文件里加分支** 或 **写一个继承类** |
| R7 | `submit_with_cuda_stream` 在 NPU 路径下，stream 是 `aclrtStream` 不是 `cudaStream_t`。Python 侧统一传 `int`（uintptr_t），C++ 侧 cast 回真实类型 | Python 类型校验失败 / 类型混用 |
| R8 | 共享 expert（`shared_experts`）和 router gate 不要 offload，留在 NPU | 共享 expert 是 dense MLP，offload 不划算；router 必须在 NPU 上做 topk |
| R9 | 前 `first_k_dense_replace` 层是 dense MLP，没有 256 个 expert，offload 逻辑要 skip | KeyError/IndexError |
| R10 | NEXTN（speculative decoding）第一版不要开 | sglang 的 NPU NEXTN 路径本身有坑 |

---

## 3. 关键决策对照表

| 维度 | 选项 | 选了 | 理由 |
| - | - | - | - |
| GPU 框架 | SGLang fork / 自写 | SGLang fork | 8 卡已跑通 |
| CPU 后端 Phase 1-3 | KML W8A8 / LLAMAFILE Q8_0 / BLIS | LLAMAFILE Q8_0 | KML 的 W8A8 路径有 SVE 汇编（R1） |
| CPU 后端 Phase 4 | LLAMAFILE Q8_0 / KML CBLAS | KML `cblas_gemm_s8s8s32` | 避免 Q8_0 重量化的精度损失；CBLAS 路径无 SVE |
| Hot/Cold 切分 | 全 CPU / 全 NPU / hybrid | hybrid，~16/256 NPU + ~240/256 CPU | 64GB HBM 算账：主干+KVcache+16 hot+activation ≈ 50-55GB |
| Speculative (NEXTN) | 开 / 关 | 关 | 减少 Phase 2 初版复杂度 |
| ARM march | 8.2 / 8.6 | armv8.2-a+fp16+dotprod | K920 实际能力（R1） |
| Async API | aclrtLaunchCallback / 手起 host worker | aclrtLaunchCallback | 与 CUDA 语义对齐成本最低 |
| P1.0 随机权重 PoC | 做 / 跳 | 跳 | 用户决定 |

---

## 4. Phase 1.1 · W8A8 → GGUF Q8_0 离线转换器（规格）

### 4.1 待新建文件

```
tools/_w8a8_dequant.py            # 内部 helper, 不对外
tools/convert_w8a8_to_gguf_q8_0.py # CLI 入口
```

### 4.2 接口契约

**CLI**（要求实施方实现成 `argparse` 命令行工具）：

```
python tools/convert_w8a8_to_gguf_q8_0.py \
    --src /path/to/DeepSeek-V4-Flash-W8A8 \
    --dst /path/to/out_dir \
    --layers 3-42                # 可选, 默认全部, 跳过 first_k_dense_replace
    --experts all                # 可选, 默认全部 256 个
    --workers 8                  # 可选, 多进程数
    --dry-run                    # 可选, 不写盘只跑数值校验
    --verify-cosine 0.9995       # 可选, 每个 tensor 算 cosine sim 与 dequant 后矩阵比, 低于阈值报错退出
```

**输出布局**（建议但非强制；选定后要在文档里固化，因为 P1.2 和 P2 都要按这个路径加载）：

```
<dst>/
  layer_{L}/
    expert_{E}/
      gate.q8_0.bin      # raw block_q8_0 数组, 大小 = (2048*4096/32)*34
      up.q8_0.bin        # 同 gate
      down.q8_0.bin      # 大小 = (4096*2048/32)*34
  manifest.json          # 列出全部生成文件 + 每个 tensor 的 原始/重量化 cosine sim
```

`manifest.json` 必填字段：`layer`, `expert`, `tensor` (gate/up/down), `path`, `shape`, `cosine_sim`, `max_abs_diff`, `bytes`。

### 4.3 算法约束

1. **W8A8 反量化**：`fp32_tensor = int8_weight.astype(fp32) * fp32_scale  # broadcast (N,1)*(N,M)`。
2. **Q8_0 重量化**：每 row 沿 input dim 切成 32 元 block，每 block 一个 fp16 scale 一个 int8 qs[32]，参考 `llama.cpp/ggml-quants.c::quantize_row_q8_0_ref`。
3. 处理顺序：**逐 expert 逐 tensor**，每个 tensor 处理完立即从 RAM 释放（safetensors 用 mmap 打开，处理完 close）。整 671B 权重不能一次性进 RAM。
4. **数值自检**：每个 tensor 转完后，把生成的 Q8_0 反量化（按 32 元 block，`d * qs`）和 §4.3.1 的 fp32_tensor 算 cosine similarity。要求 ≥ `--verify-cosine`（默认 0.9995）。低于阈值打印 tensor 名 + 实际 cosine + max diff，**continue 但累加错误计数**；超过 10 个就退出。
5. **多进程并行**：用 `concurrent.futures.ProcessPoolExecutor`。每个 worker 负责一个 (layer, expert) 元组，3 个 tensor 一起转。注意：safetensors mmap 在子进程要重新 open。

### 4.4 Phase 1.1 验收（在验证环境上跑）

- [ ] `--dry-run --layers 3 --experts 0` 跑通，无报错，cosine ≥ 0.9995。
- [ ] `--layers 3 --experts 0` 实际写盘，3 个 `.q8_0.bin` 文件大小符合预期（gate=`2048*4096/32*34`=8912896 字节，down 同样大小）。
- [ ] manifest.json 三条记录格式正确。
- [ ] 用 llama.cpp 的 `quantize` 工具或自己反量化对比，验证 Q8_0 字节序、scale 顺序无误。
- [ ] 全量转换：43 层 × 256 expert × 3 tensor = 33024 个 tensor，耗时 < 2h（8 worker）。总输出 ~580GB。

### 4.5 实施提示

- **盲写注意**：实施方拿不到 `quantize_row_q8_0_ref` 的源码，他可以自己写一个等价 Python 实现，但要在 docstring 里明确给出公式 + 注明对应 llama.cpp 函数名，以便用户在验证环境用 llama.cpp 工具回归对比。
- safetensors 取 tensor：用 `safetensors.safe_open(path, framework="pt", device="cpu")`，按 key 拿。**不要用 `safetensors.torch.load_file`** 整文件加载，会爆内存。
- `weight_scale` 形状是 `(N, 1)`，乘 `(N, M)` 时记得 `*=` 不要写 `* scale.unsqueeze(-1)`（已经是 2-D）。

---

## 5. Phase 1.2 · 单层 Hybrid Forward PoC（规格）

### 5.1 待新建文件

```
script/poc_dsv4_moe_p11_real_weights.py
```

### 5.2 目标

在 1 张 NPU + Kunpeng 上跑 1 层 MoE 的 forward，验证：

- `kt-kernel` 的 CPU 路径（LLAMAFILE Q8_0）数值正确。
- "全 NPU 算" 和 "hybrid (NPU + CPU offload)" 两条路径输出 cosine ≥ 0.999、max abs diff ≤ 5e-3。

### 5.3 接口契约（CLI）

```
python script/poc_dsv4_moe_p11_real_weights.py \
    --weights /workspace/models/DeepSeek-V4-Flash-W8A8 \
    --q8-dir /path/to/converted_q8_0 \
    --layer 3 \
    --hot 0,1,2,3 \              # 哪几个 expert 留在 NPU
    --batch 4 --seq 16 \
    --device npu:0
```

### 5.4 模块结构（实施方自行设计，但必须包含这些组件）

1. **`load_layer_router(weights, layer) -> torch.Tensor`**：从 safetensors 加载 `mlp.gate.weight`，转 fp16，搬到 NPU。
2. **`load_hot_experts(weights, layer, hot_ids) -> dict[int, ExpertModule]`**：把指定 expert 的 W8A8 权重反量化成 fp16，搬到 NPU 上，包成可调用模块。
3. **`load_cold_experts(q8_dir, layer, cold_ids) -> KTMoEHandle`**：通过 `kt_kernel_ext.moe.MOE(...)` 注册 CPU 端 expert（参数 shape、数量、Q8_0 路径），返回一个 handle。
4. **`forward_full_npu(hidden, router_w, all_experts_npu) -> output`**：把所有 256 个 expert 都搬 NPU、按 topk=6 跑（**仅用于参照基线**，可能 OOM；如果 OOM 就退化成"逐 expert NPU 跑"逐个累加，慢但正确）。
5. **`forward_hybrid(hidden, router_w, hot_experts_npu, cold_handle) -> output`**：在 NPU 上跑 router→topk→分流，hot id 走 NPU expert，cold id 通过 `cold_handle.submit_with_cuda_stream(...)` 走 CPU，最后合并。
6. **Verifier**：算 `(full_npu_out, hybrid_out)` 的 cosine + max abs diff，超阈报错。

### 5.5 验收

- [ ] 全 NPU 路径能跑（如果 OOM，回退方案要明确，比如逐 expert 顺算）。
- [ ] Hybrid 路径能跑，输出 shape 对，无 NaN/Inf。
- [ ] cosine ≥ 0.999、max abs diff ≤ 5e-3。
- [ ] 打印 hot/cold 各自耗时，CPU 部分能算出来不卡死即可（Phase 1 不追性能）。

### 5.6 实施提示

- **盲写注意**：`kt_kernel_ext.moe.MOE` 的具体构造函数签名实施方拿不到。**在代码里把它包装成自己的 `class KTMoEHandle`**，构造函数接受 `(intermediate_size, num_experts, q8_paths)` 这种语义化参数，内部组装 `MOEConfig` 调真实接口。如果验证时发现真实接口字段名不对，用户改 wrapper 内部一处即可，不要散落多处。
- 用 `torch.npu.current_stream().npu_stream` 取 stream，cast 成 `int` 透传 C++。
- `forward_hybrid` 里**必须在 sync 之前继续做 NPU 上能做的工作**（router 后处理、hot expert 累加），不要 submit 完立刻 sync。这样后面 Phase 3 的 overlap 才有意义。

---

## 6. Phase 2 · SGLang 单卡集成（规格）

> 用户最关心的里程碑："整网在单卡跑通" = Phase 2 完成。

### 6.1 子任务清单

| 编号 | 内容 | 交付物 |
| - | - | - |
| P2.1 | 放开 sglang `server_args.py` 对 NPU 后端 `--kt-*` 的 assert | patch 到 `sglang/python/sglang/srt/server_args.py` |
| P2.2 | MoE block 注入 kt-kernel offload 分支 | patch 或继承类 `deepseek_v4_flash_ktmoe.py` |
| P2.3 | Python wrapper `KTMoEHandle` 适配 NPU stream | `kt_kernel/npu_adapter.py` 或同级 |
| P2.4 | ACL callback poller 线程接入 `CPUInfer` | C++ patch 到 `kt-kernel/cpu_backend/cpuinfer.cc` |
| P2.5 | Hot/Cold 静态切分配置 | `config/cpu_offload_dsv4_flash.yaml` |
| P2.6 | 权重加载：CPU 走 Q8_0 路径；NPU 只加载 hot expert | 在 P2.2 同文件里实现 |
| P2.7 | 端到端冒烟：prompt → 32 token decode 与 8 卡服务对齐 | 测试脚本 + 对齐 log |
| P2.8 | 单卡启动脚本 | `sglang_dsv4_ascend_cann850_singlecard.sh` |

### 6.2 P2.1 详细规格

`server_args.py` 在 npu 分支硬 assert `kt_*` 全部为默认值。修改方式：

- **不要删除 assert**，把它改成"当 `--kt-cpu-experts` / `--kt-amx-method` 等非默认值出现时，记 `self._kt_offload_enabled = True` 并跳过 assert，否则保留原行为"。
- 新增的 server arg 字段（建议）：

| 参数 | 类型 | 默认 | 含义 |
| - | - | - | - |
| `--kt-cpu-experts` | path | None | `cpu_offload_dsv4_flash.yaml` 的路径 |
| `--kt-amx-method` | str | `LLAMAFILE_Q8_0` | 兼容 GPU 路径的同名参数，但在 NPU 路径下只接受 `LLAMAFILE_Q8_0` 或 `KML_S8S8`（后者 Phase 4 加） |
| `--kt-num-cpu-workers` | int | 128 | `CPUInfer(n_workers=...)` 透传 |
| `--kt-cpu-weight-dir` | path | None | Q8_0 转换器的输出目录 |

### 6.3 P2.2 详细规格

注入位置：DSv4-Flash 的 MoE block forward。伪代码描述（实施方不要照抄，要按真实模型类的 forward 签名重写）：

```
def forward(self, hidden_states):
    # 1. 原 router
    router_logits = self.gate(hidden_states)
    topk_w, topk_idx = topk(router_logits, k=6)

    # 2. 共享 expert (留 NPU, 不动)
    shared_out = self.shared_experts(hidden_states)

    # 3. 路由分流: hot id 走 NPU, cold id 走 CPU
    is_cold = mask_in(topk_idx, self.kt_cold_set)   # 在 CPU 跑的 expert id
    # 3a. 提交 CPU 任务 (异步, 不立即等)
    cold_handle = self.kt_moe.submit(hidden_states, topk_idx, topk_w, mask=is_cold,
                                      stream=torch.npu.current_stream().npu_stream)
    # 3b. NPU 上算 hot expert (与 CPU 并行)
    hot_out = self._npu_topk_moe(hidden_states, topk_idx, topk_w, mask=~is_cold,
                                  experts=self.kt_hot_experts)
    # 3c. 取回 CPU 结果
    cold_out = cold_handle.wait()    # 内部走 sync_with_cuda_stream

    return shared_out + hot_out + cold_out
```

**关键约束**：

- 注入要尽量"贴"原 forward 的 shape/dtype，避免动 attention/normalize 等上下文。
- 不要走 sglang 的 capture-graph 路径里复制 kt-kernel 调用（cuda graph 兼容性留到 Phase 3 再说，第一版强制 `--disable-cuda-graph` 或对应 NPU 参数）。
- `self.kt_cold_set` / `self.kt_hot_experts` / `self.kt_moe` 这些字段在 `__init__` 里通过读 `--kt-cpu-experts` yaml 装配。

### 6.4 P2.3 Python wrapper 规格

`KTMoEHandle` 至少暴露：

```python
class KTMoEHandle:
    def __init__(
        self,
        layer_id: int,
        intermediate_size: int,
        n_routed_experts: int,
        cold_expert_ids: list[int],
        q8_weight_dir: str,
        cpu_infer: kt_kernel_ext.CPUInfer,
    ) -> None: ...

    def submit(
        self,
        hidden_states: torch.Tensor,   # (B*T, H)  on NPU
        topk_idx: torch.Tensor,        # (B*T, k)  on NPU
        topk_w: torch.Tensor,          # (B*T, k)  on NPU
        cold_mask: torch.Tensor,       # (B*T, k)  bool, True=offload to CPU
        stream: int,                   # aclrtStream as uintptr_t
    ) -> KTMoEFuture: ...

class KTMoEFuture:
    def wait(self) -> torch.Tensor:    # (B*T, H)  on NPU
        """内部调 sync_with_cuda_stream, 然后返回回填到 NPU 的累加结果"""
```

**职责**：

- 把 NPU 上的 `hidden_states` 拷到 host pinned buffer（用 `torch.empty(..., pin_memory=True)` 或 `tensor.to('cpu', non_blocking=True)` + sync event）。
- 在 host 端按 cold_mask 分发到对应 cold expert，调 `kt_kernel_ext.moe.MOE.submit_with_cuda_stream(stream, ...)`。
- 结果累加完成后从 host 拷回 NPU。
- 拷贝两端都要走 stream，不要插同步 `to(...)`。

### 6.5 P2.4 ACL Poller 规格

修改 `kt-kernel/cpu_backend/cpuinfer.cc`（实施方需要先 Read 这个文件确认实际接口）：

- 在 `CPUInfer` 构造时，**若 `KTRANSFORMERS_USE_ASCEND_NPU` 编译开**，起一个 std::thread 跑 poller。
- Poller 线程函数体：
  ```
  pthread_setname_np("kt-acl-poll");
  aclrtSubscribeReport(reinterpret_cast<uint64_t>(pthread_self()), stream);
  while (!stop_) {
      aclrtProcessReport(100);   // 100ms 超时
  }
  aclrtUnSubscribeReport(reinterpret_cast<uint64_t>(pthread_self()), stream);
  ```
- 这里的 `stream` 来自哪儿？**问题**：CPUInfer 一开始不知道用户会用哪条 stream。两种方案，选一种：
  - **(a) Lazy subscribe**：第一次 `submit_with_cuda_stream` 时记录 stream 并 subscribe。后续 submit 检查 stream 是否变了，变了就 re-subscribe（多 stream 时维护一个 set→thread 映射）。
  - **(b) 显式 register**：暴露一个 `CPUInfer.register_npu_stream(int stream)` Python 方法，调用方在初始化时手动注册全部要用的 stream。
  - **推荐 (a)**，对调用方更友好；如果 thread 安全难处理就用 (b)。

### 6.6 P2.5 yaml 规格

```yaml
# config/cpu_offload_dsv4_flash.yaml
model: deepseek_v4_flash
strategy: static
hot_size: 16              # 每层 hot expert 数
cold_size: 240            # 每层 cold expert 数
shared_experts: on_npu    # 始终
router: on_npu            # 始终

per_layer:
  default:
    hot: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  3:                       # layer-3 个性化, 可选
    hot: [...]
```

**第一版策略**：所有 MoE 层都用 `default.hot = [0..15]`，无个性化。后续可以基于 profiling 替换。

### 6.7 P2.7 验收

- [ ] 单卡能拉起 sglang，HBM 占用 ≤ 60GB。
- [ ] 单 prompt 输入，前 8 token 输出与 8 卡参考服务**完全一致**（greedy）。
- [ ] 前 32 token 一致，或 logits MSE ≤ 1e-3。
- [ ] 跑 100 个 prompt 不崩，无 NaN。

### 6.8 P2.8 启动脚本

复制 `sglang_dsv4_ascend_cann850.sh`，按以下要点改：

- `--tp 1`（或对应单卡参数）。
- 加 `--kt-cpu-experts /path/to/cpu_offload_dsv4_flash.yaml --kt-amx-method LLAMAFILE_Q8_0 --kt-cpu-weight-dir /path/to/converted_q8_0 --kt-num-cpu-workers 128`。
- 关 NEXTN / speculative decoding 相关参数（具体参数名以 sglang 现有为准）。
- 关 cuda graph / npu graph capture：第一版用 eager mode。
- 容器入参 `ASCEND_VISIBLE_DEVICES=0`（或对应名）。
- 保留容器内 apt/pip 代理配置。

---

## 7. Phase 3 · 异步 Overlap（规格）

### 7.1 目标

decode 阶段，CPU expert 计算 + NPU↔Host 拷贝 + NPU expert 计算三段流水化，**CPU 部分至少 70% 被 overlap 掉**。

### 7.2 设计要点

1. P2.2 注入位置已经实现 "submit 后继续算 hot，再 wait" 的模式，Phase 3 主要是优化 KTMoEHandle 内部：
   - host pinned buffer 预分配，避免每次拷贝再分配。
   - NPU→host 拷贝用 `non_blocking=True` + record event。
   - CPU 算完后 host→NPU 拷贝也走 stream。
   - 这一对 record/wait 用 `torch.npu.Event`。
2. CPU 端 `MOE.submit_with_cuda_stream` 已有 NUMA worker pool，但要确保 worker 数和 `cold_mask.sum()` 平衡，避免空转。
3. 测量方法：在 P2.7 基础上加 timing。要求 decode TPOT（time per output token）相比 P2.7 提升 ≥ 1.5×。

### 7.3 验收

- [ ] TPOT ≥ 1.5× P2.7 的值。
- [ ] HBM/DRAM 占用没明显涨（pinned buffer ≤ 1GB）。
- [ ] 输出 token 与 P2.7 完全一致（性能优化不能改数值）。

---

## 8. Phase 4 · KML W8A8 直算精度回归（规格）

### 8.1 待做

1. 让 `kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel/` 下的 KML 源参与编译，但**只编依赖 CBLAS API 的 `.c/.cpp`，绕开所有 `.S` SVE 汇编**。具体做法：在 CMakeLists.txt 加 `set(KT_KML_USE_CBLAS_ONLY ON)`，按 source 列表白名单 add_library。
2. 新建 backend `Int8_KERNEL_MOE_KML_CBLAS`：
   - 输入：W8A8 int8 weight + fp32 per-row scale。
   - GEMM：`cblas_gemm_s8s8s32(layout, transA, transB, transC, M, N, K, alpha=1.0f, A, lda, oA=0, B, ldb, oB=0, beta=0.0f, C_s32, ldc, oC=null)`（具体签名以 KML 2.5.0 doc 为准）。
   - 后处理：`C_fp32 = C_s32 * scale_A_broadcast * scale_B_broadcast`，再走原 SiLU/Down 流程。
3. backend 通过 `--kt-amx-method KML_S8S8` 启用。

### 8.2 验收

- [ ] LLAMAFILE_Q8_0 和 KML_S8S8 两条路径，logits MSE vs GPU 全精度参考：KML_S8S8 至少降 30%。
- [ ] KML_S8S8 的 TPOT 不比 Q8_0 慢超过 20%。

---

## 9. 交付与验证流程（盲写场景关键）

### 9.1 实施方交付物形态

实施方在自己机器上不能跑 cmake/python 验证，**所以交付物必须是 git patch 系列**，每个 Phase 一组：

```
deliverables/
  phase1_1_converter.patch         # tools/_w8a8_dequant.py + tools/convert_w8a8_to_gguf_q8_0.py
  phase1_1_converter.notes.md      # 设计说明 + 已知 TODO + 用户在验证环境要看的 N 个点
  phase1_2_poc.patch
  phase1_2_poc.notes.md
  phase2_1_server_args.patch
  phase2_2_moe_injection.patch
  phase2_3_kt_handle.patch
  phase2_4_acl_poller.patch
  phase2_5_yaml_config.patch
  phase2_6_weight_loader.patch
  phase2_7_smoke_test.patch
  phase2_8_launch_script.patch
  phase3_overlap.patch
  phase4_kml_cblas.patch
```

每个 patch 要：

- 基线明确（`git apply` 在 `d7b5b49` + Phase 0 改动 之上）。
- 不混合其他 phase 内容。
- 每个 `.notes.md` 写：(1) 设计要点、(2) 已知未验证的不确定点 (实施方猜测的接口名 / 字段名) 、(3) 用户在验证环境要 grep / 跑哪个命令来确认。

### 9.2 在验证环境上的回归流程（用户侧）

```
# 1. 同步 Phase 0 基线（已 commit, 或本地 patch）
cd /workspace/code/ktransformer/ktransformers-AK
git status                         # 确认 Phase 0 改动在位
ls kt-kernel/cpu_backend/vendors/ascend_npu.h  # Phase 0 关键文件存在

# 2. 应用实施方 patch
git apply --check deliverables/phaseX.patch && git apply deliverables/phaseX.patch
# 失败则把冲突给实施方修

# 3. 编译（aarch64 容器内）
cd kt-kernel && bash install.sh
# 期望: 自动探到 K920 + CANN, build 成功, import kt_kernel_ext 不报错

# 4. 按 phase 跑对应验收命令
bash deliverables/phaseX_verify.sh  # 如果实施方提供了 verify 脚本
```

### 9.3 不确定点回写约定

实施方在 `.notes.md` 里列的不确定点，用户回到验证环境跑过后，要把结果**写回到本文档的 §10 "运行记录"**，下次实施方接着写时能看到。

---

## 10. 运行记录（待用户填写，每次回归后补充）

| 日期 | Phase | 实施方 patch 版本 | 验证结果 | 调整点 |
| - | - | - | - | - |
| TBD | 1.1 | v1 | ⏳ | ⏳ |

---

## 11. 调试 hints（盲写者最容易犯的错）

| 现象 | 大概率原因 | 看哪里 |
| - | - | - |
| `undefined symbol: iqk_mul_mat_moe_arm82` | `third_party/llamafile/iqk_mul_mat_arm82.cpp` 那两行 `#define` 又被注释掉了（rebase 上游引入） | grep `iqk_mul_mat` 看 define 是否在 |
| `SIGILL` 跑 SDOT/SVE 错码 | march 里混了 `+sve`，或 KML 链了 SVE 路径 | 看 cmake configure 输出 `-march=...`；看 `CPUINFER_ENABLE_KML` |
| 程序卡在 `sync_with_cuda_stream` | ACL poller 线程没起，callback 永不触发 | 看 CPUInfer 构造日志里有没有 `kt-acl-poll` 线程；ps -L 看线程名 |
| `cosine sim 比 0.99 还低` | W8A8 反量化时 scale 没 broadcast 对、或者 Q8_0 block scale 用了 fp32 而非 fp16 存盘 | 在 dry-run 里加 print 看中间 fp32 矩阵；用 llama.cpp `quantize --check` 对照 |
| HBM 一启动就 OOM | NPU 误加载了 cold expert | 在 `load_weights` 里加 log，看 cold expert 的 tensor 是不是 skip |
| 输出 logits 全 NaN | 第一层 MoE 的 fp32 中间结果未在合适位置 cast 回 fp16 / bf16 | 在 hybrid forward 各步骤加 isfinite check |
| sglang 启动报 `kt-* not supported on npu` | P2.1 没生效 / patch 没 apply | grep server_args.py 看 assert 文本 |
| kt-kernel import 报缺 libascendcl | `LD_LIBRARY_PATH` 没指向 `$ASCEND_TOOLKIT_HOME/lib64` | echo $LD_LIBRARY_PATH；source `set_env.sh` |
| 整网编译过但 import 报 `KTRANSFORMERS_USE_ASCEND_NPU` 相关 macro 行 hit 不到 | `setup.py` 没透传 `-DKTRANSFORMERS_USE_ASCEND_NPU=ON` | rm -rf build && env CPUINFER_USE_ASCEND_NPU=1 pip install -e . -v 看 cmake 命令 |

---

## 12. 参考资料

仓库内：

- `doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md`：老 R1 单卡教程，思路对，细节过时，**只**看 MoE 注入点位置。
- `doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md`：Qwen3-MoE NPU 注入参考，**主要参考它的 Python wrapper 模式**。
- `doc/en/DeepSeek-V4-Flash.md`：GPU 上 V4-Flash 跑法，看 server args 命名习惯。
- `sglang_dsv4_ascend_cann850.sh`：8 卡 NPU 启动脚本基线。
- `kt-kernel/python_bindings/`、`kt-kernel/cpu_backend/`、`kt-kernel/moe/`：实施方写 wrapper 前**必须读**的源。
- `doc/zh/DeepSeek-V4-Flash-K920-Single-NPU-Handoff.md`：用户当前进度 handoff，**实施方不必读**，给用户和 Cursor 用的。

外部：

- CANN aclrt API：`$ASCEND_TOOLKIT_HOME/include/acl/acl_rt.h`、华为 CANN 文档 "aclrtLaunchCallback"。
- llama.cpp Q8_0 参考实现：`third_party/llama.cpp/ggml/src/ggml-quants.c::quantize_row_q8_0_ref`。
- KML 2.5.0 CBLAS doc：`/usr/local/kml/include/kblas.h` + 华为 BoostKit 文档 KML 章节。

---

*文档版本 v1, 2026-05-12, 对应仓库基线 `d7b5b49` + Phase 0 patch（见 `doc/zh/DeepSeek-V4-Flash-K920-Single-NPU-Handoff.md`）。*
