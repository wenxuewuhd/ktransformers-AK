# DeepSeek-V4-Flash 单卡 910B + K920 CPU MoE —— 总体方案与当前进展（整合版）

> **文档定位**：本文是 DeepSeek-V4-Flash「单卡 Ascend 910B + Kunpeng-920 CPU MoE offload」方案的**唯一现行总纲**,
> 整合了以下 4 份来源,并对其中**已过时的结论做了显式标注与取代**:
>
> | 来源 | 时间 | 角色 | 现状 |
> |---|---|---|---|
> | `DeepSeek-V4-Flash-K920-Single-NPU-Spec.md` | 2026-05-12 | 盲写阶段规格书(实施方契约) | 架构/接口仍有效;**量化结论已过时** |
> | `DeepSeek-V4-Flash-K920-Single-NPU-Handoff.md` | 2026-05-12 | Phase 计划与红线 | 红线仍有效;**Q8_0/KML 结论已过时** |
> | `DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md` | 2026-05-19 | Phase 0/1/2 完成实录 | 大部分现行;graph 状态见 §6.3 |
> | `DeepSeek-V4-Flash_单卡910B_从0拉起服务全记录.md` | 本会话 | 从 0 拉起的最新实操 | **最新事实基准** |
>
> 当来源之间冲突时,**以本文为准**;本文又以「全记录」+ 实测为最新事实基准。
> 维护分支:`dsv4_one_card_dev`。最后更新:2026-06-08(graph capture 已闭合)。

---

## 0. 一句话现状

单卡 910B + K920 的 DeepSeek-V4-Flash(W8A8)推理**已可端到端拉起并输出连贯文本,且 NPU graph
性能路径已闭合**。当前生产配置:**Q8_0 GGUF(~275 GiB)+ CPU MoE offload + NPU attention + graph-on**。
**graph capture(坑⑥ `aclrtMemcpy 107030`)+ 图重放跨 stream(`Unsupport run graph`)已于
2026-06-08 修复并闭环验证**:真实权重 graph-on 全程跑通,decode `npu graph: True` ~3.5–3.9 tok/s
(达基线 ~3.6,取代 eager ~1.6),F2 四 prompt 连贯(§6.3)。eager 回退仍保留作对照。

---

## 1. 环境与硬件规格

### 1.1 硬件

| 部件 | 配置 |
|---|---|
| **CPU** | Kunpeng-920 5250,4 socket × 48 core = **192 物理核**,8 NUMA(每 NUMA 24 核 ~192 GB),**1.5 TB DRAM** |
| **CPU ISA** | ARMv8.2-A + `asimddp`(NEON SDOT)+ `asimdhp`/`fphp`(FP16);**无 SVE / 无 BF16 / 无 I8MM / 无 SME** |
| **NUMA 距离** | 同 die(0,1)/(2,3)…=11;跨 die=24–25;跨 socket=32 |
| **NPU** | 8 × Atlas 910B1(每张 64 GB HBM);**项目只用 1 张** |
| **CANN** | 8.5.0,`/usr/local/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/cann-8.5.0` |

> **ISA 红线(R1)**:任何 SVE / BF16 / I8MM 指令(`+sve` march、SVE 汇编、`__bf16`、`smmla`/`usdot`/`ptrue`)
> 在 K920 上会 **SIGILL**。编译 march 固定 `-march=armv8.2-a+fp16+dotprod`。

### 1.2 软件栈

| 组件 | 版本/路径 |
|---|---|
| OS | Ubuntu 22.04 (jammy), aarch64 |
| Python | `/usr/local/python3.11.14/bin/python3.11`(3.11.14) |
| PyTorch | `2.8.0+cpu` |
| torch_npu | `2.8.0.post2`(支持 `pin_memory=True`、`torch.npu.Stream/Event`) |
| SGLang | 子模块 `third_party/sglang/`(启动前必须 export `PYTHONPATH`) |
| llama.cpp | 子模块 `third_party/llama.cpp/`,公开 tag **b3173**(`a94e6ff`) |
| KML | 2.5.0,`/usr/local/kml/`(**仅 Phase 4 候选,当前不链**) |
| hwloc | `libhwloc-dev 2.7.0-2ubuntu1`(kt-kernel 硬依赖;**每容器需重装**,见 §4.1) |
| numactl | 已装 |

### 1.3 模型规格(`/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8/config.json`)

| 参数 | 值 |
|---|---|
| 层数 `num_hidden_layers` | **43**(全 MoE,`first_k_dense_replace=0`) |
| `hidden_size` | 4096 |
| `n_routed_experts` | 256 / 层 |
| `num_experts_per_tok` | **6**(top-k,注意不是 8) |
| `n_shared_experts` | 1 / 层 |
| `moe_intermediate_size` | 2048 |
| `head_dim` | 512;`num_attention_heads`=64;`num_key_value_heads`=1 |
| Attention | MLA + NSA(稀疏)+ Lightning Indexer(`index_n_heads`=64,`index_topk`=512,`index_head_dim`=128) |
| `topk_method` / `scoring_func` | `noaux_tc` / `sqrtsoftplus`;`routed_scaling_factor`=1.5;`norm_topk_prob`=True |
| `torch_dtype` | bfloat16 |
| `num_nextn_predict_layers` | 1(speculative,**本项目禁用**) |

**W8A8 权重张量**(safetensors,对称量化、无 zero-point、per-output-channel fp32 scale):

| 张量(别名) | shape | dtype | scale shape/dtype |
|---|---|---|---|
| `…experts.E.gate_proj.weight`(w1) | (2048, 4096) | int8 | (2048,1) fp32 |
| `…experts.E.up_proj.weight`(w3) | (2048, 4096) | int8 | (2048,1) fp32 |
| `…experts.E.down_proj.weight`(w2) | (4096, 2048) | int8 | (4096,1) fp32 |

> 权重根目录共 46 个 safetensors shard,~275 GB。`mlp.gate.weight`(router)与 `shared_experts.*` 留 NPU。

---

## 2. 系统架构与数据流

```
单卡:Atlas 910B (64 GB HBM) + K920 (1.5 TB DRAM, 192 核, 8 NUMA)

input → [NPU: embedding / RoPE / MLA+NSA+Indexer attention]
      → [NPU: MoE router gate → topk_ids, topk_weights(k=6)]
      → ┌──────────────────────────┬──────────────────────────┐
        │ NPU experts (N=32 默认)   │ CPU experts (224 默认)     │
        │ W8A8 safetensors         │ kt-kernel LLAMAFILE GGUF  │
        │ + shared experts (常驻)   │ (Q8_0 / BF16)             │
        └──────────────────────────┴──────────────────────────┘
      → merge → linear + residual → 下一层
```

### 2.1 NPU 端
- Attention:MLA + NSA + Lightning Indexer + Compressor(SGLang `--attention-backend ascend`)。
- NPU MoE:`fused_experts_npu`(W8A8),承载前 N 个 routed expert + shared expert + router topk。
- KV cache:HBM(必要时可 offload DRAM)。

### 2.2 CPU 端(kt-kernel)
- backend:LLAMAFILE(`kt-kernel/operators/llamafile/moe.hpp` → `LLAMA_MOE_TP`)。
- 8 个 NUMA worker pool(每 NUMA 24 **核**;默认 `--kt-cpuinfer 24 --kt-threadpool-count 8` →
  `24/8 = 3` 线程/subpool,即每 NUMA 3 线程),NEON SDOT 内核。
- Expert layout(经 Z.2 修复后):
  - **gate/up**:`(E=256, intermediate=2048, hidden=4096)`,沿 hidden 分 Q8_0 block;
  - **down**:`(E=256, hidden=4096, intermediate=2048)`,沿 intermediate 分 block。

### 2.3 NPU↔CPU 桥(graph callback,任务2)
- `kt-kernel/cpu_backend/ascend_callback_worker.{cpp,h}`:后台线程 `aclrtSubscribeReport` + 循环
  `aclrtProcessReport`,把 CPU MoE 的 submit/flush 接进 NPU graph 的 host callback。
- 关键差异(红线 R2/R3):ACL 的 `aclrtLaunchCallback` **不会自动触发**,必须有专用 poller 线程;
  否则表现为「卡在 sync、NPU 空闲」。

### 2.4 SGLang 集成
- 模型:`third_party/sglang/python/sglang/srt/models/deepseek_v4.py`。
- KT wrapper:`…/layers/moe/kt_ep_wrapper.py`(per-layer `KTMoEWrapper`,`mask_cpu_expert_routing`,
  prefill/decode 分化,graph 走 host callback)。
- 设备抽象:`…/utils/kt_accel.py`(CUDA↔NPU stream/event/sync 透明)。
- Triton 兜底:`triton 3.7 × triton-ascend 3.2` 错配时走 torch 等价实现(`SGLANG_NPU_ALLOC_FORCE_NAIVE=1`),不影响数值。

---

## 3. 量化与权重方案 ⚠️(关键:旧结论已取代)

### 3.1 现行事实(最新基准)

> **取代声明**:Spec/Handoff(05-12)里「Q8_0 在 aarch64 会 NaN」「MOE_INT8/KML 在 K920 不可用,故必须 BF16(555 GiB)」
> 的结论**已过时**。实测:**Q8_0(int8)CPU offload 完全可用**,离线对账 cosine ≈ **0.9999**(BF16 ≈ 0.999997)。

| 格式 | 单层 | 43 层合计 | 现状 |
|---|---|---|---|
| **Q8_0** | ~6.85 GiB | **~275 GiB** | **现行生产路径** |
| BF16 | ~12.9 GiB | ~555 GiB | 数值基线/回退,非必须 |

### 3.2 W8A8 → GGUF 转换
- 工具:`tools/batch_convert_w8a8_layers_mp.py`(`ProcessPoolExecutor`,`--jobs` = 并发层数)。
- dequant→requant:`W_fp32 = int8 * fp32_scale[out_ch]`;再按 Q8_0 block(每 32 元 fp16 scale + int8 qs[32])重量化。
- 调优(本机 192 核):`--jobs 32` 较优(聚合 ~129/192 核,~121 GB RAM,磁盘 I/O 成瓶颈)。
- 命令见 §4.3。

### 3.3 KML / MOE_INT8 —— **不做**
K920 无 SVE/i8mm,KML `prefillgemm` 的 `usdot`/`ptrue` 内联汇编不可编译;CBLAS `s8s8s32`(NEON)理论可用但
当前 Q8_0 已满足精度,**不投入 Phase 4**。

---

## 4. 复现 / 拉起流程

### 4.1 每次新 container 需重做的(仅 1 条)

`/workspace` 挂载持久化:代码、子模块指针、编译产物 `kt_kernel_ext*.so`、GGUF 权重、软链都在。
**唯一非持久**的是系统 apt 包 hwloc(`~/.ssh`、CANN 镜像层视镜像而定):

```bash
apt-get install -y libhwloc-dev          # 运行期 import kt_kernel 依赖 libhwloc.so.15
```

> 自检:`apt` 装好后 `import kt_kernel`、`import torch_npu`、`kt_ep_wrapper` 应一次过;
> `bash tools/p27_e2e_preflight.sh` → PASS(43 GGUF + kt_kernel_ext 路径)。

### 4.2 编译 kt-kernel(如需重编;通常 .so 已持久化不必)

```bash
cd /workspace/code/ktransformers-AK/kt-kernel
CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
# 预期 march=armv8.2-a+fp16+dotprod;ldd 链接 libascendcl/libruntime;import kt_kernel_ext 不报 undefined symbol
```

### 4.3 W8A8 → 43 层 GGUF(Q8_0)

```bash
mkdir -p /workspace/models/cache
nohup /usr/local/python3.11.14/bin/python3.11 tools/batch_convert_w8a8_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --quant q8_0 --jobs 32 --verify-sample 3 \
  > /tmp/kt_convert.log 2>&1 &
# 输出 dsv4_layer{0..42}.gguf
```

### 4.4 拉起服务

设备选择:`NPU_DEVICE_ID=<卡号>`(脚本内部 export 成 `ASCEND_RT_VISIBLE_DEVICES`;进程内逻辑为 `npu:0`)。
启动前先 `npu-smi info` 选空闲卡。

```bash
cd /workspace/code/ktransformers-AK

# (A) 默认:graph-on(性能路径,脚本默认不传 --disable-cuda-graph)
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=0 \
bash tools/p27_launch_ds4flash_npu.sh

# (B) 回退:eager + 强制同步(功能正确,见 §6.3 坑⑥⑦)
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=0 KT_FORCE_SYNC_SUBMIT=1 EXTRA_FLAGS="--disable-cuda-graph" \
bash tools/p27_launch_ds4flash_npu.sh
```

关键 launch 参数:`--device npu --tp 1 --attention-backend ascend --quantization compressed-tensors
--dtype bfloat16 --kt-method LLAMAFILE --kt-num-gpu-experts 32 --kt-weight-path .../dsv4_layer{layer_idx}.gguf
--kt-threadpool-count 8 --kt-cpuinfer 24 --chunked-prefill-size 2048`(**勿传 -1**,见坑⑨)。

### 4.5 验证(等 ~100s 加载;P0+P1 加载加速前为 ~9 min)

```bash
curl -sf http://127.0.0.1:8000/health        # 200
curl -sS -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是","sampling_params":{"max_new_tokens":40,"temperature":0}}'
# 期望连贯("北京…");若"我,我,我…"退化重复 → 走到了 async 未 flush 路径,用 (B) 回退
```

---

## 5. 全坑汇总(合并 + 现状)

| # | 现象 | 根因 | 修复 / 现状 |
|---|------|------|------|
| ① | CMake 找不到 hwloc | 系统未装 | `apt-get install -y libhwloc-dev`(**每容器**) |
| ② | llamafile 编译 `llama.cpp/ggml-impl.h: No such file` | llama.cpp 子模块在新版,头文件布局变了 | 钉公开 tag **b3173**(`a94e6ff`);**本会话已固化**(§6.1) |
| ③ | `import kt_kernel` → `undefined symbol: iqk_mul_mat_moe_arm82` | `iqk_mul_mat_arm82.cpp` 两行 rename `#define` 被注释 | 取消注释 + 重编;**本会话已 commit** |
| ④ | 转换 `--verify-sample` 报 `newbyteorder` removed | gguf-py NumPy 2.0 不兼容 | apply `tools/kt_dsv4_npu_patches/llama_cpp/0001-*.patch`(只影响读取/校验,不影响权重与 serving) |
| ⑤ | 启动崩 `quant fp8 != compressed-tensors` / `n_activated_experts` | sglang 子模块切错 fork(无 KT 补丁) | 切 `dsv4_release@a347a9ad5`;**本会话已固化到 yy_repo**(§6.1) |
| ⑥ | **graph 捕获崩 `aclrtMemcpy 107030`** | `mask_cpu_expert_routing` 内 `gpu_experts_mask.to(device)` 在 capture 期做同步 H2D memcpy | **✅ 已修(2026-06-08)**:`process_weights_after_loading` 内 capture 前把 mask 预搬到 device,`.to()` 短路为 no-op(§6.3 改动一) |
| ⑥b | graph **重放**崩 `Unsupport run graph with different stream`(ERR03005) | `mask_cpu_expert_routing` 被 `@torch.compile`,NPU torchair 后端把它降为绑定 stream 的独立子图,外层图跨 stream replay 冲突 | **✅ 已修(2026-06-08)**:去掉该函数 `@torch.compile`,eager 折进外层图(§6.3 改动五) |
| ⑦ | eager 出 token 但乱码(重复) | CPU MoE async submit 未 flush → 输出全零 | `KT_FORCE_SYNC_SUBMIT=1`(§6.3) |
| ⑧ | Q8_0 aarch64 NaN(历史) | `iqk_mul_mat_arm82`/`tinyBLAS_Q0_ARM` 在 dotprod-only 上 NaN | 已修(无 i8mm 时回退 `ggml_vec_dot_q8_0_q8_0`);Q8_0 现可用 |
| ⑨ | `--chunked-prefill-size -1` → `malloc unaligned tcache` | `max_len=-1` → C++ 按 1 分配 buffer,prefill 越界 | 默认改 2048;`llamafile.py` ≤0 回落 2048 |
| ⑩ | W8A8→GGUF layout 错(输出退化 " ! ! !") | 转换器 `permute(2,1,0)` 与 kt-kernel pointer 算术不一致(Z.2) | 去 permute、expert 维 `axis=0` concat;BF16 cosine ≥ 0.999996 |
| ⑪ | Triton-on-NPU `0 active drivers` | `triton 3.7 × triton-ascend 3.2` 错配 | torch fallback;`SGLANG_NPU_ALLOC_FORCE_NAIVE=1` |
| ⑫ | apt 镜像签名 403 | Huawei ports 源 GPG | `--allow-unauthenticated -o Acquire::AllowInsecureRepositories=true` |

---

## 6. 当前修改与进展

### 6.1 本会话环境固化 —— 分支 `dsv4_one_card_dev`(5 个 commit)

目的:把「从 patch 已打好但散乱的工作树」固化成**可复现、remote 可达**的基线,供后续 graph 工作干净起步。

| commit | 内容 | 可复现性 |
|---|---|---|
| `8b0bbe6` | `[chore](sglang)`:子模块切到 **yy_repo/dsv4_release@a347a9ad5**;`.gitmodules` url → `git@github.com:wenxuewuhd/sglang-dsv4.git` | 已确认 `a347a9ad5` 在 yy_repo 上,裸 clone 可 fetch |
| `e2343bf` | `[fix](llamafile)`:取消 `iqk_mul_mat_arm82.cpp` 两行 rename 注释(坑③) | 主仓内 vendored 文件,直接生效 |
| `6b45430` | `[chore](llama.cpp)`:指针 `ac315ccc → a94e6ff`(公开 b3173);url 换公开 `ggerganov/llama.cpp` | 公开可达零鉴权;坑④ NumPy2 修复保留为 tracked patch |
| `bdd3b88` | `[chore](repo)`:跟踪 `tools/kt_dsv4_npu_patches/`(37 patch + readme,复现包);`.gitignore` 忽略 `.claude/`、`kernel_meta/`、`fusion_result.json`、`kt-kernel/kt_kernel`,并用 `!…/*.patch` 反忽略全局 `*.patch` | 复现链闭合 |
| `dc4a57f` | `[chore](tools)`:p27 launch 默认 `MODEL_PATH` → `DeepSeekV4/` 实际路径 | — |

**预期残留(非问题)**:`third_party/llama.cpp` 长期显示 `modified content` —— 即坑④ patch 的 apply 态
(`gguf-py/gguf/gguf_reader.py`),逐行等于 tracked patch,**不要 commit 进子模块**。

**子模块 remote 现状**:
- sglang → `git@github.com:wenxuewuhd/sglang-dsv4.git`(私有,SSH,需 key)
- llama.cpp → `https://github.com/ggerganov/llama.cpp.git`(公开)
- 两者的 `a347a9ad5` / `a94e6ff` 均已确认在对应 remote 可达。

### 6.2 功能进展状态

| 模块 | 状态 |
|---|---|
| Phase 0 编译期 NPU 适配(CANN aclrt 包装、ARM feature、链接) | ✅ |
| W8A8 → GGUF(BF16 + Q8_0) | ✅(Q8_0 cosine 0.9999,生产可用) |
| 单卡整网 wiring(SGLang + CPU MoE) | ✅ HTTP 200,输出连贯 |
| eager + `KT_FORCE_SYNC_SUBMIT` 功能路径 | ✅(~1.6 tok/s,保留作对照) |
| **NPU graph + ACL callback worker(任务2)** | **✅ 已闭合(2026-06-08,见 §6.3);decode `npu graph: True` ~3.6 tok/s** |
| Triton fallback / GGUF layout / chunked-prefill | ✅(坑⑩⑪⑨ 已修) |

### 6.3 ✅ graph capture 修复(2026-06-08 已闭合)

> 原专项文档 [graph_mode_fix_handoff.md](graph_mode_fix_handoff.md) 的 5 个嫌疑点经实测**均非真因**——
> 真因是两个之前未被点名的点(改动一、改动五)。下面是最终修好的全部改动与实测。

**真因(两层)**:
- **第一层(坑⑥,107030)**:`kt_ep_wrapper.py` 的 `mask_cpu_expert_routing` 里
  `gpu_experts_mask.to(topk_ids.device)` / `logical_to_gpu_index.to(...)` 是 CPU→NPU 同步 H2D。
  mask 在 `kt_expert_masks.py` 以 `device="cpu"` 建;**NPU 上 capture 前没有 eager 预热 forward**
  (CUDA 才有 `kernel_warmup`),首个 forward 即 capture,该 H2D 撞 ACL「captured-stream 禁止同步 memcpy」。
- **第二层(坑⑥b,Unsupport run graph)**:修好第一层后 capture 能过,但首请求 prefill 图重放崩
  `Unsupport run graph with different stream`。因 `mask_cpu_expert_routing` 被
  `@torch.compile(backend=get_compiler_backend())`,NPU 上该 backend 是 **torchair(max-autotune)**,
  把它编成绑定 trace-stream 的独立 NPU 子图,与外层 sglang NPU graph 在不同 stream 上 replay 冲突。

**最终改动(5 处)**:

| # | 文件 | 改动 | commit |
|---|---|---|---|
| 一(P0) | `third_party/sglang/.../kt_ep_wrapper.py` `process_weights_after_loading` | capture 前(权重加载完)把 `gpu_experts_mask`/`logical_to_gpu_index` 预搬到 `layer.w13_weight.device`;`.to()` 变 no-op,消除 107030 | sglang `456687a0f` |
| 五(P0) | 同文件 `mask_cpu_expert_routing` | **去掉 `@torch.compile`**(torchair 跨 stream),改 eager 折进外层图 | sglang `456687a0f` |
| 三(P2) | 同文件 `process_weights_after_loading` 末尾 | capture 前预订阅 ACL report stream(幂等、best-effort) | sglang `456687a0f` |
| 二(P1) | `kt-kernel/python/experts_base.py` `_wait_device` | 加 `get_is_capture_mode()` 兜底:`torch.npu.is_current_stream_capturing()` 在 capture 期若返回 False/抛异常,改查 sglang 全局 capture 标志,防误入 `synchronize()`(107027) | 父仓本次 |
| 四 | `kt-kernel/python/utils/{loader,llamafile}.py` | `KT_DUMMY_CPU_WEIGHTS` 调试开关(见 §6.5) | 父仓本次 |

> 改动二是**潜在崩溃加固**:默认 graph 路径不走 `_wait_device`(仅 `KT_FORCE_SYNC_SUBMIT=1` bypass 才触达),
> 但 `is_current_stream_capturing()` 在 torch_npu graph 下可靠性未知,故按 `_npu_use_graph_host_callback`
> 同款双重检测加固。原嫌疑的 `copy_inputs_to_cpu_buffers`/`copy_forward_output_to_device` 经核实已是
> `non_blocking=True` pinned 异步拷贝,capture-safe,无需改。

**实测闭环(NPU 1,真实 W8A8 + Q8_0 CPU MoE)**:
- Load weight 477s → capture **6.79s** 通过 → health 200。
- `tools/p27_curl_f2_prompts.sh` 四 prompt 全部连贯(fib 递归 / 监督学习 / transformer 中文 / pandas-sklearn 代码),无 NaN/感叹号/乱码。
- decode 日志 `npu graph: True`,gen throughput **3.46–3.89 tok/s**(prefill 批 `npu graph: False` 正常)。

### 6.4 其它开放项(优先级低于 graph)

| 优先级 | 任务 | 备注 |
|---|---|---|
| P1 | graph 性能调参 | `TASK_QUEUE_ENABLE=0` 对照、多 `cuda-graph-bs`、shared-expert 双 stream |
| P2 | mxfp4 原生权重 | NPU MoE 换 mxfp4,CPU 保留 GGUF,重算 `kt-num-gpu-experts` 预算 |
| P3 | EPLB 动态 hot-expert | 当前硬编码「前 N」;按 activation 频次取 top-16 |
| P4 | CPU MoE 慢加载提速 | 真实权重 GGUF 读取 ~477s(单线程 TP 切分 + 多 GB I/O);dbg 期用 §6.5 dummy 绕过,生产提速需并行化 C++ `load_weights_task`(另议) |
| – | MOE_INT8 / KML | **不做**(K920 无 SVE/i8mm) |

### 6.5 🔧 dbg 期绕过 CPU MoE 权重加载(`KT_DUMMY_CPU_WEIGHTS`)

**痛点**:真实权重每次拉起要等 **~477s**(GGUF 单线程 TP 切分 + 多 GB 磁盘 I/O,日志刷
`TP MOE layer N` / `Llamafile TP splitting`)。调 graph/capture 时反复重启,这段加载是主要时间开销。

**方案**:本会话新增环境开关 `KT_DUMMY_CPU_WEIGHTS=1`,**跳过 GGUF 磁盘读取**,只按张量元数据
(shape / ggml 量化类型 / 元素数,从 `GGUFLoader.tensor_info` O(1) 取)fabricate **同字节布局的
零 buffer**;之后 C++ `MOE(moe_config)` / `load_weights_task` 路径**完全不变**(buffer 尺寸、kernel
选择、capture 与 forward 全部忠实执行),只是权重值是 0。

实现落点:
- `kt-kernel/python/utils/loader.py`:`GGML_QUANT_SIZES` 提到模块级;新增
  `GGUFLoader.get_dummy_tensor_and_ggml_type(name)`——只读 `tensor_info`,返回
  `torch.zeros(n_bytes, uint8)` + 原 ggml 类型,不碰 mmap 数据。
- `kt-kernel/python/utils/llamafile.py` `load_weights`:开关命中时改用 dummy 加载器(layer 0 打印醒目告警)。

用法 / 边界:
```bash
# dbg：跳过慢加载，快速反复调 capture（输出乱码，仅验证“能跑通图”）
KT_DUMMY_CPU_WEIGHTS=1 NPU_DEVICE_ID=<空闲卡> bash tools/p27_launch_ds4flash_npu.sh
# 验收：去掉开关，真实权重 + tools/p27_curl_f2_prompts.sh 看连贯 + p27_cpu_moe_reference_check.py 对账
```
- ⚠️ dummy 权重输出**无意义**,严禁用于精度验收;只用于「capture / 图重放能否跑通」这类结构性调试。
- ⚠️ 当前只省**磁盘 I/O**(实测最大头);C++ 单线程 TP 切分仍跑。若实测瓶颈在切分,需进一步并行化(P4)。
- ⚠️ 生产勿长期开 `KT_DUMMY_CPU_WEIGHTS` / `KT_FORCE_SYNC_SUBMIT` / `KT_DEBUG_*` / `SGLANG_NPU_PROFILE_ENABLE`。

---

## 7. 性能数据(参考,未大规模调参)

| 项 | 值 |
|---|---|
| Graph capture 时间(实测 06-08) | 6.79 s(bs=1,真实权重);dummy 9.89 s |
| Decode 吞吐 — graph(实测 06-08) | **3.46–3.89 tok/s**(`npu graph: True`) |
| Decode 吞吐 — eager | ~1.6 tok/s |
| 模型加载 | **~100s**(43 层 MoE GGUF ~47s〔P0+P1 加速〕+ 46 shard/建模 ~54s);旧 ~9 min,见 `DeepSeek-V4-Flash_CPU权重加载加速_P0-P1.md` |
| HBM 占用(N=32) | ~16 GB expert + attention + KV |
| DRAM 占用 | ~275 GB(Q8_0)/ ~555 GB(BF16) |

---

## 8. 关键约束 / 红线

| # | 红线 | 后果 |
|---|---|---|
| R1 | 不上 SVE/BF16/I8MM 指令;march 固定 `armv8.2-a+fp16+dotprod` | SIGILL |
| R2 | C++ pybind 模块不 `#include <torch_npu/...>` | ABI 不稳 |
| R3 | ACL callback 必须专用 poller 线程 subscribe+process | 卡 sync、NPU 空闲 |
| R4 | W8A8→Q8_0 不可 reinterpret int8 块(scale 粒度不同) | 数值错但不报错 |
| R5 | 不把 `/workspace`、`/usr/local/Ascend` 等环境路径硬编码进**代码** | 换环境撞死 |
| R6 | SGLang 不 fork 整模型实现,只加分支/继承 | 升级 submodule 破坏 |
| R8 | shared_experts / router gate 不 offload,留 NPU | 路由/精度 |
| R9 | `first_k_dense_replace` 层无 256 expert,offload 要 skip(本模型=0,全 MoE) | KeyError |
| R10 | NEXTN(speculative)第一版不开 | sglang NPU NEXTN 有坑 |
| — | 勿 `pkill -f "sglang.launch_server"`(会杀掉自身 shell);按 PID 杀 | exit 1、像没运行 |

---

## 9. 命令速查

```bash
# 体检
ls third_party/sglang/python/sglang/__init__.py third_party/llama.cpp/gguf-py/gguf/__init__.py
find kt-kernel -name "kt_kernel_ext*.so"; ls /workspace/models/cache/dsv4_layer*.gguf | wc -l
npu-smi info | head -20; numactl --hardware

# 每容器
apt-get install -y libhwloc-dev

# 编译(必要时)
cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace

# 转权重(Q8_0)
/usr/local/python3.11.14/bin/python3.11 tools/batch_convert_w8a8_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --quant q8_0 --jobs 32 --verify-sample 3

# 预检 + 拉起(graph 默认 / eager 回退)
bash tools/p27_e2e_preflight.sh
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 NPU_DEVICE_ID=0 bash tools/p27_launch_ds4flash_npu.sh
MODEL_PATH=… NPU_DEVICE_ID=0 KT_FORCE_SYNC_SUBMIT=1 EXTRA_FLAGS="--disable-cuda-graph" bash tools/p27_launch_ds4flash_npu.sh

# CPU MoE 离线对账(定位数值问题)
PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel" /usr/local/python3.11.14/bin/python3.11 \
  tools/p27_cpu_moe_reference_check.py --w8a8 /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --gguf /workspace/models/cache/dsv4_layer3.gguf --layer-idx 3 --method LLAMAFILE
```

---

*整合自 4 份来源,以本文为现行总纲。维护:`dsv4_one_card_dev`。graph capture 已闭合(§6.3);
后续主攻 §6.4 性能调参 / CPU MoE 慢加载提速(dbg 期可用 §6.5 `KT_DUMMY_CPU_WEIGHTS` 绕过)。*
</content>
</invoke>
