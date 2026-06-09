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
> 维护分支:`dsv4_one_card_dev`。最后更新:2026-06-08。

---

## 0. 一句话现状

单卡 910B + K920 的 DeepSeek-V4-Flash(W8A8)推理**已可端到端拉起并输出连贯文本**。
当前生产配置:**Q8_0 GGUF(~275 GiB)+ CPU MoE offload + NPU attention**。
**唯一未闭合的关键项**:NPU graph capture 路径(性能路径,~3.6 tok/s)在最新 from-0 复现里仍会崩
(`aclrtMemcpy 107030`),eager 回退(~1.6 tok/s)功能正确 —— 这是下一步要攻的点(§6.3)。

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
- 8 个 NUMA worker pool(每 NUMA 24 线程),NEON SDOT 内核。
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
| ⑥ | **graph 捕获崩 `aclrtMemcpy 107030`** | capture 期 KT 同步拷贝撞 NPU graph capture 限制 | **未闭合**;eager 回退 `--disable-cuda-graph`(§6.3) |
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
| eager + `KT_FORCE_SYNC_SUBMIT` 功能路径 | ✅(~1.6 tok/s) |
| NPU graph + ACL callback worker(任务2,commit `b31d349`) | ⚠️ 已实现且为 launch **默认**,但见 §6.3 |
| Triton fallback / GGUF layout / chunked-prefill | ✅(坑⑩⑪⑨ 已修) |

### 6.3 🎯 下一步:graph capture 修复(当前主攻)

> **专项作战文档**:[doc/zh/graph_mode_fix_handoff.md](doc/zh/graph_mode_fix_handoff.md) —— 含完整代码地图(path:line)、
> 5 个待证伪嫌疑点、验证手段,由 graph 专项 session 维护。**细节与进度以该文为准**;本节只给概要。

**状态冲突(需正视)**:Ascend Handoff(05-19)称 graph(任务2)已验证、launch 默认 graph-on;
但最新 from-0 复现仍在 capture 崩 `aclrtMemcpy 107030`(不允许在 captured-stream 上同步 memcpy),
eager 回退功能正确。**第一步必须复跑默认 graph、抓"当前"崩在哪一行**(`b31d349` 合入后旧栈未必成立)。

**根因方向**(坑⑥⑦同源 = KT 同步拷贝):graph-on 时 capture 期 KT submit 的 sync copy 撞限制 → crash;
eager 时 async 未 flush → 全零乱码(需 `KT_FORCE_SYNC_SUBMIT=1`)。头号嫌疑:
`kt_ep_wrapper.py:452 copy_inputs_to_cpu_buffers` 的 D2H 是否 blocking;
`experts_base.py:64 _wait_device` 的 capture 保护是否真生效(取决于 capture 期
`torch.npu.is_current_stream_capturing()` 是否返回 True)。

**改动落点**:`kt-kernel/cpu_backend/ascend_callback_worker.*`、`kt-kernel/python/experts_base.py`、
`third_party/sglang/.../kt_ep_wrapper.py`(sglang 是子模块,改动需在子模块 commit + 同步父仓指针)。

### 6.4 其它开放项(优先级低于 graph)

| 优先级 | 任务 | 备注 |
|---|---|---|
| P1 | graph 性能调参 | `TASK_QUEUE_ENABLE=0` 对照、多 `cuda-graph-bs`、shared-expert 双 stream |
| P2 | mxfp4 原生权重 | NPU MoE 换 mxfp4,CPU 保留 GGUF,重算 `kt-num-gpu-experts` 预算 |
| P3 | EPLB 动态 hot-expert | 当前硬编码「前 N」;按 activation 频次取 top-16 |
| – | MOE_INT8 / KML | **不做**(K920 无 SVE/i8mm) |

---

## 7. 性能数据(参考,未大规模调参)

| 项 | 值 |
|---|---|
| Graph capture 时间(报告值) | 7–11 s(bs=1) |
| Decode 吞吐 — graph(基线) | ~3.6 tok/s |
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

*整合自 4 份来源,以本文为现行总纲。维护:`dsv4_one_card_dev`。下一步主攻 §6.3 graph capture。*
</content>
</invoke>
