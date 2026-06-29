# 业界对比研究 notes（2.2 章原始素材 + 来源）

> 来源说明：deep-research 后台工作流于 6-26 被回收（卡在 Search 阶段），改为前台 WebSearch/WebFetch 手动调研。
> 数字多来自二手源（新闻/博客/官方 spec），decode tok/s 多为单用户、给出区间；正式引用须标"二手源/约"。
> 日期：2026-06-29

## 一、各平台数据点

### 1. ktransformers（CPU+GPU 异构 offload，我们的同源参照）
- 硬件：RTX **4090D 24GB** GDDR6X + Xeon Gold 6454S / **1TB DDR5-4800**（2 NUMA）
- offload：routed experts → CPU 就地算（compute-on-CPU，方案A）；always-active 留 GPU
- 量化：4-bit GGUF / AMX int8
- 模型：DeepSeek-V3/R1 **671B**（A37B）
- 吞吐：**prefill 286 tok/s、decode 14 tok/s**（vs llama.cpp 2×32核 10.31 tok/s，宣称 27.79× 加速）
- 瓶颈：CPU 本地 DDR 带宽
- 源：[TechNode](https://technode.com/2025/02/17/tsinghua-universitys-ktransformers-enables-full-powered-deepseek-r1-with-low-cost-graphics-card/)、[SOSP25 论文](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)、[GitHub tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)

### 2. NVIDIA 单卡 + llama.cpp `--override-tensor`
- 硬件：单 **RTX 4090 24GB** + CPU DDR
- offload：routed expert FFN → CPU（compute-on-CPU，方案A），shared expert + dense 留 GPU
- 量化：GGUF Q4 / ubergarm R4
- 模型：DeepSeek **V3 671B**（A37B）
- 吞吐：**~12 output tok/s @ context <10000**（单卡4090，全部专家除shared在CPU）
- 瓶颈：CPU 本地 DDR 带宽
- 源：[HF blog (Doctor-Shotgun)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)、[gist guide](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)

### 3. Apple Mac M3 Ultra（UMA 统一内存，容量够→不 offload）
- 硬件：**512GB LPDDR5x 统一内存，819 GB/s** 带宽；60核GPU/32核NPU
- offload：**无**——671B 4-bit（~404GB）整模型装进 UMA，"DDR=HBM"搬运≈0（天然方案B）
  - 需手动 allocate ~448GB VRAM
- 量化：4-bit（MLX / GGUF）
- 模型：DeepSeek-R1 **671B**
- 吞吐：**~16–20 tok/s**（多源区间：16-18 / 17-18 / 20，<200W）
- 瓶颈：UMA 带宽（819 GB/s）
- 源：[Apple specs](https://www.apple.com/mac-studio/specs/)、[MacRumors](https://www.macrumors.com/2025/03/17/apples-m3-ultra-runs-deepseek-r1-efficiently/)、[VentureBeat](https://venturebeat.com/ai/deepseek-v3-now-runs-at-20-tokens-per-second-on-mac-studio-and-thats-a-nightmare-for-openai)、[Slashdot](https://apple.slashdot.org/story/25/03/25/2054214/deepseek-v3-now-runs-at-20-tokens-per-second-on-mac-studio)

### 4. AMD Strix Halo / Ryzen AI Max+ 395（UMA，但容量/带宽受限）
- 硬件：**128GB LPDDR5X-8000，256-bit**，标称 ~256 GB/s（实测有效 ~120 GB/s 共享）；Radeon 8060S 40CU RDNA3.5
- offload：UMA，无独立显存 offload
- 量化：1.6-bit ~ 4-bit
- 可跑：~120B 级 MoE 舒适（GPT-OSS 120B **55 t/s**、Qwen3-30B-A3B **100 t/s**）；671B 装不下
- ★**同模型对照**：跑 **DeepSeek-V4-Flash**（284B total / 13B activated）——为塞进 128GB 被迫量化到 **IQ1_S-XL（1.6bit，58GB）**，实测仅 **1-2 tok/s**（理论带宽上限 ~46 t/s，overhead 拖垮）
- 瓶颈：UMA 容量 + 带宽
- 源：[AMD V4-Flash on Strix Halo](https://tinycomputers.io/posts/running-deepseek-v4-flash-on-amd-strix-halo.html)、[runaihome](https://runaihome.com/blog/ryzen-ai-max-395-strix-halo-local-llm-2026/)、[AMD trillion-param cluster](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html)

### 5. 我们的方案（Ascend 910B3 + DDR offload）
- 硬件：单卡 **910B3 64GB HBM**（HBM3e，~1.2TB/s 待核 E5）+ CPU(K920) DDR
- offload：**expert-level，A/B 混合**——热专家常驻 HBM(NPU算)、冷专家 CPU 就地算
- 量化：**MXFP4 4-bit** 同源
- 模型：**DeepSeek-V4-Flash 300B（284B/A13B）**
- 吞吐：decode **~18.9 tok/s**〔待核 B1〕、prefill 流式 4096 14s〔待核 C1〕
- 瓶颈：CPU DDR 带宽 / CPU↔NPU 互联

## 二、★同模型铁证（DSV4-Flash 284B/A13B 的独立交叉确认）
- AMD 文章独立给出 DSV4-Flash = **284B total / 13B activated**，与我们 config 推算一致（routed 277B + shared/attn）。
- **Strix Halo（128GB UMA）跑同一个 DSV4-Flash：被迫 1.6-bit、仅 1-2 tok/s**；我们 64GB HBM + offload + 4-bit → ~18.9 tok/s。
- → 直接坐实 2.3.3 论点：UMA 带宽是墙；独立高带宽 HBM + offload 在"小快内存"上反而赢。

## 三、关键带宽坐标（连接 2.3.3 / 第4章 roofline）
| 内存类型 | 带宽 | 平台 |
|---|---|---|
| 独立 HBM3e | ~1.2 TB/s〔待核 E5〕 | 我们 910B3 |
| GDDR6X(4090) | ~1 TB/s | ktransformers/NV（但 offload 部分受 PCIe + CPU DDR 限） |
| UMA LPDDR5x | 819 GB/s | Mac M3 Ultra 512GB |
| UMA LPDDR5X | ~256(标称)/~120(有效) GB/s | AMD Strix Halo 128GB |
| CPU DDR5-4800(server) | ~数百 GB/s 聚合 | ktransformers/NV CPU 侧 |

## 四之补（top-up 轮）：强化多章节的新发现

### (a) 补全 NV 栈：vLLM / TensorRT-LLM / 学术系统
- **vLLM MoE offload（RFC #38256）**：★"expert weights 放 CPU pinned memory + 固定大小 GPU cache 装最热专家，**LFRU 驱逐 + cross-layer prediction** 减 miss" —— **与我们 hot-dynamic 常驻池 + 预测预取几乎同构**，独立印证我们的算法方向。源：[vllm#38256](https://github.com/vllm-project/vllm/issues/38256)
- **TensorRT-LLM**：NV 专用，Hopper/Blackwell 上服务 DeepSeek 最稳，自定义 kernel + 投机；偏大显存，非 offload 路线。
- **MoE-Gen（arxiv 2503.09716）**：module-based batching，单 GPU，decode 吞吐 **31×**、整体 7.7-11×——靠批量摊销（throughput 向）。
- **TriMoE（arxiv 2603.01058，2026）**：★"GPU + AMX CPU + **DIMM-NDP（近数据处理）** offload 做高吞吐 MoE" —— **直接喂第6章下一代硬件启发**（近内存计算）。
- **PreScope（arxiv 2509.23638）/ SpecOffload（2505.10259）**：MoE offload 的**预取**专题 —— 喂 3.1.2 预取 roadmap 的学术先例。

### (b) ★方案B（权重流式 data-to-compute）的真实参照 —— 补强 2.3.3
- **FlexGen（arxiv 2303.06865）**：把权重+KV offload 到 CPU/disk，**靠超大 batch 摊销搬运**；OPT-175B 单 16GB GPU → batch 144 才到 **1 tok/s**；比 DeepSpeed ZeRO-Inference/Accelerate 高 69×（**纯吞吐向，单用户延迟极差**）。
- **ZeRO-Inference（DeepSpeed）**：offload **全部**权重流式喂 GPU，大 batch 优于 partial offload；吞吐向。
- → ★**坐实 2.3.3**：纯方案B（权重流式到算力）= **throughput/大batch 才划算，单用户 decode 延迟极差**（FlexGen 1 tok/s）。我们的 decode/低并发场景必须走 A 或 A/B 混合。

### (c) ktransformers 硬源（SOSP25 论文，替换新闻二手数）
- **加速比**：prefill **4.62–19.74×**、decode **1.25–4.09×**（vs 现有系统）。
- **核心优化**：AMX tiling-aware layout + cache-optimized AMX kernel + AVX-512 动态回退 + **异步 CPU-GPU 调度**。
- ★**论文自陈瓶颈 = "CPU 算力上限 + CPU-GPU 同步开销"** —— 正是我们 side-stream（同步开销）和 K920 CPU kernel（算力）所攻击的，方向完全对齐。
- **v0.3**：AMX + selective expert；prefill 3.45×(vs v0.2)/27.79×(vs llama.cpp)；decode 6-expert selective **13.69 tok/s**(vs llama.cpp 4.51)。
- 源：[SOSP25 论文](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)、[ACM DOI](https://dl.acm.org/doi/10.1145/3731569.3764843)、[LMSYS sglang+kt blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/)（与我们 sglang+kt 同栈，可深挖）

## 五、待核 / 存疑
- E5（新增）：910B3 HBM 带宽（源说 910B3=HBM3e 1.2TB/s，但有 400GB/s 基础变体的冲突信息）→ 查我们卡实际 spec
- E3：CPU↔NPU PCIe 代数/带宽
- 各 decode tok/s 多为二手源单用户区间，正式引用标"约/二手"
- ktransformers/NV 的 "GDDR6X ~1TB/s" 对 offload 无意义（瓶颈在 CPU 侧/PCIe），表里要注明
