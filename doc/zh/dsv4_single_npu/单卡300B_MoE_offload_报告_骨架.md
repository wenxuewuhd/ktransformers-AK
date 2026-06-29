# 报告骨架：单机单卡 MoE Offload 运行 300B（DeepSeek V4-Flash）

> 状态：**骨架阶段**（只定结构，不写正文）。配套文件：`待复核数字清单.md`
> 读者：公司技术论坛 ｜ 定位：展示技术方案 + 本工程作为实践样例 + roadmap ｜ 只讲方案与结果
> 核心论点：方案设计本身 + 这个工程是一个可参照的实践样例

---

## 1. 摘要
- 一句话核心结论 + 3~4 个硬指标（decode tok/s、长 prompt prefill 加速、精度对齐、内存占用）
- 最后写

## 2. 背景与方案设计
### 2.1 背景与挑战
- 矛盾：仅 routed 专家 277B 参数，4-bit≈138.5GB = 64GB HBM 的 2×+（A8/G 已核）
- 转机：MoE 稀疏激活——每 token 只激活 top-6/256 专家，活跃权重 ≪ 总权重
- 核心洞察：专家是参数大头但每步只用一小撮 → 放慢速大容量 DDR、按需取用（offload 物理基础）

### 2.2 业界已有方案对比　【已撰写正文】

> 数据来源见配套文件《…业界对比研究notes.md》。decode 吞吐多为单用户、二手源（新闻/博客）给出的近似区间，正式发布前以标注来源为准；本节数字均为单卡/单机、低并发场景。

在"用有限的快内存（显存/HBM）跑超大 MoE"这件事上，业界已形成两条主要技术路线：

- **路线一：独立加速卡 + CPU/DDR offload。** 快内存（GPU 显存 / NPU HBM）只装常驻部分（attention、共享专家、最热路由专家），把参数大头的路由专家放到容量大、成本低的 CPU 内存，按需参与计算。代表：ktransformers、llama.cpp 的 `--override-tensor`。**本方案属于这一路线（NPU 实例）。**
- **路线二：统一内存（UMA）整模型驻留。** SoC 上 CPU/GPU 共享一大块内存，把整个（量化后）模型一次性装入，不存在"搬运"概念。代表：Apple Mac（M 系列）、AMD Strix Halo。

两条路线的取舍本质即 2.3.3 将展开的"算力搬到数据 vs 数据搬到算力"：路线一在独立 HBM + 慢互联(PCIe) 下让 CPU 就地算冷专家、避开互联墙；路线二因内存物理统一、搬运成本≈0，但受限于 UMA 自身的带宽与容量上限。

#### 横向对比

| 平台/方案 | 快内存 | 带宽 | offload 策略 | 量化 | 最大 MoE | decode(单用户) | 主瓶颈 |
|---|---|---|---|---|---|---|---|
| **本方案(910B3+DDR)** | 64GB HBM | ~1.2TB/s〔待核 E5〕 | expert 级，A/B 混合 | MXFP4 4bit | **DSV4-Flash 300B**(284B/A13B) | **~18.9**〔待核 B1〕 | CPU DDR/PCIe |
| ktransformers | 4090D 24GB ＋1TB DDR5 | — | 路由专家→CPU 就地算 | 4bit/AMX | V3/R1 671B | 14(v0.3≈13.69) | CPU 算力/DDR+同步开销 |
| NVIDIA + llama.cpp `-ot` | 4090 24GB ＋DDR | — | 路由专家→CPU,shared 留卡 | GGUF Q4 | V3 671B | ~12(ctx<10k) | CPU DDR |
| Apple Mac M3 Ultra | 512GB UMA | 819GB/s | 无(整模型入 UMA) | 4bit | R1 671B | ~16–20 | UMA 带宽 |
| AMD Strix Halo(395) | 128GB UMA | ~256标称/~120有效 | 无 | 1.6–4bit | ~120B 级 | 同模型仅 1–2(见③) | UMA 容量+带宽 |

来源：ktransformers [SOSP25 论文](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)、[tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)；NV [HF blog](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)；Mac [Apple specs](https://www.apple.com/mac-studio/specs/)、[MacRumors](https://www.macrumors.com/2025/03/17/apples-m3-ultra-runs-deepseek-r1-efficiently/)；AMD [Strix Halo V4-Flash](https://tinycomputers.io/posts/running-deepseek-v4-flash-on-amd-strix-halo.html)。

#### 三个差异化定位

**① vs ktransformers / NVIDIA 单卡（同路线对照）。** 三者同属"路线一 + compute-on-CPU"。差异在硬件实例与优化深度：他们在 GPU+x86，本方案在 **NPU+ARM(K920)**，并叠加热专家动态常驻(A/B 混合)、同源量化(depool)、流式 prefill 等（第 3 章）。ktransformers SOSP25 论文将瓶颈明确归为"CPU 算力上限 + CPU-GPU 同步开销"——正是本方案 side-stream 与 CPU kernel 优化所针对的方向。单用户 decode 上本方案 ~18.9 高于其 ~14。

**② vs Apple Mac（UMA 暴力流）。** M3 Ultra 用 **512GB** 大 UMA 把 671B 4-bit 整模型装下，无需 offload 技巧即得 ~16–20 tok/s——但代价是 **8× 于本方案的快内存容量**。本方案以 **64GB HBM + offload** 达到 ~18.9，在快内存小一个数量级的前提下逼近 819GB/s 的 M3 Ultra，靠的是 HBM 高带宽 + offload 调度，而非堆内存。

**③ vs AMD Strix Halo —— 同模型铁证。** 已有公开测试在 Strix Halo（128GB UMA、有效带宽 ~120GB/s）上运行**同一个 DeepSeek-V4-Flash**：因 128GB 装不下，被迫量化到 **IQ1_S-XL(1.6-bit)** 才塞进 58GB，实测仅 **1–2 tok/s**（理论带宽上限 ~46 t/s，被开销拖垮）。本方案在 64GB HBM + offload + 4-bit 下达 ~18.9 tok/s——**同一模型，快一个数量级，且无需牺牲到 1.6-bit 精度**。这组对照最直观地说明 UMA 的带宽与容量是"小快内存"路线的双重墙。
（附：该测试独立给出 DSV4-Flash = 284B total / 13B activated，与本报告据 config 推算的 routed 277B 相互印证。）

#### 业界趋势印证

本方案设计与业界正在收敛的方向一致：
- **算法层趋同**：vLLM 的 MoE offload 设计（CPU pinned 放专家 + GPU 缓存最热 + LFRU 驱逐 + 跨层预测）与本方案热专家动态常驻 + 预测预取(3.1)几乎同构；PreScope / SpecOffload 等近期工作亦专攻 MoE offload 预取。
- **路线B 的适用边界**：FlexGen、DeepSpeed ZeRO-Inference 代表的"纯权重流式(data-to-compute)"靠超大 batch 摊销搬运（FlexGen OPT-175B batch 144 才 1 tok/s），是**吞吐导向、单用户延迟极差**——从反面印证 2.3.3 中"低并发 decode 必须走 compute-to-data 或 A/B 混合"。
- **硬件方向**：TriMoE 等探索 GPU + AMX CPU + DIMM 近数据处理(NDP) 的 MoE offload 形态，为第 6 章"下一代硬件启发"提供业界参照。

> 小结：本方案在路线一(offload)内以 NPU 实例 + 更深的放置/调度优化取得竞争力；相对路线二(UMA)，则在小一个数量级的快内存下用高带宽 HBM + offload 跑赢容量受限的 UMA、逼近大容量 UMA。下一节展开方案总体架构与这一取舍的技术依据。

### 2.3 方案总体设计（本章重点，搭骨架）
- 2.3.1 模型切分：attention+共享专家(1)+router/dense → NPU HBM 常驻；routed 专家(256) → CPU/DDR offload
- 2.3.2 内存层次与预算：HBM(64GB,快)/DDR(大,慢)；静态〔待核 A2〕，剩余〔待核 A3〕给 KV+slot+常驻池
- 2.3.3 异构执行模型(hybrid)：一次 forward 数据流（冷专家@CPU/DDR、热专家@NPU/HBM）+ decode/prefill 两路概览图（高层方框图）
  - **两种 offload 哲学对照（本节核心论证）**：
    - 方案A 算力搬到数据(compute-to-data)：权重躺 DDR，**CPU 就地算**，只回传激活(KB级) ← 我们的冷专家路径
    - 方案B 数据搬到算力(data-to-compute)：miss 时**stall**，权重 DDR→HBM 搬过去，**NPU 一次过算**
    - 对照维度：谁算(CPU/NPU)｜跨总线搬什么(激活KB / 权重MB)｜**瓶颈资源(CPU本地DDR带宽 / CPU↔NPU互联PCIe)**｜延迟(无stall可overlap / miss即stall)｜批量摊销(不摊 / transfer可被高复用摊薄)｜算力上限(CPU弱 / NPU强+HBM高带宽)
    - 量化论据(已确认config)：单专家4-bit=12.58MB；每token 6×43=258专家调用→最坏全miss搬 **3.17GB/token**；18tok/s 需 ~57GB/s 跨PCIe >> Gen4~32GB/s〔待核E3〕→ **纯B在"独立HBM+PCIe"上 decode 不可行**；CPU本地DDR~614GB/s〔待核E2〕不跨PCIe→A可行
    - ★核心洞察：**我们的方案=A/B混合**——热专家走B(常驻HBM、一次搬多token复用=已摊销，绕开stall)，冷专家走A(CPU就地算，避PCIe墙)；纯B=全走B+每次miss同步stall
    - 适用分界：纯B划算前提=互联快或无互联→典型**UMA统一内存(Mac/AMD Strix Halo)搬运≈0**(2.2分野所在)；独立HBM+PCIe上A或A/B混合才是decode解；预取roadmap=想让B可行(提前搬藏stall)但带宽天花板〔D5〕证明逃不开PCIe墙→喂第6章
- 2.3.4 量化方案：专家 MXFP4(4-bit,GGML=39) 同源；为什么选 MXFP4（精度 cos〔待核 F4〕/带宽/CPU-NPU 一致）
- 2.3.5 承上启下 → 第三章技术点

## 3. 方案技术点（方法层，可迁移；落地细节留第4章）
> 每节模板：问题本质 → 设计思想 → 方法要点 →（效果定性，数字留第5章）

- **3.0 开篇**：两种 offload 哲学（引 2.3.3）+ 瓶颈模型（decode=DDR带宽bound；prefill=全专家激活/容量）；后面每个技术点攻一个瓶颈
- **3.1 放置与命中率（算法核心，本章重点）**
  - 3.1.1 热专家池设计【已落地】：池容量、放置策略（静态prefix vs 动态top-K频率）、更新机制、⚠️mask-remap陷阱（remap须连mask/索引一起搬，否则weight-region flush拖NSA）
  - 3.1.2 专家预测与预取【探索/roadmap】：第L层hidden预测L+1激活、可行性闸门(69%命中)、per-token vs per-request、带宽天花板(~48-50%)→喂第6章
- **3.2 调度 — Side Stream**（计算/传输重叠）【已落地】：side + shared-expert stream 并发；增益∝1/串行地板，与命中率相乘
- **3.3 长序列 — Prefill Stream**（流式预填充）【已落地】：专家逐层流式过预留HBM slot分时复用；NZ转换必过HBM；解全专家激活OOM
- **3.4 内存 — 同源量化(depool) + 权重去重(dedup)**【已落地】：同源消两套量化漂移；dedup复用CPU已mmap权重去冗余pinned池
- **3.5 正确性 — 异步混合执行竞态根治**【已落地】：同步submit+无条件wait_device+保留subscribe；force-sync关仍又对又快；decode快靠subscribe非异步submit
- **3.6（可选）CPU 算子 — 访存优化**（K920 软预取 / MXFP4 repack）【待定：独立成节 or 并入第4章】
> 节内：side-stream/prefill/hot-dynamic（用户点的三个）全保留；hot-dynamic升格为"放置与命中率"算法大节，预测预取并入

## 4. 基于 DSV4-Flash 的实践（落地层）
- 4.1 硬件架构解构（910B3 NPU + CPU/DDR 拓扑、内存带宽层次）
- 4.2 各技术点在本硬件上的具体实现接法（映射第3章方法 → 这块卡）
- 4.3 Roofline 分解（prefill / decode 两条，瓶颈不同）
- 4.4 当前离 roofline 的 gap + 逐项归因（同步开销/命中率/CPU kernel 效率…），每项指向第6章一条 roadmap

## 5. 结果评估
- 5.1 性能：decode 吞吐〔待核 B1〕、TTFT〔待核 C2〕、长 prompt prefill 加速〔待核 C1〕
- 5.2 精度：GPQA ⚠️三条 off 路径勿混报（〔待核 F1/F2/F3〕，写时标清配置）
- 5.3 内存：占用与节省（dedup 省〔待核 A6〕）
- 注：所有性能数须暖机后 median（见清单 B 区 ⚠️）

## 6. 未来 Roadmap + 下一代硬件启发
- 6.1 软件侧演进：doorbell/ExternalEvent 降同步开销、原生 MXFP4 NPU、NSA 选块改进、专家预取
- 6.2 硬件侧启发：从 roofline gap 反推下一代硬件该补什么（DDR 带宽 / 互联 / 片上同步原语 / 原生低比特）
