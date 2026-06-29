# 报告待复核数字清单（DSV4-flash 单卡 MoE offload）

> 来源 = 我从历史实测记忆引用的值；正式写入前须由 @用户 或实测复核。
> 状态：☐ 待核 / ☑ 已核

## A. 内存 / 容量（第2章 方案设计、第5章 结果）

| # | 数字 | 用途/章节 | 来源（记忆） | 复核方式 | 状态 |
|---|------|----------|-------------|----------|------|
| A1 | 单卡 HBM **64GB**（**910B3** 确认） | 2.3.2 预算 | **npu-smi 实机**（65536MB/片，8卡机跑单卡，PCI 0x19E5:0xD802） | 已实机确认 | ☑ |
| A2 | 静态占用 44.7GB（模型 42.32GB） | 2.3.2 / 5 内存 | hbm-budget-prefill-vs-decode | 服务启动后读实际占用 | ☐ |
| A3 | 剩余 ~15.8GB 给 KV+slot+常驻池 | 2.3.2 | 同上(推算) | A1-A2 推算复核 | ☐ |
| A4 | 专家容量上限 40/层（context 65536） | 2.3.2 / 4 硬件 | dsv4-npu-expert-capacity | 实测最大 KT_NUM_GPU_EXPERTS | ☐ |
| A5 | 流式 slot 6.44GB（256 专家 NZ，8-bit） | 3.3 prefill stream | depool-streaming-oom | 实测 slot 预留量 | ☐ |
| A6 | GGUF dedup 省 137G（used 326→189GB） | 3.4 内存 / 5 | gguf-dedup-saves-137g | system free 前后对比（非 RSS） | ☐ |
| A7 | depool Q8_0 6.8G/层 vs mxfp4 3.4G/层 | 3.4 / 2.3.4 量化 | depool-decode-needs-mxfp4 | 实测每层字节数 | ☐ |
| A8 | ☑ 已重算：**仅 routed 专家就 277B 参数**（256专家×43层×3×4096×2048）；4-bit≈**138.5GB**、8-bit≈277GB | 2.1 背景 | config 估算 | 已按真实架构算（gate+up+down 三矩阵） | ☑ |
| A8' | 推论：**单 routed 专家权重 4-bit 138.5GB 已是 64GB HBM 的 2×+** → 这是"必须 offload"的核心立论 | 2.1 矛盾 | A8 | — | ☑ |

## B. Decode 性能（第5章 结果，第3章定性）

| # | 数字 | 用途/章节 | 来源 | 复核方式 | 状态 |
|---|------|----------|------|----------|------|
| B1 | depool dynamic decode 18.9 tok/s | 5 性能 | depool-dynamic-correct-convert-folded | 暖机后 median 实测 | ☐ |
| B2 | 静态常驻 decode 18，动态修mask后 16.4→max20.2(+37%) | 3.2 hot dynamic | dynamic-resident-decode-slow | 暖机 median，区分峰值 | ☐ |
| B3 | side-stream +7%（63.2→58.8ms/tok，16→17.2tok/s） | 3.1 side stream | sidestream-sharedstream-merged-trunk | 同窗口配对实测 | ☐ |
| B4 | off_cpu 16.8ms(mxfp4) vs 22ms(Q8_0) | 3.4 量化 | depool-decode-needs-mxfp4 | 暖机后实测 | ☐ |
| B5 | dedup decode 18.2 tok/s | 5 | dedup-prefill-slow | 暖机 median | ☐ |
| ⚠️ | decode 数必须暖机后测（冷测约 ½ 偏低） | 全部 B | decode-baseline-needs-warmup | 测前打 ~10 丢弃请求 | ☐ |

## C. Prefill / TTFT（第5章，第3章定性）

| # | 数字 | 用途/章节 | 来源 | 复核方式 | 状态 |
|---|------|----------|------|----------|------|
| C1 | 流式 prefill 4096：14s vs hybrid 137s（~8×） | 3.3 prefill stream | streaming-prefill-working-prod | 实测两路对比 | ☐ |
| C2 | TTFT short 436ms（depool dynamic，无108s切换） | 5 / 3.2 | depool-dynamic-correct-convert-folded | 实测短 prompt TTFT | ☐ |
| C3 | dedup 长 prefill 95s→33s（_par_copy 16线程） | 3.4 / 5 | dedup-prefill-slow-is-singlethread | 实测 | ☐ |

## D. 命中率 / 放置（第3章 hot pool + 预取算法）

| # | 数字 | 用途/章节 | 来源 | 复核方式 | 状态 |
|---|------|----------|------|----------|------|
| D1 | 静态 prefix-32 只接 ~13% 激活 | 3.2 / 3.x 算法 | static-prefix-placement-is-random | 实测命中率 | ☐ |
| D2 | 动态 top-K 命中 ~3×（~43%） | 3.2 | dynamic-resident-decode-slow | 实测命中率 | ☐ |
| D3 | 预取 h_L 预测 L+1 top-6 命中 69%（4.15/6，depth-1） | 3.x 预取算法 / 6 roadmap | prefetch-feasibility-gate-passed | 实测 | ☐ |
| D4 | per-request 换集 N32 命中 68%(<80%门槛) | 6 roadmap | prefetch-A-to-C-decision-gate | 实测 | ☐ |
| D5 | 预取 80% 需 90-128 专家/层=95-139GB>64GB；本卡天花板~48-50% | 6 roadmap 硬件启发 | prefetch-per-token-bandwidth | 带宽推算复核 | ☐ |

## E. CPU 算子 / 带宽（第2.3.3 哲学对照 / 第3章可选 / 第4章 roofline）

| # | 数字 | 用途/章节 | 来源 | 复核方式 | 状态 |
|---|------|----------|------|----------|------|
| E1 | K920 GEMV 软预取 0.9→3.2 GB/s/核（kernel 2.4×） | 3.6 / 4 roofline | k920-hw-prefetcher-needs-sw-prefetch | micro-bench 复核 | ☐ |
| E3 | CPU↔NPU 互联（PCIe）带宽：假设 Gen4 ~32GB/s | **2.3.3 两哲学对照（纯B不可行论据）** | 待确认（910B3 PCIe 代数/lane） | 查 910B3 PCIe spec / lspci | ☐ |
| E4 | 单专家 4-bit=12.58MB；全miss搬运 3.17GB/token；18tok/s需~57GB/s | 2.3.3 论据 | config 推算（已确认架构） | 公式复核（258专家×12.58MB） | ☑ |
| E5 | 我们 910B3 HBM 带宽数值（910B3 芯片已实机确认，但 npu-smi 不暴露带宽）：网传 ~1.2TB/s 待证 | **2.2 对比表 / 4 roofline** | 二手源 | 官方 spec / HBM 带宽 micro-bench（warmup后） | ☐ |
| E2 | DDR 真 spec 614GB/s（3/4 通道 DDR4-3200） | 4 roofline | k920-gemv 记忆 | 查 K920 内存配置 spec | ☐ |

## F. 精度（第5章 — ⚠️三条路径勿混报）

| # | 数字 | 用途/章节 | 来源 | 复核方式 | 状态 |
|---|------|----------|------|----------|------|
| F1 | GPQA off baseline-fix 全配置 75.25%（prefix-32+force-sync关+非depool+mxfp4） | 5 精度 | gpqa-baseline-fix-config-7525 | 复跑确认配置 | ☐ |
| F2 | PR off 基线 73.23%（我们 +2.02pp） | 5 对标 | 同上 | 确认对标口径 | ☐ |
| F3 | depool/full-prod off 72.22%（vs PR −1pp 对齐） | 5 | gpqa-off-aligned-on-blocked | 复跑 | ☐ |
| F4 | MXFP4 CPU MoE cosine 0.99994 | 2.3.4 / 3.4 | mxfp4-cpu-moe-validated | 复跑对账 | ☐ |
| ⚠️ | 三条 off 路径数值各异，写报告必须标清配置 | 全部 F | gpqa-off-aligned/baseline-fix | — | ☐ |

## G. 量化常量 / 事实（第2/3章）—— ☑ 全部已核（config + 代码）

权威来源：`/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8/config.json`（launch 脚本默认加载）+ 代码常量。

| # | 数字 | 用途/章节 | 来源 | 复核方式 | 状态 |
|---|------|----------|------|----------|------|
| G1 | GGML_TYPE_MXFP4 = **39** | 2.3.4 | loader.py:51 / ggml.h:380 / constants.py:906 一致 | 查代码常量 | ☑ |
| G2 | 每 token 激活 routed 专家 top-K = **6**（num_experts_per_tok = n_activated_experts = 6） | 2.3.3 数据流 | config | 查 config | ☑ |
| G3 | 每层 routed 专家总数 = **256**（n_routed_experts） | 2.3.x | config | 查 config | ☑ |

### G+ 顺带确认的权威架构参数（第2章 2.3 可直接引用，无需再核）
| 参数 | 值 | 备注 |
|------|----|----|
| model_type | deepseek_v4 | — |
| num_hidden_layers | **43** | — |
| first_k_dense_replace | **0** | ⚠️**全部 43 层都是 MoE 层**（无 dense 层），每层都有 256 专家 |
| n_shared_experts | **1** | 共享专家 1 个/层（常驻 NPU） |
| hidden_size | 4096 | — |
| moe_intermediate_size | 2048 | 专家 FFN 中间维 |
| num_attention_heads | 64 | — |
| index_topk | **512** | NSA 稀疏选块 top-512（长上下文相关） |
| num_nextn_predict_layers | 1 | MTP 1 层 |
| q_lora_rank | 1024 | MLA |
| vocab_size | 129280 | — |
| 量化 | W8A8（compressed-tensors）：权重 int8/channel/对称，激活 int8/per-token/dynamic | 这是基线 ckpt 量化 |

---
**备注**：业界对比表的数字（NV/Mac/AMD/ktransformers）由 deep-research 单独产出并带引用，不在此清单。
