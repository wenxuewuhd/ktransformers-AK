# NPU MoE Microbench（独立小工程）

> **本 bench 对齐 `DSv4-Flash-W8A8` ckpt 的实际部署（int8 W8A8），不对齐论文 FP4×FP8**。
> 详见 "已知偏离与局限" 段（外部引用请先读）。

在 **不修改仓库主干** 的前提下，用 Synthetic 权重 + activation，直调 NPU MoE 计算路径上的算子：

- `torch_npu.npu_dynamic_quant`（per-token act quant）
- `torch_npu.npu_grouped_matmul`（W8A8 grouped GEMM，up + down）
- SiLU + Mul（或 fused swiglu，若 kernel 支持；A3 fused/unfused 双路径）
- `torch_npu.npu_quant_matmul`（shared expert dense W8A8 MLP）
- 对照 bench：1×grouped_matmul vs N×独立 quant_matmul（S2 系统选型）

**仅测「dispatch 已完成、token 已落到 active expert」之后的 NPU 段**。
不测：gating / routing / dispatch / combine / CPU MoE / all-to-all。

默认场景：DSv4-Flash-W8A8 decode 单 token，top_k=6，n_active=6（=100% 热专家命中）；
sweep `n_active ∈ {1..6}` 覆盖 CPU offload 命中率 ∈ [1/6, 1]。

形状（实测对齐）：`H=4096, I=moe_intermediate=2048, E_routed=256, n_shared=1`。

**关键定义（N2）**：`N = n_active_experts × tokens_per_expert` = NPU grouped GEMM 实际看到的 token 数；
默认 decode 均匀命中 → `tokens_per_expert=1`，sweep 下 `N ∈ {1..6}`，`group_list.sum()==N`。

## 目录

| 路径 | 说明 |
|------|------|
| `IMPLEMENTATION_PLAN.md` | 代码生成计划（模块规格 + 排期） |
| `results/ANALYSIS_AND_REPRODUCTION.md` | **分析结论 + 复现指南（必读）** |
| `results/README.md` | 结果索引 + 一键复现 |
| `config/dsv4_flash_moe.yaml` | 模型 / MoE / bench 参数 |
| `env.sh` | CANN + 性能调度环境变量（不自动选卡） |
| `run_all.sh` | D4：Python Event 全量 bench + summary |
| `run_msprof.sh` | D5：msprof 硬件层 + launch 对比报告 |
| `moe_bench/` | Python 包 |

---

## NPU 选卡 SOP

NPU 是共享资源，运行前**先看哪张卡空，再 export**：

```bash
npu-smi info
```

输出每张卡的 `Health / Power / Memory-Usage / AICore-Util(%)`。**空卡**判断：
- `Memory-Usage` 接近 0 MB
- `AICore-Util` = 0%
- 没有进程在 `npu-smi info -m`（也可看 `npu-smi info | grep -A 5 Processes`）

选定后：

```bash
export ASCEND_RT_VISIBLE_DEVICES=2     # 物理 2 号卡
# 进程内仍是 npu:0（visible_devices 已重映射）
python -c "import torch_npu, torch; print(torch.npu.current_device())"   # → 0
```

`env.sh` **不**自动选卡，避免抢别人的卡。`init_npu.print_npu_smi()` 在 bench 启动时打印简表作为提示。

---

## 快速开始

```bash
cd tools/moe_microbench
source env.sh

# 1) 看 / 选 NPU 卡
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=2

# 2) 不上 NPU 校验形状
DRY_RUN=1 bash run_all.sh

# 3) 单段 smoke（NPU）
python -m moe_bench.bench_gemm_up --sanity --repeat 10

# 4) D4 全量 Event bench
bash run_all.sh
# → results/summary_decode.md

# 5) D5 msprof 硬件层（需先跑 D4 或已有 Event JSON）
bash run_msprof.sh
# → results/msprof_vs_python_comparison.md
```

---

## 核心结论（摘要）

> 完整分析、43 层折算、优化建议见 [`results/ANALYSIS_AND_REPRODUCTION.md`](results/ANALYSIS_AND_REPRODUCTION.md)。

**decode bs=1, n_active=6**（2026-05-26 实测）：

| 口径 | 单层 routed | ×43 层 |
|------|------------:|-------:|
| Python Event | 9142 µs | **~393 ms/token** |
| msprof 硬件 | 202 µs | **~8.7 ms/token** |
| launch overhead | **~98%** | — |

要点：

1. **Event 数字不能当 kernel 效率**：`gemm_down util=99.78%`（D4）在硬件层仅 **~50%**（D5）。
2. **Grouped vs loop 2× 差距在 Python 层**：硬件 104 µs ≈ 102 µs；CANN grouped API fallback 到 6× quant_matmul。
3. **优化优先级**：NPUGraph / op fusion > 修 grouped API > gemm_down 带宽调优。

---

## sweep（可选）

```bash
# n_active_experts × tokens_per_expert 扫描
for E in 6 16 32; do
  for K in 1 2 4 8; do
    OUT_DIR=results/E${E}_K${K} N_ACTIVE=$E NUM_TOKENS=$((K*1)) TOP_K=1 \
      bash run_all.sh
  done
done
```

（`num_tokens × top_k / n_active_experts` 决定 `tokens_per_expert`；本 microbench 默认按 `n_active=top_k` 解释，sweep 时手动配 `num_tokens` 和 `top_k`）

---

## 已知偏离与局限（S6，外部引用请先读）

1. **与论文 FP4×FP8 偏离**：DSv4 论文 §4.2.1/§5.2.1 描述 routed expert weight=MXFP4 + activation=FP8 cast；
   本 bench 对齐的是本地 ckpt `DSv4-Flash-W8A8` 的实际部署 (`quantization_config.format=int-quantized`)：
   `weight=int8 per-channel`, `act=int8 per-token dynamic`。
   → HBM 流量 W8A8 ≈ FP4 的 **~2×**。定量对比（默认形状 decode bs=1, n_active=6, N6 已含 shared, N8）：

   | 路径 | routed weight | shared weight | 总 weight | roofline @ peak 1.6 TB/s | roofline @ eff 1.0 TB/s |
   |------|---------------|---------------|-----------|--------------------------|-------------------------|
   | 本计划 W8A8 | 144 MB | 24 MB | **168 MB** | **105 μs** | **168 μs** |
   | 论文 FP4 (含 MXFP4 block scale ≈ +6%) | ~76 MB | ~13 MB | **~89 MB** | **~56 μs** | **~89 μs** |
   | 差距 | — | — | ~1.9× | ~1.9× | ~1.9× |

   外推论文 FP4 性能时，把本 bench 的纯 weight-load 部分数字 ÷ 1.9。

   精简对比（N8，routed weight only）：

   ```
   W8 (本计划)  : 144 MB → 90 μs   @1.6 TB/s peak  / 144 μs @1.0 TB/s effective
   FP4 (论文)   : ~76 MB  → 48 μs   @1.6 TB/s peak  /  76 μs @1.0 TB/s effective
   gap          : ~2× HBM traffic（W8 vs FP4，本 bench 不反映论文 FP4 路径）
   ```
2. **不测 dispatch / combine / gating / all-to-all 通信**：与 grouped GEMM 不融合，单独算子；gating 时间 ~0.17 ms 见 attention profiling 报告。
3. **不测 CPU MoE / KT 路径**：本工程只关心 NPU 段。
4. **仅 decode bs=1**：`num_tokens=1, top_k=6`；prefill / chunked-prefill 见 D5（默认不跑）。
5. **shared expert 与 routed expert 并行**（A8）：生产中走不同 stream，端到端 latency ≈ max(shared, routed)，
   summary 表里两段数字 **不能简单相加**。
6. **CANN 算子 API 版本敏感**：`npu_grouped_matmul` signature 在不同 CANN 版本可能略异，`ops_runner.py` 单点维护。
7. **Synthetic 权重数值随机**：算子 Device Duration 主要由 shape/dtype 决定，与具体数值无关；
   但 RoPE-like fast-path 类 kernel 若内部分支依赖数据分布需 D2 时校验。
8. **roofline 下界用 HBM 1.6 TB/s 估算**：若实测 util < 1，说明带宽估高，改 yaml `roofline.hbm_bandwidth_tb_s`。
9. **fused SwiGLU+量化**（A3）：若 `sgl_kernel_npu` 不提供则自动降级到 unfused；fused/unfused 两路径都在 summary 中报。
10. **可选 bf16 baseline**：用 `act_dtype=bf16` 跑同形 grouped matmul 对照量化收益。

---

## 与 attn_microbench 的对照

| 维度 | attn_microbench | moe_microbench |
|------|-----------------|----------------|
| 测什么 | SWA / CSA / HCA attention | W8A8 grouped GEMM + dense shared MLP |
| 主要 op | `npu_sparse_attn_sharedkv` | `npu_grouped_matmul`, `npu_quant_matmul` |
| 形状决策 | seq_len 32k → page_table | num_tokens × top_k → group_list |
| 共享方法 | Event + host wall, sanity, p95/p99/std, dry-run | 同 |
| msprof 硬件 | `run_msprof.sh` + comparison | `run_msprof.sh` + comparison |
| 分析文档 | `results/P1_7_analysis_guide.md` | `results/ANALYSIS_AND_REPRODUCTION.md` |

详细模块 API 见 `IMPLEMENTATION_PLAN.md`。
