# MoE Microbench 分析结果索引

> 工作目录：`tools/moe_microbench/`  
> **主文档（结论 + 复现）**：[`ANALYSIS_AND_REPRODUCTION.md`](./ANALYSIS_AND_REPRODUCTION.md)

---

## 快速入口

| 主题 | 文档 | 原始数据 |
|------|------|----------|
| **结论与复现（必读）** | [`ANALYSIS_AND_REPRODUCTION.md`](./ANALYSIS_AND_REPRODUCTION.md) | — |
| **Python Event 汇总 @ decode** | [`summary_decode.md`](./summary_decode.md) | `act_quant.json` … `grouped_vs_loop.json` |
| **Python vs 硬件 launch 拆解** | [`msprof_vs_python_comparison.md`](./msprof_vs_python_comparison.md) | `*_msprof.json` + Event JSON |
| **n_active 硬件 sweep** | [`msprof_n_active_sweep.md`](./msprof_n_active_sweep.md) | `*_msprof_n{1..6}.json` |

---

## 一键复现

```bash
cd tools/moe_microbench
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>   # 先 npu-smi info
source env.sh

# D4：Python Event 全量 bench + summary
bash run_all.sh
# → results/summary_decode.md

# D5：msprof 硬件层 + comparison（需先跑过 D4 或自带 Event JSON）
bash run_msprof.sh
# → results/msprof_vs_python_comparison.md

# D5 + n_active sweep
SWEEP_N_ACTIVE="1 2 3 4 5 6" bash run_msprof.sh
# → results/msprof_n_active_sweep.md

# 快速冒烟（repeat=100）
QUICK=1 bash run_all.sh
QUICK=1 bash run_msprof.sh
```

详细分步、自检命令、环境说明见 [`ANALYSIS_AND_REPRODUCTION.md`](./ANALYSIS_AND_REPRODUCTION.md)。

---

## 数据落盘

### 进 git（`results/`）

| 模式 | 文件 | 说明 |
|------|------|------|
| Python Event | `{act_quant,gemm_up,silu_mul_*,gemm_down,routed_full,shared_expert,grouped_vs_loop}.json` | D4，`repeat=1000` |
| msprof 硬件 | `*_msprof.json` | D5，`mode=msprof_hardware_only` |
| msprof sweep | `*_msprof_n{1..6}.json` | D5.5 |
| sanity | `sanity_*.json` | `--sanity` 输出 |
| 报告 | `summary_decode.md`, `msprof_vs_python_comparison.md`, `msprof_n_active_sweep.md` | 生成物 |

### 不进 git（`./npu_results/`）

msprof 原始 trace，见 `.gitignore`：

```text
npu_results/<bench_name>/*_ascend_pt/ASCEND_PROFILER_OUTPUT/
├── kernel_details.csv      ← 主解析目标（Duration(us)）
├── op_statistic.csv
├── operator_details.csv
└── step_trace_time.csv
```

pattern 不匹配时自动生成 `npu_results/<name>/op_types_seen.txt`。

---

## 核心数字（decode bs=1, n_active=6, 2026-05-26 实测）

| 指标 | 单层 routed | 43 层 MoE 合计 |
|------|------------:|---------------:|
| Python Event | **9142 µs** | **~393 ms** |
| msprof 硬件 | **202 µs** | **~8.7 ms** |
| launch overhead | **~98%** | — |

完整分析见 [`ANALYSIS_AND_REPRODUCTION.md`](./ANALYSIS_AND_REPRODUCTION.md) §3–§5。
