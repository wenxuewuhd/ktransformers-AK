# Attention Microbench 分析结果索引

> 工作目录：`tools/attn_microbench/`  
> **P1.7 硬件分析主文档**：[`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md)（结论、复现命令、数据位置、解析字段）

---

## 快速入口

| 主题 | 文档 | 原始数据 |
|------|------|----------|
| **整网 attention 硬件折算 @32k** | [`network_hw_estimate.md`](./network_hw_estimate.md) | `*_msprof*.json` |
| **Python launch vs NPU kernel** | [`msprof_vs_python_comparison.md`](./msprof_vs_python_comparison.md) | `swa/csa/hca_msprof.json` + `seq_32768/*.json` |
| **CSA 硬件 seq sweep** | [`msprof_seq_sweep.md`](./msprof_seq_sweep.md) | `csa_msprof_seq_*.json` |
| **Python Event seq sweep（平坦）** | [`summary_seq_sweep_r1000.md`](./summary_seq_sweep_r1000.md) | `seq_{S}/*.json` |
| **P1.5 旧结论（已推翻）** | [`p1_5_floor_final.md`](./p1_5_floor_final.md) | 同上 Event 数据 |
| **P1.1–P1.4 诊断** | `p1_1_floor_evidence.json`, `diag_*.json`, [`p1_field_diff.md`](./p1_field_diff.md) | — |
| **P1.3 dump 阻塞** | [`p1_3_blocked.md`](./p1_3_blocked.md) | `p1_3_*.log` |

---

## 一键复现（P1.7）

```bash
cd tools/attn_microbench
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
source env.sh

# 32k 三类 msprof + Python vs hw 对比 + 整网表
bash run_msprof.sh

# CSA 硬件 seq sweep + 整网 seq 表
SEQ_LEN_SWEEP="1024 4096 8192 16384 32768" bash run_msprof.sh

# 仅从已有 JSON 重生成整网表
python scripts/summarize_network_hw.py --out results/network_hw_estimate.md
```

---

## 数据落盘位置

### 进 git（`results/`）

| 模式 | 文件模式 | 说明 |
|------|----------|------|
| msprof 硬件 | `swa_msprof.json`, `csa_msprof.json`, `hca_msprof.json` | @32k 三类 |
| msprof seq sweep | `csa_msprof_seq_{S}.json` | CSA indexer + attn |
| Python Event | `seq_{S}/swa.json`, `csa.json`, `hca.json` | repeat=1000 |
| 诊断 | `diag_*.json`, `p1_*.json` | P1.1–P1.4 |

### 不进 git（`./npu_results/`）

msprof 原始 trace，结构：

```text
npu_results/<bench_name>/*_ascend_pt/ASCEND_PROFILER_OUTPUT/
├── kernel_details.csv      ← 主解析目标
├── op_statistic.csv
├── operator_details.csv
└── step_trace_time.csv
```

---

## 应解析什么

| 目标 | 文件 | 关键列 / 规则 |
|------|------|---------------|
| **硬件 kernel 耗时** | `kernel_details.csv` | `Step Id` 非空；`Duration(us)`；`Name` 模糊匹配 |
| SWA/HCA/CSA attn | pattern `SparseAttnSharedkv` | CANN `npu_sparse_attn_sharedkv` |
| CSA indexer | pattern `QuantLightningIndexer` | CANN `npu_quant_lightning_indexer` |
| Python Event | `*.json` | `indexer_us` / `attn_us` → `mean`, `std` |
| 聚合备用 | `op_statistic.csv` | Avg Time |
| 旧版 CANN | `op_summary_*.csv` | `OP Type` + `Task Duration(us)` |

实现：`attn_bench/msprof_runner.py` → `parse_op_summary()`

---

## 核心结论（P1.7）

- 整网 attention **硬件 @32k ≈ 2.0 ms/token**（42 层）
- Python Event **~11.4 ms/token** → launch overhead **~82%**
- CSA indexer 硬件 1024→32768：**16→34 µs（2.1×）**；Python Event 仍平坦

详细公式与层架构见 [`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md) §2–§4。
