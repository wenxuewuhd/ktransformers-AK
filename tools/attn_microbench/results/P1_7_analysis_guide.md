# P1.7 分析落盘：硬件耗时、整网折算、复现与数据解析

> 日期：2026-05-26  
> 前置：P1.7 msprof 双轨测量已完成（Event 保留，msprof 新增）

---

## 1. 核心结论（摘要）

| 口径 | 整网 attention @ 32k decode | 说明 |
|------|----------------------------|------|
| **NPU 硬件 (msprof)** | **~2.0 ms / token** | 42 层加总（SWA×2 + CSA×20 + HCA×20） |
| **Python Event (eager)** | **~11.4 ms / token** | repeat=1000；launch overhead ~82% |
| **生产 msprof (token#200)** | SparseAttn **~1.12 ms / 43 层** | seq≈200，口径不同，不可直接对标 32k |

**判定**：P1.5「kernel floor」被推翻 — 平坦来自 **Python launch**；硬件 indexer 随 seq 缩放（1024→32768 约 **2.1×**）。

---

## 2. 层架构与折算公式

来源：`config/dsv4_flash.yaml` → `layer_layout_info` + `DeepSeek-V4-Flash-W8A8/config.json` compress_ratios

| 层 | 类型 | 层数 | 每 token 算子 |
|----|------|------|---------------|
| L0–L1 | SWA (ratio=1) | 2 | `npu_sparse_attn_sharedkv` ×1 |
| L2–L41 | CSA (ratio=4) | 20 | `npu_quant_lightning_indexer` + `npu_sparse_attn_sharedkv` |
| L3–L41 间 | HCA (ratio=128) | 20 | `npu_sparse_attn_sharedkv` ×1 |
| L43 | TBD | **不计** | MTP head / padding 待确认 |

**整网硬件 attention（单 decode token）**：

```text
T_net_hw = N_swa × T_swa_hw
         + N_csa × (T_indexer_hw + T_csa_attn_hw)
         + N_hca × T_hca_hw

         = 2×T_swa + 20×(T_idx + T_csa_attn) + 20×T_hca
```

**@ 32k 实测代入**（整网表用 seq sweep 的 `csa_msprof_seq_32768.json` + `swa/hca_msprof.json`）：

> **口径说明**：`csa_msprof.json`（`run_msprof.sh` 32k 三类）与 `csa_msprof_seq_32768.json`（seq sweep 独立 run）数值略有差异（indexer 36.1 vs 33.9 µs），属 run-to-run 方差；**整网 / seq 表以 sweep JSON 为准**，launch overhead 对比以 `csa_msprof.json` + Event 为准。

```text
= 2×21.6 + 20×(33.9 + 37.5) + 20×26.5
≈ 2001 µs ≈ 2.00 ms
```

| 组成部分 | 耗时 | 占比 |
|----------|------|------|
| SWA×2 | 43 µs | 2.1% |
| CSA indexer×20 | 678 µs | 33.9% |
| CSA attn×20 | 750 µs | 37.5% |
| HCA attn×20 | 530 µs | 26.5% |

---

## 3. 不同 seq_len 的整网硬件估算

CSA 用 msprof seq sweep；SWA/HCA 暂用 **32k 常数**（SWA 窗口=128 固定；HCA sweep 未跑）。

| seq_len | indexer_hw | attn_hw (CSA) | 整网 hw | vs 1024 |
|---------|------------|---------------|---------|---------|
| 1024 | 16.2 µs | 30.5 µs | **1.51 ms** | 1.00× |
| 4096 | 24.1 µs | 38.3 µs | **1.82 ms** | 1.21× |
| 8192 | 20.7 µs | 36.2 µs | **1.71 ms** | 1.13× |
| 16384 | 27.9 µs | 38.5 µs | **1.90 ms** | 1.26× |
| 32768 | 33.9 µs | 37.5 µs | **2.00 ms** | 1.33× |

---

## 4. Launch Overhead（@ 32k）

来源：`results/msprof_vs_python_comparison.md`

| op | Python Event | msprof hw | overhead_pct |
|----|--------------|-----------|--------------|
| swa_attn | 133.8 µs | 21.6 µs | **83.9%** |
| csa_indexer | 265.0 µs | 36.1 µs | **86.4%** |
| csa_attn | 157.2 µs | 38.8 µs | **75.3%** |
| hca_attn | 136.1 µs | 26.5 µs | **80.5%** |

整网：Python **~11.4 ms** vs 硬件 **~2.0 ms** → launch **~82%**。

---

## 5. 数据文件索引

### 5.1 JSON 结果（`results/`，可进 git）

| 文件 | 内容 |
|------|------|
| `swa_msprof.json` | SWA @32k 硬件 attn |
| `csa_msprof.json` | CSA @32k indexer + attn 硬件 |
| `hca_msprof.json` | HCA @32k 硬件 attn |
| `csa_msprof_seq_{1024,4096,8192,16384,32768}.json` | CSA seq sweep 硬件 |
| `swa_msprof_smoke.json` | P1.7.1 冒烟 |
| `msprof_vs_python_comparison.md` | Python vs hw 对比 + Roofline |
| `msprof_seq_sweep.md` | CSA 硬件 seq sweep 表 |
| `seq_{S}/swa.json`, `csa.json`, `hca.json` | Python Event @ repeat=1000 |
| `summary_seq_sweep_r1000.md` | Python Event seq sweep |
| `network_hw_estimate.md` | **自动生成**整网折算表 |

### 5.2 msprof Trace（`./npu_results/`，**不进 git**）

| 目录 | 内容 |
|------|------|
| `npu_results/swa_attn_seq32768/` | SWA trace |
| `npu_results/csa_indexer_seq32768/` | CSA indexer trace（独立） |
| `npu_results/csa_attn_seq32768/` | CSA attn trace（独立） |
| `npu_results/hca_attn_seq32768/` | HCA trace |
| `npu_results/seq_{S}/csa_indexer_seq{S}/` | seq sweep indexer |
| `npu_results/seq_{S}/csa_attn_seq{S}/` | seq sweep attn |

每个 trace 子目录结构：

```text
npu_results/<name>/<host>_<pid>_<ts>_ascend_pt/
└── ASCEND_PROFILER_OUTPUT/
    ├── kernel_details.csv      ← **主解析目标**
    ├── op_statistic.csv        ← 聚合统计（备用）
    ├── operator_details.csv    ← PyTorch op 级（备用）
    └── step_trace_time.csv
```

匹配失败时：`npu_results/<name>/op_types_seen.txt`（自动 dump）

---

## 6. 应解析哪些数据

### 6.1 主路径：`kernel_details.csv`

| 列 | 用途 |
|----|------|
| `Step Id` | **非空** = profiler active 区间内的单次 kernel（取 active=10 行） |
| `Name` / `Type` | op 名；模糊匹配 pattern |
| `Duration(us)` | **硬件 device time**（统计 mean/std/p95） |

**Op pattern（`parse_op_summary` 参数）**：

| Bench | Pattern | 对应 CANN op |
|-------|---------|--------------|
| SWA / HCA / CSA attn | `SparseAttnSharedkv` | `npu_sparse_attn_sharedkv` |
| CSA indexer | `QuantLightningIndexer` | `npu_quant_lightning_indexer` |

实现：`attn_bench/msprof_runner.py` → `parse_op_summary()`

> 注：CANN 新版本无 `op_summary_*.csv`，代码已 fallback 到 `kernel_details.csv`。

### 6.2 备用路径

| 文件 | 何时用 |
|------|--------|
| `op_statistic.csv` | 只需 aggregate Avg Time，不需要 per-call 分布 |
| `operator_details.csv` | pattern 在 kernel 名对不上时；列 `Device Self Duration With AICore(us)` |
| `op_summary_*.csv` | 旧版 CANN；列 `OP Type` + `Task Duration(us)` |

### 6.3 JSON 字段

msprof bench 输出统一：

```json
{
  "mode": "msprof_hardware_only",
  "indexer_hw": { "device_mean_us", "device_std_us", "matched_rows", "op_pattern", "source_csv" },
  "attn_hw": { ... }
}
```

`matched_rows` 应等于 yaml `msprof.active`（默认 **10**）。

---

## 7. 复现命令

### 7.0 环境

```bash
cd tools/attn_microbench
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>   # npu-smi info 查看 HBM
source env.sh
```

### 7.1 冒烟（P1.7.1）

```bash
python -c "from attn_bench.msprof_runner import run_with_msprof, parse_op_summary; print('OK')"

python -m attn_bench.bench_swa --seq-len 1024 --msprof \
  --msprof-out ./npu_results \
  --out results/swa_msprof_smoke.json

# 自检
test -f results/swa_msprof_smoke.json
python -c "import json; d=json.load(open('results/swa_msprof_smoke.json')); assert d['mode']=='msprof_hardware_only'; assert d['attn_hw']['matched_rows']>=10"
```

### 7.2 三类 @32k 正式 msprof（P1.7.2）

```bash
python -m attn_bench.bench_swa  --seq-len 32768 --msprof --msprof-out ./npu_results --out results/swa_msprof.json
python -m attn_bench.bench_csa  --seq-len 32768 --msprof --msprof-out ./npu_results --out results/csa_msprof.json
python -m attn_bench.bench_hca  --seq-len 32768 --msprof --msprof-out ./npu_results --out results/hca_msprof.json

jq 'has("indexer_hw") and has("attn_hw")' results/csa_msprof.json   # true
```

### 7.3 Python vs 硬件对比（P1.7.3）

```bash
# Event 数据（若已有 seq_32768/*.json 可跳过）
SEQ_LEN=32768 REPEAT=1000 WARMUP=30 SANITY_FLAG="" bash run_all.sh

python -m attn_bench.report --mode comparison \
  --msprof-jsons results/swa_msprof.json results/csa_msprof.json results/hca_msprof.json \
  --event-jsons results/seq_32768/swa.json results/seq_32768/csa.json results/seq_32768/hca.json \
  --out results/msprof_vs_python_comparison.md
```

### 7.4 CSA seq sweep msprof（P1.7.4）

```bash
for S in 1024 4096 8192 16384 32768; do
  python -m attn_bench.bench_csa --seq-len $S --msprof \
    --msprof-out ./npu_results/seq_$S \
    --out results/csa_msprof_seq_$S.json
done

python -m attn_bench.report --mode msprof-sweep \
  --inputs results/csa_msprof_seq_1024.json results/csa_msprof_seq_4096.json \
           results/csa_msprof_seq_8192.json results/csa_msprof_seq_16384.json \
           results/csa_msprof_seq_32768.json \
  --out results/msprof_seq_sweep.md
```

### 7.5 一键脚本

```bash
bash run_msprof.sh                                    # 32k 三类 + comparison
SEQ_LEN_SWEEP="1024 4096 8192 16384 32768" bash run_msprof.sh   # 含 seq sweep
```

### 7.6 重新生成整网折算表

```bash
python scripts/summarize_network_hw.py \
  --out results/network_hw_estimate.md
```

### 7.7 pattern 找不到时

```bash
cat npu_results/swa_attn_seq1024/op_types_seen.txt
# 或
python -c "
from pathlib import Path
from attn_bench.msprof_runner import list_op_types_in_trace
print(list_op_types_in_trace(Path('npu_results/swa_attn_seq32768')))
"
# 修改 bench_*.py 中 parse_op_summary 的 pattern 后重跑
```

---

## 8. 与生产 msprof 对照（只读参考）

| 指标 | 生产 (token#200) | Microbench hw @32k |
|------|------------------|---------------------|
| SparseAttnSharedkv | 1.12 ms / 43 次 ≈ 26 µs/次 | 42 次 sparse 合计 ~1.32 ms |
| 整步 NPU busy | ~28 ms / 43 层 forward | **仅 attention 硬件 ~2 ms** |

生产报告：`doc/zh/DeepSeek-V4-Flash_eager_decode_profiling_分析报告_20260525.md`

**不可直接对比的原因**：seq_len 不同、生产含 MoE/MatMul/Compressor、生产有 fusion/overlap、microbench 为 isolated 逐 op 相加。

P2 目标：同 seq_len（~200）跑 microbench + in-server msprof。

---

## 9. 相关文档

| 文件 | 说明 |
|------|------|
| `IMPLEMENTATION_PLAN_V4.md` §0.6 / §6 P1.7 | 设计与里程碑 |
| `doc/zh/DeepSeek-V4-Flash_Attention_Microbench_审核报告.md` §12 | 对外结论 |
| `config/dsv4_flash.yaml` → `msprof:` | profiler 参数 |
| `attn_bench/msprof_runner.py` | 抓 trace + 解析 |
| `attn_bench/report.py` | comparison / msprof-sweep 报告 |

---

*本文件由 P1.7 分析落盘；数据更新后请重跑 §7 命令并执行 `scripts/summarize_network_hw.py`。*
