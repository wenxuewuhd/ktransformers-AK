# MoE Microbench — 分析结论与复现指南

> **实测环境**：Ascend 910B，`ASCEND_RT_VISIBLE_DEVICES=1`，CANN + torch_npu 2.8.0  
> **模型/ckpt**：DeepSeek-V4-Flash-W8A8（W8A8 int-quantized，非论文 FP4）  
> **默认场景**：decode `num_tokens=1`，`top_k=6`，`n_active_experts=6`，`N=6`  
> **层数**：`num_hidden_layers=43`（全 MoE，无 dense 替换层）

---

## 1. 测量口径

### 1.1 测什么 / 不测什么

| 包含 | 不包含 |
|------|--------|
| post-dispatch NPU 段：act_quant → gemm_up → silu_mul → gemm_down | gating / routing / dispatch / combine |
| shared expert dense MLP（与 routed 并行，latency 取 max） | CPU MoE / KT offload 路径 |
| grouped vs loop 对照 | Attention / embedding / LM head |

### 1.2 双轨计时（D4 + D5）

| 轨道 | 方法 | 代表含义 |
|------|------|----------|
| **Python Event**（D4） | `torch.npu.Event` + host wall | SGLang/KT **eager 模式**真实 op 调用成本（含 launch overhead） |
| **msprof 硬件**（D5） | `torch_npu.profiler` Level1 + `kernel_details.csv` | NPU kernel **纯 device time**（不含 Python/driver） |

两者差额 ≈ **launch overhead**，是 NPUGraph / kernel fusion 的优化目标量。

### 1.3 CANN 实测路径

- `npu_grouped_matmul` 在本环境报 **CANN 161002**，自动 **fallback 到 6× `npu_quant_matmul` loop**
- msprof 抓到的 op 为 **`QuantBatchMatmul`**（非 GroupedMatmul）
- 权重 layout：`gate_up [2I,H]`，down `[I,H]`（= ckpt w2.T），per-channel W8 合成

---

## 2. D4 结论（Python Event）

数据来源：[`summary_decode.md`](./summary_decode.md)（`repeat=1000`，warmup=30）

### 2.1 各段耗时（n_active=6）

| segment | device_mean_us | host_mean_us | dispatch_overhead_us | util_vs_achievable |
|---------|---------------:|-------------:|---------------------:|-------------------:|
| act_quant_post | 166 | 300 | 134 | — |
| act_quant_pre | 186 | 313 | 127 | — |
| gemm_up | **4274** | 4438 | 164 | 50.95% |
| silu_mul_unfused | 350 | 502 | 152 | — |
| gemm_down | **4185** | 4351 | 166 | **99.78%** ⚠ |
| routed_full | **9142** | 9311 | 169 | 72.66% |
| shared_expert | 979 | 1132 | 153 | 46.67% |

`routed_full` 分解：compute_only ≈ **9105 µs**，post_dispatch ≈ **9142 µs**（act_quant 仅 +37 µs）。

### 2.2 Grouped vs Loop（Python 层）

| target | grouped_us | loop_us | 倍率 |
|--------|----------:|--------:|-----:|
| gemm_up | 4508 | **2071** | 2.2× |
| gemm_down | 4460 | **2038** | 2.2× |

D4 表面结论：loop 路径在 Python 层快约 **2×**。

### 2.3 n_active sweep（Event）

| n_active | gemm_up_us | gemm_down_us | routed_full_us | util_eff |
|---------:|-----------:|-------------:|---------------:|
| 1 | 4507 | 4114 | 8472 | — |
| 6 | 4393 | 4451 | 8981 | 71.4% |

`act_quant(pre)` 固定 `[1,H]`，不随 n_active 变化（N5）。

---

## 3. D5 结论（msprof 硬件层）

数据来源：[`msprof_vs_python_comparison.md`](./msprof_vs_python_comparison.md)

### 3.1 Launch overhead 拆解

| segment | python_event_us | msprof_device_us | overhead_pct |
|---------|----------------:|-----------------:|-------------:|
| act_quant_post | 166 | 3.3 | **98.0%** |
| gemm_up | 4274 | 107.4 | **97.5%** |
| silu_mul | 350 | 6.7 | **98.1%** |
| gemm_down | 4185 | 84.2 | **98.0%** |
| routed_full | 9142 | 202.0 | **97.8%** |
| shared_expert | 979 | 46.5 | **95.2%** |

**判定**：MoE NPU 段在 eager 模式下 **~95–98% 时间是 Python launch overhead**，真实 kernel 仅占 2–5%。

### 3.2 Roofline（硬件层 @ HBM measured 1.20 TB/s）

| segment | hw_us | lb @ 1.2 TB/s | util_vs_measured |
|---------|------:|--------------:|-----------------:|
| gemm_up | 107 | 84 | **78%** |
| gemm_down | 84 | 42 | **50%** |
| shared_expert | 47 | 21 | **45%** |

**关键修正（D4 → D5）**：

- D4 报 `gemm_down util_vs_achievable = 99.78%` → **不可信**（Event 含 launch，凑近 roofline 下界）
- D5 硬件层 `gemm_down util = 49.8%` → kernel **未打满带宽**，仍有 ~2× 优化空间

### 3.3 Grouped vs Loop（硬件层 — 解开 D4 谜团）

| path | python_event_us | hw_device_us | overhead_pct |
|------|----------------:|-------------:|-------------:|
| grouped | 4508 | **104.0** | 97.7% |
| loop | 2071 | **102.2** | 95.1% |

**判定**：`hw_grouped ≈ hw_loop`（104 vs 102 µs）。D4 的 2× 差距 ** entirely 来自 grouped fallback 的 Python 调度路径**，不是 NPU kernel 本身更慢。优化方向：修复 grouped API 或走 loop + 减少 Python 调用次数（NPUGraph）。

### 3.4 n_active 硬件 sweep

数据来源：[`msprof_n_active_sweep.md`](./msprof_n_active_sweep.md)

| n_active | gemm_up_hw | gemm_down_hw | routed_full_hw |
|---------:|-----------:|-------------:|---------------:|
| 1 | 16.7 | 13.0 | 37.2 |
| 6 | 102.5 | 79.3 | 193.0 |

- gemm_up/down hw **近似线性**随 n_active 增长（每增 1 expert ≈ +17 µs up / +13 µs down）
- util 稳定在 **~50–82%**，不随 n_active 显著升高（M=1 GEMV 形态，固定 launch floor）

---

## 4. 43 层整网折算（MoE FFN only）

### 4.1 单层

| 口径 | routed | shared | **层 latency（max）** |
|------|-------:|-------:|---------------------:|
| Python Event (µs) | 9142 | 979 | **9142** |
| msprof hw (µs) | 202 | 47 | **202** |

shared ≪ routed，整网由 routed 主导。shared ‖ routed 并行，**禁止相加**（A8）。

### 4.2 ×43 层合计

| 口径 | 计算 | 结果 |
|------|------|------|
| Python Event | 9142 × 43 | **≈ 393 ms / token** |
| msprof 硬件 | 202 × 43 | **≈ 8.7 ms / token** |
| ~~错误：routed+shared~~ | (9142+979)×43 | ~~435 ms~~ ❌ |

### 4.3 瓶颈分解（43 层 Event）

| 段 | 单层 (µs) | ×43 |
|----|----------:|----:|
| gemm_up | 4274 | **184 ms** |
| gemm_down | 4185 | **180 ms** |
| silu_mul | 350 | 15 ms |
| act_quant | 166 | 7 ms |

GEMM up+down 占 MoE Event 时间 **~93%**。

### 4.4 若叠加 Attention（粗估，口径不同）

| 组件 | hw @ seq=32k | Event @ seq=32k | 来源 |
|------|-------------:|----------------:|------|
| Attention（42 层） | ~2.0 ms | ~11.4 ms | `tools/attn_microbench/results/network_hw_estimate.md` |
| MoE（43 层） | ~8.7 ms | ~393 ms | 本 bench |
| **粗加总** | **~10.7 ms** | **~404 ms** | 串行上界 |

decode 短 KV 时 Attention 远小于 2 ms；上表仅作 seq=32k 量级参考。

---

## 5. 优化建议（按优先级）

1. **NPUGraph / op fusion**：launch overhead 占 97%+，单 op 融合 routed_full 可省 ~4× launch
2. **修复 grouped matmul API**：消除 fallback 的 Python 调度开销（hw 无差异，Event 差 2×）
3. **gemm_down 带宽**：硬件 util ~50%，非 Event 报的 99%；可查 tile/M 维、weight layout
4. **CPU offload 命中率**：n_active sweep 显示 linear scaling；命中率 1/6 → 6/6 时 routed_full hw 37→193 µs

---

## 6. 复现指南

### 6.1 环境准备

```bash
cd tools/moe_microbench
source env.sh                    # CANN + PYTHON_BIN + PYTHONPATH

npu-smi info                     # 选空卡
export ASCEND_RT_VISIBLE_DEVICES=1 # 示例：物理 1 号卡

# 验证 NPU
$PYTHON_BIN -c "import torch, torch_npu; print(torch.npu.is_available())"
```

依赖：`torch` + `torch_npu` + `pyyaml`；Python 默认 `/usr/local/python3.11.14/bin/python3.11`。

### 6.2 D1 — Dry-run（无 NPU）

```bash
DRY_RUN=1 python -m moe_bench.synthetic --selftest
DRY_RUN=1 bash run_all.sh
```

### 6.3 D2 — NPU smoke

```bash
python -m moe_bench.bench_gemm_up --sanity --repeat 10 --quick-mode
python -m moe_bench.bench_gate_up_order_check --ckpt /path/to/DSv4-Flash-W8A8 --layer 3
python -c "from moe_bench.roofline import measure_hbm_effective_tb_s; print(measure_hbm_effective_tb_s('npu:0'))"
```

### 6.4 D4 — Python Event 全量 bench

```bash
# 正式（repeat=1000，约 10–20 min）
bash run_all.sh

# 快速（repeat=100）
QUICK=1 bash run_all.sh

# 可选：n_active sweep + torch profile trace
SWEEP_N_ACTIVE="1 2 3 4 5 6" PROFILE=1 QUICK=1 bash run_all.sh
```

**输出**：`results/summary_decode.md` + 各段 `*.json`

**单段**：

```bash
python -m moe_bench.bench_gemm_up --out results/gemm_up.json --quick-mode
python -m moe_bench.bench_act_quant --out results/act_quant.json   # 含 pre+post 两条
python -m moe_bench.bench_routed_full --out results/routed_full.json
```

### 6.5 D5 — msprof 硬件层

```bash
# 需先存在 D4 Event JSON（run_msprof.sh 用于 comparison 对照）
bash run_msprof.sh

# 含 n_active sweep
SWEEP_N_ACTIVE="1 2 3 4 5 6" bash run_msprof.sh

# 单段 smoke
python -m moe_bench.bench_gemm_up --msprof --msprof-out ./npu_results \
    --quick-mode --out results/gemm_up_msprof_smoke.json
```

**输出**：

- `results/*_msprof.json` — 硬件数字
- `results/msprof_vs_python_comparison.md` — launch 拆解 + roofline 硬件层
- `results/msprof_n_active_sweep.md` — sweep（若设 `SWEEP_N_ACTIVE`）
- `./npu_results/` — 原始 trace（**不进 git**）

**pattern 校准**（若 parse 失败）：

```bash
python -c "
from pathlib import Path
from moe_bench.msprof_runner import list_op_types_in_trace
print(list_op_types_in_trace(Path('./npu_results/gemm_up')))
"
cat ./npu_results/gemm_up/op_types_seen.txt
```

当前环境：`QuantBatchMatmul`，loop fallback 时 `matched_rows=60`（10 steps × 6 experts），聚合后 **~107 µs/call**。

### 6.6 仅从 JSON 重生成报告

```bash
python -m moe_bench.report \
  --inputs results/act_quant.json results/gemm_up.json ... \
  --out results/summary_decode.md

python -m moe_bench.report --mode comparison \
  --msprof-jsons results/*_msprof.json \
  --event-jsons results/{act_quant,gemm_up,silu_mul_unfused,gemm_down,routed_full,shared_expert,grouped_vs_loop}.json \
  --out results/msprof_vs_python_comparison.md

python -m moe_bench.report --mode msprof-n-active-sweep \
  --inputs results/*_msprof_n*.json \
  --out results/msprof_n_active_sweep.md
```

### 6.7 自检清单

| 阶段 | 命令 | 期望 |
|------|------|------|
| D1 | `DRY_RUN=1 python -m moe_bench.synthetic --selftest` | n_active=1..6 全 OK |
| D4 | `test -f results/summary_decode.md` | 8+ 行主表 |
| D5 smoke | `jq '.mode' results/gemm_up_msprof_smoke.json` | `"msprof_hardware_only"` |
| D5 | `jq '.device_mean_us > 0' results/gemm_up_msprof.json` | `true` |
| D5 comparison | `grep overhead_pct results/msprof_vs_python_comparison.md` | 有匹配 |

---

## 7. 输出文件说明

| 文件 | 生成命令 | 内容 |
|------|----------|------|
| `summary_decode.md` | `run_all.sh` | Event 计时主表 + sweep + roofline |
| `msprof_vs_python_comparison.md` | `run_msprof.sh` | launch 拆解 + hw roofline + grouped/loop 判定 |
| `msprof_n_active_sweep.md` | `SWEEP_N_ACTIVE=... run_msprof.sh` | 硬件 n_active 曲线 |
| `*.json` | 各 `bench_*.py --out` | 原始 timing JSON（可 jq 二次分析） |
| `sanity_*.json` | `--sanity` | NaN/abs_max 检查 |

JSON schema（Event）：`device_mean_us / host_mean_us / p50 / p95 / p99 / std / max / n / dispatch_overhead_us`  
JSON schema（msprof）：`mode=msprof_hardware_only / device_mean_us / matched_rows / op_pattern / trace_dir`

---

## 8. 已知局限

1. **W8A8 vs 论文 FP4**：HBM 流量 ~2×，见根目录 `README.md` N8 对比表
2. **Synthetic 权重**：shape/dtype 正确，数值为 per-channel 量化随机权重
3. **不含 Attention**：整网延迟需叠加 `attn_microbench` 结果
4. **CANN 版本敏感**：grouped API、kernel 名随 CANN 升级可能变化；以 `op_types_seen.txt` 为准
5. **Event util 不可直接解读为 kernel 效率**：必须用 D5 msprof 硬件数字

---

## 9. 相关文档

| 文档 | 路径 |
|------|------|
| 工程 README | [`../README.md`](../README.md) |
| 实现计划 v4 | [`../IMPLEMENTATION_PLAN_V4.md`](../IMPLEMENTATION_PLAN_V4.md) |
| 配置 | [`../config/dsv4_flash_moe.yaml`](../config/dsv4_flash_moe.yaml) |
| Attention 整网 @32k | [`../../attn_microbench/results/network_hw_estimate.md`](../../attn_microbench/results/network_hw_estimate.md) |

---

*生成说明：本文档汇总 D1–D5 闭环实测结论（2026-05-26）。数字以 `results/` 下 JSON 为准；重跑后请同步更新 §2–§4 表格或依赖 report 自动生成。*
