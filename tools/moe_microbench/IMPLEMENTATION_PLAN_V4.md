# NPU MoE Microbench — 实施计划 v4

> **v3 → v4 更新（2026-05-26，加入纯硬件 msprof 测量路径，对齐 attn_microbench v4）**
> - **§0.5 新增**：纯硬件算子时间测量目标（msprof Level1，解析 op_summary CSV）
> - **§3.8 重写**：`profile.py` → `msprof_runner.py`，从「只 dump trace」升级到「dump + 解析 op_summary 出数字」
> - **§3.10 增强**：8 段 `bench_*.py` 全部加 `--msprof / --msprof-out`，与 Event 计时**双轨并存**
> - **§3.11 增强**：`report.py` 加 `msprof_vs_python_comparison` 表，量化 launch overhead；roofline util 改用硬件数字
> - **§6 新增**：D5 子里程碑 D5.0–D5.6 — msprof 抓硬件时间，可能推翻 D4 Event 计时的 util_eff 结论
> - **§9 新增**：D4 Event 计时结论标"待 D5 验证 — Python event 含 launch overhead，硬件 util 可能更低"
>
> **v2 → v3 更新留存**：见 §9 决策表
> **v1 → v2 更新留存**：见 §9 决策表

---

## 0. 范围与默认场景

### 0.1 论文 vs 实际 ckpt vs 本计划（基线对齐表）

| 字段 | 论文 (Flash, §4.2.1 / §5.2.1) | 本地 ckpt `DSv4-Flash-W8A8` 实测 | 本计划默认 | 备注 |
|------|------|------|------|------|
| `hidden_size` | 4096 | 4096 | 4096 | ✅ |
| `moe_intermediate_size` | 2048 | 2048 | **2048** | A1 修正 |
| `n_routed_experts` | 256 | 256 | 256 | ✅ |
| `num_experts_per_tok` | 6 | 6 | 6 | ✅ |
| `n_shared_experts` | 1 | 1 | 1 | ✅ |
| `shared expert 中间维` | 2048 | 2048 (= n_shared × moe_inter) | 2048 | ✅ |
| `num_hidden_layers` | — | 43 (全 MoE, `first_k_dense_replace=0`, `moe_layer_freq=1`) | — | 仅作信息 |
| **routed expert 权重精度** | MXFP4 (FP4) | **int8 per-channel** | **int8** | 对齐 ckpt, 不对齐论文 |
| **激活精度** | FP8 cast | **int8 per-token dynamic** | **int8** | 对齐 ckpt |
| **量化格式** | FP4 + FP8 mixed | `int-quantized` (compressed-tensors) | int8 W8A8 | 对齐 ckpt |
| 路由 | 前 3 层 Hash + 后续 expert | 实际 routing 与 per-expert 计算无关 | 不测 routing | — |

**关键判断**：用户场景是「CPU offload MoE 后所有热专家命中 NPU」，使用的是 **DSv4-Flash-W8A8 ckpt 的实际部署路径**，所以 bench 走 W8A8。README 必须明示这与论文 FP4×FP8 的差异（理论 HBM 流量约 2×）。

### 0.2 测试覆盖（双轨）

**关键定义（N2，不变）**：
- `n_active_experts` = 落在 NPU 的 active expert 数（CPU offload 命中数，∈ [1, top_k]）
- `tokens_per_expert` = 每 expert 平均处理 token 数（默认 1）
- **`N = n_active_experts × tokens_per_expert`** = NPU 上 grouped GEMM 实际看到的 token 数
- `group_list.sum() == N` ✅ 自洽

**v4 新增双轨测量**：
1. **Python 层 end-to-end** (`torch.npu.Event` 计时, D1-D4)：反映 SGLang eager 模式下 op 调用真实成本
2. **纯硬件 device time** (`torch_npu.profiler` Level1, **D5 新增**)：反映 kernel 在 AICore/Cube 上真实执行时间，不含 Python/driver/launch overhead

两个数字差额 = NPU launch overhead，是 NPUGraph / kernel fusion 等优化的目标量。

| 段 | 算子 | 默认 shape (decode, n_active=6, tokens/exp=1 → N=6, I=2048, H=4096) |
|----|------|---------------------------------------------------------------------|
| **act_quant** (post-dispatch, N5 默认) | `torch_npu.npu_dynamic_quant` (per-token) | bf16 `[N, 4096]` → int8 `[N, 4096]` + fp32 scale `[N]` |
| **act_quant** (pre-dispatch, N5 可选) | 同上 | bf16 `[num_tokens, 4096]` → int8 `[num_tokens, 4096]` |
| **gemm_up** | `torch_npu.npu_grouped_matmul` (W8A8) | int8 `[N, 4096]` × int8 `[E_act, 4096, 4096]` → bf16 `[N, 4096]` |
| **silu_mul** (unfused / fused, A3) | SiLU(gate) × up + act_quant_mid | bf16 `[N, 4096]` → bf16 `[N, 2048]` → int8 `[N, 2048]` |
| **gemm_down** | `npu_grouped_matmul` (W8A8) | int8 `[N, 2048]` × int8 `[E_act, 2048, 4096]` → bf16 `[N, 4096]` |
| **routed_full** | 上 4 段串联 | 端到端 (不含 dispatch/combine) |
| **shared_expert** | dense W8A8 MLP | bf16 `[num_tokens, 4096]` → `[num_tokens, 4096]`，中间维 2048 |
| **grouped_vs_loop** (S2) | 1×grouped vs N×quant_matmul 对比 | 同 gemm_up shape |

**不测**：`npu_moe_gating_top_k`（gating）/ `npu_moe_init_routing` / `npu_moe_finalize_routing`（dispatch/combine）/ KT CPU MoE / all-to-all 通信。

### 0.3 默认参数 + sweep（不变）

| 参数 | 默认 | sweep | 说明 |
|------|------|-------|------|
| `num_tokens` | 1 | 1（主线，decode） | 用户 bs 中拿到了多少 token |
| `top_k` | 6 | 6（主线） | DSv4-Flash `num_experts_per_tok` |
| `n_active_experts` | 6 | **{1,2,3,4,5,6}** (A5) | NPU 上 active expert 数 |
| `tokens_per_expert` | 1 | 1（主线） | 每 active expert 处理几个 token |
| **`N = n_active × tpe`** | **6** | sweep 时 `N ∈ {1..6}` | NPU grouped GEMM token 数 |
| `group_list` | `[1]*n_active` | 同 | shape `(n_active,)`，sum = N |
| `dtype` | w8a8 | w8a8（主线），bf16（可选 baseline） | |
| `warmup` | 30 | — | |
| `repeat` | **1000** (S3) | `--quick-mode → 100` | |

### 0.4 D4 实测结果（已完成，作为 D5 对照基线）

n_active=6, decode bs=1, repeat=1000, NPU 0：

| segment | dev_mean_us | host_mean_us | util_vs_achievable | 备注 |
|---------|-------------|--------------|---------------------|------|
| gemm_up | 4274 | 4438 | 50.95% | grouped fallback to loop (CANN 161002) |
| gemm_down | 4185 | 4351 | 99.78% | ⚠ 接近 1.0，可能 lb 估高或 Event 含 overhead |
| routed_full | 9142 | 9311 | 72.66% | compute_only=9105 / post_dispatch=9142 |
| shared_expert | 979 | 1132 | 46.67% | |
| HBM effective | 1.20 TB/s (D2 实测回填) | | | |

**已知问题**（D5 要回答）：

1. `gemm_down util_vs_achievable = 99.78%` 看起来 kernel 已打满带宽 — 但 Event 计时含 launch overhead，**硬件层 util 应低于此**。需要 msprof 拆分才能给真实 kernel 效率
2. `routed_full = 9142 µs` vs `gemm_up + gemm_down + silu = 4274 + 4185 + 350 ≈ 8809 µs`，差额 333 µs = pre/post host overhead 累积（每个 op ~165 µs dispatch_overhead × 2）
3. Grouped vs Loop 表显示 **loop 比 grouped 快 2×**（up: 2071 vs 4508），需要硬件层确认是真的 kernel 差异，还是 grouped API fallback 路径导致的额外开销

### 0.5 D5 目标：纯硬件算子时间测量

**为什么必要**：D4 数字都是 Python end-to-end，无法回答：

1. NPU kernel 本身硬件需要多久？
2. ~165 µs/op 的 dispatch overhead 在 kernel 硬件层是不是已经体现？
3. `gemm_down util_eff = 99.78%` 是真的打满，还是 Python overhead 在凑数？
4. loop 比 grouped 快 2× 是 NPU kernel 差异，还是 grouped fallback (CANN 161002 → loop quant_matmul) 在 Python 层多了一层调度？

**方法**：用 `torch_npu.profiler` Level1 包住单 op 调用，从 `op_summary_*.csv` 的 `Task Duration(us)` 列读硬件 device time。

**与现有 Event 计时关系**：**并存，不替换**。Event 计时仍代表 eager 模式生产成本（CPU offload + SGLang 路径都走这），msprof 代表 kernel 硬件极限。两者差额 = launch overhead，是优化指南针。

### 0.4 单条计算量参考（默认形状，不变）

| 指标 | gemm_up | gemm_down | shared (gate_up+down) | 总 (routed) |
|------|---------|-----------|----------------------|-------------|
| weight bytes (n_active=6, W8) | 96 MB | 48 MB | 24 MB | 144 MB |
| roofline @ HBM peak 1.6 TB/s | 60 µs | 30 µs | 15 µs | **90 µs** |
| roofline @ HBM effective 1.0 TB/s | 96 µs | 48 µs | 24 µs | **144 µs** |
| roofline @ HBM measured 1.2 TB/s (D2) | 80 µs | 40 µs | 20 µs | **120 µs** |

---

## 1. 工作空间结构（v4）

```
tools/moe_microbench/
├── README.md                          (NPU 选卡 SOP + 用法 + 已知局限 §S6 + FP4 对比 N8)
├── IMPLEMENTATION_PLAN.md             (本文 v4)
├── env.sh                             (CANN + 性能调度)
├── run_all.sh                         (一键 + --profile + --quick-mode + n_active sweep)
├── run_msprof.sh                      (D5 新增: 一键 8 段 msprof + comparison 报告)
├── config/
│   └── dsv4_flash_moe.yaml            (含 msprof 段)
└── moe_bench/                         (Python 包)
    ├── __init__.py
    ├── config.py                      (MoEConfig + msprof dataclass)
    ├── init_npu.py                    (NPU select + npu-smi + version log)
    ├── synthetic.py                   (权重 + activation + group_list + 全套 assert §3.4)
    ├── ops_runner.py                  (quant / grouped_matmul / silu_mul + 自动探测 fused / shared mlp)
    ├── timing.py                      (Event + host wall; 文件头 NOTE 同步 attn_bench)
    ├── sanity.py                      (NaN + 量级)
    ├── profile.py                     (D4: trace-only dump, 保留兼容)
    ├── msprof_runner.py               (D5 新增: 跑 trace + 解析 op_summary 出数字)
    ├── roofline.py                    (S1+N4+N6)
    ├── bench_act_quant.py             (+ --msprof)
    ├── bench_gemm_up.py               (+ --msprof)
    ├── bench_silu_mul.py              (+ --msprof + --fused/--no-fused)
    ├── bench_gemm_down.py             (+ --msprof)
    ├── bench_routed_full.py           (+ --msprof, 分段 trace)
    ├── bench_shared_expert.py         (+ --msprof)
    ├── bench_grouped_vs_loop.py       (+ --msprof, grouped/loop 分别抓)
    └── report.py                      (双轨表 + roofline + comparison)
```

---

## 2. NPU 选卡 SOP（不变）

```bash
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=2
python -c "import torch_npu, torch; print(torch.npu.current_device())"
```

---

## 3. 模块规格

### 3.1-3.7 与 v3 相同

详见 v3 PLAN。D5 之前的代码都已落盘并验证（D1-D4 全部 ✅）。

### 3.8 `moe_bench/msprof_runner.py`（v4 新增 — 核心）

```python
"""纯 NPU 硬件算子时间测量 (Level1 trace + op_summary CSV 解析).
- 不是 Event 计时, 不含 Python overhead.
- 与 timing.py 双轨并存; 共同回答 'kernel 硬件 vs Python 端到端' 拆分.
- 与 attn_bench/msprof_runner.py 同结构 (NOTE 手动同步, 不抽共用层 / S7).
"""
# NOTE: keep in sync with tools/attn_microbench/attn_bench/msprof_runner.py
# Changes here should be mirrored manually (no shared module by design).

import glob
from pathlib import Path
import pandas as pd
import torch_npu


def run_with_msprof(fn, name: str, out_dir: str,
                    skip_first: int = 5, warmup: int = 2, active: int = 10,
                    profiler_level: str = "Level1",
                    aic_metrics: str = "PipeUtilization",
                    record_shapes: bool = True,
                    with_stack: bool = False) -> Path:
    """
    跑 fn() (skip_first + warmup + active) 次, msprof 抓 trace 到 out_dir/name/.
    返回 trace 目录路径; 后续用 parse_op_summary() 读时间.
    """
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)

    level = getattr(torch_npu.profiler.ProfilerLevel, profiler_level)
    metrics = getattr(torch_npu.profiler.AiCMetrics, aic_metrics)

    cfg = torch_npu.profiler._ExperimentalConfig(
        profiler_level=level,
        aic_metrics=metrics,
    )
    sched = torch_npu.profiler.schedule(
        wait=0, warmup=warmup, active=active, repeat=1, skip_first=skip_first,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=sched,
        experimental_config=cfg,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
        record_shapes=record_shapes,
        with_stack=with_stack,
    ) as prof:
        for _ in range(skip_first + warmup + active):
            fn()
            prof.step()
        torch_npu.npu.synchronize()

    return target


def parse_op_summary(trace_dir: Path, op_pattern: str) -> dict:
    """
    从 msprof trace 读 op_summary CSV; 按 OP Type 模糊匹配 op_pattern,
    返回 device duration 的 mean / std / p50 / p95 / p99 / max / min / count.
    """
    csvs = glob.glob(str(trace_dir / "*/ASCEND_PROFILER_OUTPUT/op_summary_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no op_summary CSV under {trace_dir}")

    df = pd.read_csv(csvs[0])
    matched = df[df['OP Type'].str.contains(op_pattern, case=False, na=False)]
    if matched.empty:
        avail = sorted(df['OP Type'].unique().tolist())
        dump = trace_dir / "op_types_seen.txt"
        dump.write_text("\n".join(avail))
        raise ValueError(
            f"no op match {op_pattern!r}; "
            f"saw {len(avail)} unique op types, dumped to {dump}"
        )

    dur_col = 'Task Duration(us)' if 'Task Duration(us)' in matched.columns else 'Duration(us)'
    dur = matched[dur_col].astype(float)

    return {
        "op_pattern": op_pattern,
        "matched_rows": int(len(dur)),
        "device_mean_us": float(dur.mean()),
        "device_std_us": float(dur.std()) if len(dur) > 1 else 0.0,
        "device_p50_us": float(dur.quantile(0.50)),
        "device_p95_us": float(dur.quantile(0.95)),
        "device_p99_us": float(dur.quantile(0.99)),
        "device_max_us": float(dur.max()),
        "device_min_us": float(dur.min()),
    }


def list_op_types_in_trace(trace_dir: Path) -> list[str]:
    """调试用: 列出 trace 里所有 OP Type, 帮助调 op_pattern."""
    csvs = glob.glob(str(trace_dir / "*/ASCEND_PROFILER_OUTPUT/op_summary_*.csv"))
    if not csvs:
        return []
    df = pd.read_csv(csvs[0])
    return sorted(df['OP Type'].unique().tolist())
```

### 3.9 `roofline.py`（v4 — 用硬件数字算 util）

`utilizations()` 不变；但 `report.py` 在生成 D5 双轨表时，**用 msprof device_mean_us 而不是 Event device_mean_us** 作为输入。

### 3.10 `bench_*.py` CLI 增强（v4 — 加 `--msprof`）

每个 bench 都加：

```python
parser.add_argument("--msprof", action="store_true",
                    help="用 torch_npu.profiler 抓纯硬件 device time, 不是 Python end-to-end")
parser.add_argument("--msprof-out", default=None,
                    help="msprof trace 落盘根目录; 默认 cfg.msprof.out_dir")
```

#### 各 bench 的 op pattern（D5.1 smoke 时校准）

| bench | --msprof 时抓哪个 op | 候选 pattern (D5.1 验证) |
|---|---|---|
| bench_act_quant | dynamic quant | `DynamicQuant` |
| bench_gemm_up | grouped matmul OR loop quant_matmul (fallback) | `GroupedMatmul` / `QuantMatmul` |
| bench_silu_mul (unfused) | silu + mul + dyn_quant 三个 sub-op | `Swiglu`/`Silu`/`Mul`/`DynamicQuant` |
| bench_silu_mul (fused) | 一个 fused kernel | (D5.1 探测) |
| bench_gemm_down | 同 gemm_up | 同上 |
| bench_routed_full | **分 4 次 trace 抓**：act_quant / gemm_up / silu / gemm_down | 每段独立 trace 目录 |
| bench_shared_expert | npu_quant_matmul (dense) | `QuantMatmul` |
| bench_grouped_vs_loop | grouped 一次 trace、loop 一次 trace | 各自独立 |

**关键约束**：

- `bench_routed_full --msprof` 必须**分段抓**（每段独立 trace 目录），不要在同一次 profiler context 内跑完整 pipeline（不然各段无法切开）。如果想要 pipeline 整体硬件时间，单独再跑一次「routed_full_pipeline」trace
- `bench_grouped_vs_loop --msprof` 必须**分两次抓**（grouped 一次、loop 一次）。loop 内部有 6 次 quant_matmul 调用，msprof 会把 6 行都记下来，按 `matched_rows` 划分 1 次 vs 6 次
- 找不到 op pattern 时 `parse_op_summary` 自动 dump `op_types_seen.txt`

### 3.11 `report.py`（v4 — 加双轨对比表）

```markdown
# NPU MoE Microbench Summary (v4)

> ⚠ Python 层数字反映 SGLang/KT eager 模式真实端到端调用成本
> ⚠ msprof 数字为 NPU 硬件 device time, 不含 Python/driver/launch overhead
> ⚠ 两者差额 = launch overhead, 是 NPUGraph / kernel fusion 优化的目标量

## Python 层 (Event 计时, D4 数据)
| segment | dev_mean_us | host_mean_us | util_vs_achievable |
| ... 同 v3 ...

## 硬件层 (msprof Level1, D5 数据)
| segment | op_pattern | device_mean_us | p50 | p95 | p99 | max | n |
| act_quant_post | DynamicQuant       | ...            | ... | ... | ... | ... | 10 |
| gemm_up        | GroupedMatmul/QM   | ...            | ... | ... | ... | ... | 10 |
| silu_mul_unf   | Silu+Mul+DynQuant  | ...            | ... | ... | ... | ... | 30 |
| gemm_down      | 同上                | ...            | ... | ... | ... | ... | 10 |
| shared_expert  | QuantMatmul        | ...            | ... | ... | ... | ... | 10 |

## Launch Overhead 拆解 (核心结论)
| segment    | python_event_us | msprof_device_us | launch_overhead_us | overhead_pct |
|------------|-----------------|------------------|--------------------|--------------|
| act_quant  | 166             | T_hw             | py - hw            | overhead/py  |
| gemm_up    | 4274            | T_hw             | py - hw            | overhead/py  |
| gemm_down  | 4185            | T_hw             | py - hw            | overhead/py  |
| silu_mul   | 350             | T_hw             | py - hw            | overhead/py  |
| shared     | 979             | T_hw             | py - hw            | overhead/py  |

> overhead_pct > 50% → Python launch 主导, 优化方向 = NPUGraph / op fusion
> overhead_pct < 30% → NPU kernel 主导, 优化方向 = kernel 内部 / 算法

## Roofline 对照 (硬件层)
| segment | device_us (hw) | lb @ 1.2 TB/s measured | util_vs_measured |
|---------|----------------|------------------------|------------------|
| gemm_up    | T_hw | 80 µs  | T_hw / 80   |
| gemm_down  | T_hw | 40 µs  | T_hw / 40   |
| shared     | T_hw | 20 µs  | T_hw / 20   |

> ⚠ 若 gemm_down util_vs_measured 仍 ≈ 1.0, 真实接近带宽上限
> ⚠ 若 gemm_down util_vs_measured 明显 < 1.0, D4 的 99.78% 主要由 Python overhead 贡献

## Grouped vs Loop (S2, 硬件层验证)
| path    | hw_device_us (mean ± std) | python_event_us | overhead_pct |
| grouped | T_hw_g                     | 4508            | ...          |
| loop    | T_hw_l (6 op 总和)         | 2071            | ...          |

> D4 显示 loop 比 grouped 快 2× (Python 层); 硬件层应验证:
>   (A) 若硬件层 loop ≈ grouped → 差异来自 grouped fallback 的 Python 调度
>   (B) 若硬件层 loop < grouped → 真有 kernel 差异, grouped 实现欠优化
```

---

## 4. 环境变量（不变）

`env.sh` 同 v3。

---

## 5. 已知风险与 fallback（v4 新增）

| 风险 | fallback |
|------|----------|
| `parse_op_summary` op_pattern 不 match | 自动 dump `op_types_seen.txt`；D5.1 看实际 OP Type 调 pattern |
| msprof trace 体积大 (单段几十 MB ~ 几百 MB) | Level1 而非 Level2；`with_stack=False`；`active=10` 不要无限大 |
| msprof 抓 trace 时性能开销 5-15% | 不与 Python Event 数字直接比绝对值；只看「同一份 trace 内 op 间差距」和「python vs hw 差额」 |
| CANN 版本不同时 schedule/handler API 微差 | D5.0 smoke 时若 API 不存在自动 fallback 到无 schedule 的 `with profile: for _ in range(N): fn()` 形态 |
| (v3 留存) grouped matmul fallback 到 loop | ops_runner 单点改；msprof 时分别给 grouped/loop pattern |

---

## 6. 里程碑

| 阶段 | 主要交付 | 状态 |
|------|----------|------|
| **D0 配置校验** | yaml 实测对齐 | ✅ |
| **D1 synthetic + shape** | config + synthetic + assert + roofline | ✅ |
| **D2 NPU smoke** | init_npu + ops_runner + HBM 校准 1.2 TB/s + gate-first | ✅ |
| **D3 8 段 bench + sanity** | act_quant×2 / gemm_up / silu×2 / gemm_down / routed_full / shared / grouped_vs_loop | ✅ |
| **D4 报告 + roofline + sweep** | summary_decode.md (Event 计时) + n_active sweep | ✅ |
| **D5 纯硬件 msprof** | 见下方子里程碑 D5.0-D5.6 | 🟡 v4 主交付 |
| **D6（可选）** | A6/A7: prefill / multi-token / bf16 baseline | ⏸ D5 之后 |

### D5 子里程碑（v4 主交付，按顺序执行）

| 子项 | 内容 | 自检 | 输出物 |
|------|------|------|--------|
| **D5.0** | 落 `msprof_runner.py`；yaml 加 `msprof` 段；8 段 bench CLI 加 `--msprof` | `python -c "from moe_bench.msprof_runner import run_with_msprof"` | 代码 |
| **D5.1** | dry-run smoke：n_active=6 跑 `bench_gemm_up --msprof --quick-mode`，确认 trace + op_summary 解析；如 pattern 不 match 看 `op_types_seen.txt` 调 | trace 落盘 + `op_types_seen.txt` 列出 op；最终 JSON 含 device_mean_us | 1 trace + JSON |
| **D5.2** | 8 段正式 msprof 跑：n_active=6 每段独立抓 | 8 个 JSON 含 `mode: "msprof_hardware_only"` | 8 JSON |
| **D5.3** | comparison 报告：复用 D4 Event 数字 + D5.2 硬件数字，落 `report.py` 双轨表 | `results/msprof_vs_python_comparison.md` 含 launch_overhead_pct | comparison.md |
| **D5.4** | Roofline 用硬件数字重算：gemm_up/down/shared 的 util_vs_measured | summary 表 Roofline 列填上 | 更新 |
| **D5.5** | n_active sweep msprof：{1..6} 每个跑一次 gemm_up + gemm_down + routed_full 三段 | 18 个 JSON；硬件 util 随 n_active 变化曲线 | `results/msprof_n_active_sweep.md` |
| **D5.6** | Grouped vs Loop 硬件层确认：grouped / loop 各跑一次 msprof | 判定 D4 的 loop 快 2× 是 kernel 差异还是 Python overhead | `results/msprof_grouped_vs_loop.md` |

**D5 终局判定矩阵**：

| 关键问题 | msprof 数据 | 结论 |
|---|---|---|
| `gemm_down util_eff=99.78%` 真实吗？ | msprof device_us / lb_measured > 0.8 | 真打满；优化方向 = batching / 算法 |
| 同上 | < 0.5 | D4 Event 数字含大量 launch overhead，真实 kernel 还有空间 |
| Loop 快 2× 真的吗？ | hw_loop ≈ hw_grouped | 是 grouped fallback 的 Python 路径差异；优化 grouped API 实现可解 |
| 同上 | hw_loop << hw_grouped | NPU kernel 本身 grouped 实现差；改 kernel |
| Overhead 占比？ | overhead_pct > 50% | NPUGraph 是首要优化 |
| 同上 | overhead_pct < 30% | kernel 内部是瓶颈 |

---

## 7. 首跑命令（v4）

### 7.1 D5.0-D5.1: msprof 路径冒烟

```bash
cd tools/moe_microbench
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
source env.sh

# 单元测试 msprof_runner 能 import
python -c "from moe_bench.msprof_runner import run_with_msprof, parse_op_summary, list_op_types_in_trace; print('OK')"

# D5.1: dry-run msprof, n_active=6 跑 gemm_up 快速冒烟
python -m moe_bench.bench_gemm_up --msprof --msprof-out ./npu_results --quick-mode \
    --out results/gemm_up_msprof_smoke.json

# 检查 trace 落盘 + op type 可解析
ls ./npu_results/gemm_up/
cat results/gemm_up_msprof_smoke.json | python -m json.tool
```

若 `parse_op_summary` 报错找不到 pattern：
```bash
cat ./npu_results/gemm_up/op_types_seen.txt
# 根据实际看到的 op 名修 bench_gemm_up.py 里的 pattern
```

### 7.2 D5.2: 8 段正式 msprof

```bash
for seg in act_quant gemm_up silu_mul gemm_down routed_full shared_expert grouped_vs_loop; do
    python -m moe_bench.bench_$seg --msprof --out results/${seg}_msprof.json
done
```

**关键**：
- `bench_act_quant --msprof` 同次输出 pre/post 两条 hw 数据
- `bench_silu_mul --msprof` 默认 unfused；显式跑 `--fused --msprof` 测 fused 路径
- `bench_routed_full --msprof` 分段抓（4 段独立 trace 目录）
- `bench_grouped_vs_loop --msprof` 分两次抓

### 7.3 D5.3-D5.4: comparison + roofline 重算

```bash
python -m moe_bench.report --mode comparison \
    --msprof-jsons results/*_msprof.json \
    --event-jsons results/{act_quant,gemm_up,silu_mul,gemm_down,routed_full,shared_expert,grouped_vs_loop}.json \
    --out results/msprof_vs_python_comparison.md
```

### 7.4 D5.5: n_active sweep msprof

```bash
for N in 1 2 3 4 5 6; do
    for seg in gemm_up gemm_down routed_full; do
        python -m moe_bench.bench_$seg --n-active-experts $N --msprof \
            --msprof-out ./npu_results/n_active_$N \
            --out results/${seg}_msprof_n${N}.json
    done
done

python -m moe_bench.report --mode msprof-n-active-sweep \
    --inputs results/*_msprof_n*.json \
    --out results/msprof_n_active_sweep.md
```

### 7.5 一键脚本

```bash
bash run_msprof.sh                          # D5.2 + D5.3 + D5.4
SWEEP_N_ACTIVE="1 2 3 4 5 6" bash run_msprof.sh   # 含 D5.5 sweep
```

---

## 8. 与 attention microbench 的对照

| 维度 | attn_microbench v4 | moe_microbench v4 |
|------|---------------------|---------------------|
| 主要 op | `npu_sparse_attn_sharedkv` + `npu_quant_lightning_indexer` | `npu_grouped_matmul`(×2, fallback loop) + `npu_quant_matmul` |
| Synthetic 难点 | page_table 物理 id + sinks + indexer scale squeeze | weight scale + group_list cumsum + grouped fallback |
| 共用模块 | timing.py / roofline.py / **msprof_runner.py**（NOTE 手动同步） |
| Python 层计时 | ✅ | ✅ |
| **纯硬件 msprof** | ✅ (P1.7) | ✅ (D5, v4 主交付) |
| 当前阶段 | P1.7 (msprof 抓硬件) | D5 (msprof 抓硬件) |

---

## 9. 关于 Claude 建议的处理记录

### v1 → v2（A1-A8, S1-S8）

| ID | 决策 | 实施位置 |
|----|------|----------|
| A1-A8 / S1-S8 | 见 v2 表 | 全部 ✅ |

### v2 → v3（N1-N9）

| ID | 决策 | 实施位置 |
|----|------|----------|
| N1-N9 | 见 v3 表 | 全部 ✅ |

### v3 → v4（纯硬件 msprof 路径）

| ID | 决策 | 落点 |
|----|------|------|
| **D4 util_eff = 99.78% 待验证** | gemm_down 看似打满，但 Event 含 launch overhead；硬件 util 可能更低 | D5.4 重算 |
| **Loop 快 2× 待验证** | D4 显示 grouped→loop 加速 2×；可能是 fallback 路径在 Python 层多调度，不一定是 kernel | D5.6 硬件层确认 |
| **D5 (新增)** | msprof Level1 抓纯硬件 device time | §0.5 / §3.8 / §6 / §7 全篇 |
| **launch_overhead 拆解** | report.py 加 comparison 表，量化 Python vs HW 差额 | §3.11 |
| **roofline 硬件对照** | 用 msprof device time 算 util_vs_measured | §3.11 + D5.4 |
| **msprof_runner.py** | 与 timing.py 双轨并存，不替换；与 attn 同结构 (NOTE 同步) | §3.8 |
| **yaml.msprof 段** | profiler_level / skip_first / warmup / active / out_dir / record_shapes | §2 |

---

*v4 — 2026-05-26：在 v3 基础上加入纯硬件 msprof 测量路径（D5）；与 attn_microbench v4 对齐双轨方法学；可能推翻 D4 的 util_eff 99.78% 结论，把 ~4185 µs 拆解为 Python launch overhead vs NPU kernel device time。*