# Synthetic 32k Attention Microbench — 实施计划 v4

> **v3 → v4 更新（2026-05-26，加入纯硬件 msprof 测量路径）**
> - **§0.6 新增**：纯硬件算子时间测量目标（msprof Level1，解析 op_summary CSV）
> - **§3.9 新增**：`msprof_runner.py`（`run_with_msprof()` + `parse_op_summary()`）
> - **§3.10 增强**：三类 `bench_*.py` 加 `--msprof / --msprof-out`，与 Event 计时**双轨并存**
> - **§3.11 增强**：`report.py` 加 `msprof_vs_python_comparison` 表，量化 launch overhead
> - **§6 新增**：P1.7 子里程碑 — msprof 抓硬件时间，可能推翻 P1.5 "kernel floor" 结论
> - **§9 新增**：P1.5 结论标"待 P1.7 验证"
>
> **v2 → v3 更新留存**：见 §9 决策表
> **v1 → v2 更新留存**：见 §9 决策表

---

## 0. 范围与背景

### 0.0 NPU 环境与卡选择（与 v2 相同）

```bash
npu-smi info                                # 看 HBM-Usage / Process id 列
export ASCEND_RT_VISIBLE_DEVICES=0          # 选空闲卡
cd tools/attn_microbench && source env.sh   # 自动 source CANN + 对齐 p27_launch_ds4flash_npu.sh
```

详见 `README.md`。算子注册依赖 CANN `custom_ops` 包（不在 `sgl_kernel_npu.so`），`env.sh` 已对齐
`tools/p27_launch_ds4flash_npu.sh` 的 `IS_DEEPSEEK_V4` / `USE_PA_*` / `ASCEND_USE_FIA` /
`LD_LIBRARY_PATH=/usr/local/kml/lib`。

### 0.1 微基准目标（v4 扩展）

用 Synthetic KV + page table + metadata，直调与 SGLang 生产相同的 `torch.ops.custom.*`：

| 层类型 | compress_ratio | 算子通路 |
|--------|----------------|----------|
| SWA (L0/L1, 2 层) | 1 | `npu_sparse_attn_sharedkv`（c1a metadata） |
| CSA (≈20 层) | 4 | `npu_quant_lightning_indexer` → `npu_sparse_attn_sharedkv`（c4a） |
| HCA (≈20 层) | 128 | `npu_sparse_attn_sharedkv`（c128a，`cmp_sparse_indices=None`） |

默认：**decode 单 token**，`seq_len=32768`，`batch_size=1`，主 KV bf16。

**双轨测量（v4）**：
1. **Python 层 end-to-end** (`torch.npu.Event` 计时)：反映 SGLang eager 模式下生产真实 launch + dispatch + kernel 总成本
2. **纯硬件 device time** (`torch_npu.profiler` Level1)：反映 kernel 在 AICore/Cube 上真实执行时间，不含 Python/driver/launch overhead

两个数字差额 = NPU launch overhead，是 NPUGraph / kernel fusion 等优化的目标量。

### 0.2 不在范围

- 主干 (`third_party/sglang`, `ascend_backend.py`, scheduler) **不改**
- QKV 线性 / HC / MoE / 端到端 ITL 不测
- FP8 主 KV 路径搁置（NPU kernel 限制）
- 不做数值 gold 对比（`--sanity` 仅 NaN / 量级检查）

### 0.3 已知缺陷与修复状态

| ID | 位置 | 缺陷 | 状态 |
|----|------|------|------|
| **B1** | `page_table.py` | swa/c4/c128 公式越界 → 越界访存 | ✅ 修；invariant: physical page id |
| **B2** | PLAN.md / synthetic | `seqused_kv` 文档笔误 | ✅ 修；invariant: `[B]` 不是 `[seq_len]` |
| **B3** | `synthetic.li_key*` | 用 bf16 / squeeze 维错 → `aclnnQuantLightningIndexer failed` | ✅ 修；invariant: int8 + scale 4D + squeeze(-2) |
| **B4** | `synthetic.cmp_kv_c4` | 第二维语义二义 (128 token 域 vs 32 compressed 域) | ⚠️ TBD；P1.3 dump 生产 buffer 钉死（已 waive，见 §6） |
| **B5** | 主 KV dtype | bf16-only (NPU kernel 限制) | 🟡 README 声明，~2× HBM 偏离 |
| **B6** | RoPE 不分维 | synthetic 全 randn | 🟡 README 声明 |
| **B7** | `bench_swa.sinks` | 论文未明 SWA 是否带 sink | ✅ 默认 -inf (≡ 无 sink)，`--with-sink` 切换 |

### 0.4 P0 诊断结论（2026-05-26，repeat=100, warmup=30, NPU 0）

**§4 旧数字（503 µs indexer / 1.9× ratio）作废**，原因：

#### (A) seq_len sweep — indexer 平坦 ⚠️

| seq_len | c4_cols | indexer (µs) | SWA attn (µs) |
|---------|---------|--------------|---------------|
| 1024 | 256 | 250.3 ± 19.4 | 129.2 ± 5.1 |
| 4096 | 1024 | 251.7 ± 6.8 | 127.8 ± 5.0 |
| 32768 | 8192 | 244.5 ± 4.6 | 124.1 ± 3.1 |

SWA 平坦符合 window=128 预期；**indexer 在 c4 候选数 256 → 8192 (32×) 下仍 ~245 µs**，不符物理预期。

#### (B) 极端 seqused_kv — 未通过 ⚠️

| seqused_kv (S=32k) | indexer (µs) |
|--------------------|--------------|
| 32768 | 244.4 ± 3.8 |
| 128 | 240.8 ± 3.3 |

#### (C) actual_seq_lengths_key — 无差异

| key 长度 | indexer (µs) |
|----------|--------------|
| 32768 (token_len) | 242.7 ± 3.7 |
| 8192 (c4_len) | 242.8 ± 3.7 |

#### P0 诊断结论（三种假说待 P1 区分）

| 假说 | 排除方法 | 状态 |
|------|----------|------|
| (i) metadata 字段语义错（seqused_kv 应传别的字段） | P1.3 dump 生产传参对照 | 🟡 P1.3 waived；P1.4 已基于静态对照修 strided block_table |
| (ii) page_table 物理 page 复用，kernel 只读一份 KV | P1.2 page_table unique pages 重测 | ✅ 排除（diff 0.08%） |
| (iii) kernel 有固定 launch floor (~240 µs) | P1.1 极端小 c4_cols 重测 + **P1.7 msprof 拆分** | 🟡 P1.1 部分排除（spread 9.9%）；**P1.7 拆分 Python vs 硬件 才能终局** |

### 0.5 P1 当前可对外陈述（v4 更新）

**可以说**：
- 三类 NPU attention 算子在 32k synthetic 布局下均可 forward（无 NaN，shape 正确）
- SWA Python 层 sweep 平坦符合 sliding-window 预期，侧面印证测量框架可用
- **Python 层 end-to-end 单算子调用** 在 seq 1k → 32k 下平坦在 ~250–300 µs；这反映 SGLang eager 模式下每次 op 调用的真实成本
- P1.4 已修 strided block_table（对齐 ascend_backend）；P1.5 修复后 sweep 仍平坦 → Python 层 floor 实锤

**不能说**（直至 P1.7 完成）：
- 这 ~250 µs floor 是 **NPU kernel 硬件 floor** 还是 **Python launch overhead floor** ← P1.7 拆分
- 32k indexer/attn 在 **NPU 硬件层** 的真实算子时间（待 msprof）
- 「indexer kernel 接近 roofline」or「indexer kernel 仍有大量优化空间」（待 msprof + roofline 比对）
- 与 token#200 msprof 的数值对比（seq_len 不匹配；P2 完成后可对比）

### 0.6 P1.7 目标：纯硬件算子时间测量

**为什么必要**：P1.5 的 "Python 层 floor 250 µs" 结论无法回答"NPU kernel 本身要算多久"。两个可能：

1. **kernel 真的需要 ~200+ µs**（kernel 硬件 floor）→ 优化方向：改 kernel 内部 / 改算法
2. **kernel 只要 ~20-50 µs，剩下 200+ µs 是 Python + driver + launch**（Python launch floor）→ 优化方向：NPUGraph / kernel fusion / batch op call

两个方向的工作量和价值天差地别。**只有 msprof Level1 trace 能区分**。

**方法**：用 `torch_npu.profiler` 包住单 op 调用，从 `op_summary_*.csv` 的 `Task Duration(us)` 列读硬件 device time。

**与现有 Event 计时关系**：**并存，不替换**。Event 计时仍代表 eager 模式生产成本，msprof 代表 kernel 硬件极限，两者差额 = launch overhead。

---

## 1. 工作空间结构（v4）

```text
tools/attn_microbench/
├── README.md                       (含 §"已知偏离" 顶部 banner)
├── IMPLEMENTATION_PLAN.md          (本文 v4)
├── env.sh
├── run_all.sh
├── run_diag.sh                     (P0/P1 诊断, 走 yaml.diag 旋钮)
├── run_msprof.sh                   (P1.7 新增: 一键三类 msprof + comparison 报告)
├── config/dsv4_flash.yaml          (含 invariants / diag / roofline / msprof 段)
└── attn_bench/
    ├── __init__.py
    ├── config.py                   (含 metadata_keys / invariants / diag / msprof dataclass)
    ├── init_npu.py                 (log_versions + print env)
    ├── page_table.py               (B1 修复; 支持 diag.page_table_unique_pages)
    ├── synthetic.py                (B3/B4 修复 + S6 assert + diag overrides)
    ├── metadata.py                 (S7: 从 yaml.metadata_keys 读)
    ├── ops_runner.py
    ├── timing.py                   (Python 层: Event mean±std/p95/p99/max + host wall)
    ├── msprof_runner.py            (P1.7 新增: 纯硬件 device time)
    ├── sanity.py                   (S9)
    ├── roofline.py                 (P1 加入; 与 moe_microbench 共用算法; 文件头 NOTE 同步)
    ├── bench_swa.py                (+ --msprof / --msprof-out)
    ├── bench_csa.py                (+ --skip-indexer + --msprof)
    ├── bench_hca.py                (+ --msprof)
    ├── bench_diag.py               (P0 落地; 跑 yaml.diag 各 override 组合)
    ├── reference_check.py          (B4 小 case: page128 vs page32 + topk 粗对比)
    └── report.py                   (S1 衍生列 + msprof_vs_python_comparison 表 + isolated_device_sum_us)
```

---

## 2. 配置文件（v4 新增 msprof 段）

详见 `config/dsv4_flash.yaml`。要点：

- `model.softmax_scale`：**显式** 1/√512
- `runtime.dtypes`：q/ori_kv/cmp_kv/out/indexer_kv/indexer_weight 分别声明
- `bench.repeat=1000`，`quick_mode_repeat=100`
- `layer_layout_info`：L0/L1=SWA，L2-L42 c4/c128 交替，L43 待 P1.3 dump（已 waived）
- `metadata_keys`：每行注释源（ascend_backend.py / nsa_indexer.py）
- `invariants`：B1-B7 修复点固化
- `diag`：P0/P1 诊断旋钮（默认全 null / false）
- `roofline`：与 moe_microbench 共用字段
- **`msprof` (v4 新增)**：profiler 参数（profiler_level / skip_first / warmup / active / out_dir）

```yaml
# v4 新增 msprof 段示例
msprof:
  enabled: false              # CLI --msprof 触发, 默认关闭 (避免普通 bench 误抓 trace)
  profiler_level: "Level1"    # Level0=op only, Level1=op+task, Level2=完整 pipeline (体积大)
  aic_metrics: "PipeUtilization"
  skip_first: 5               # 避开 kernel 首次构建 + metadata 编译的 cold start
  warmup: 2                   # profiler 自身暖机
  active: 10                  # 抓 10 次 active 求均值
  out_dir: "./npu_results"    # trace 落盘根目录, 不进 git
  record_shapes: true         # trace 里能看每 op 输入 shape (调试 metadata 时关键)
  with_stack: false           # decode 单步 stack 太多, trace 会爆几百 MB
```

---

## 3. 模块规格（v4 增改）

### 3.1 `config.py`

`BenchConfig` 新增字段：

```python
@dataclass(frozen=True)
class BenchConfig:
    # ... v3 字段 ...
    msprof: dict                    # v4: profiler 参数
```

### 3.2-3.8 与 v3 相同

详见 v3 PLAN。

### 3.9 `msprof_runner.py`（v4 新增 — 核心）

```python
"""纯 NPU 硬件算子时间测量 (Level1 trace + op_summary CSV 解析).
- 不是 Event 计时, 不含 Python overhead.
- 与 timing.py 双轨并存, 共同回答 'kernel 硬件 vs Python 端到端' 拆分.
"""
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
    返回 device duration 的 mean / std / p50 / p95 / p99 / max / count.
    """
    csvs = glob.glob(str(trace_dir / "*/ASCEND_PROFILER_OUTPUT/op_summary_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no op_summary CSV under {trace_dir}")

    df = pd.read_csv(csvs[0])
    matched = df[df['OP Type'].str.contains(op_pattern, case=False, na=False)]
    if matched.empty:
        # 列出所有 op type 帮助用户调 pattern
        avail = sorted(df['OP Type'].unique().tolist())
        types_dump = trace_dir / "op_types_seen.txt"
        types_dump.write_text("\n".join(avail))
        raise ValueError(
            f"no op match {op_pattern!r}; "
            f"saw {len(avail)} unique op types, dumped to {types_dump}"
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

### 3.10 `bench_*.py` CLI 增强（v4 — 加 `--msprof`）

每个 bench 都加：

```python
parser.add_argument("--msprof", action="store_true",
                    help="用 torch_npu.profiler 抓纯硬件 device time, 不是 Python end-to-end")
parser.add_argument("--msprof-out", default=None,
                    help="msprof trace 落盘根目录; 默认 cfg.msprof.out_dir")
```

#### `bench_csa.py` 关键逻辑（其他 bench 类似）

```python
if args.msprof:
    from attn_bench.msprof_runner import run_with_msprof, parse_op_summary
    out_dir = args.msprof_out or cfg.msprof["out_dir"]
    kw = dict(
        out_dir=out_dir,
        skip_first=cfg.msprof["skip_first"],
        warmup=cfg.msprof["warmup"],
        active=cfg.msprof["active"],
        profiler_level=cfg.msprof["profiler_level"],
        aic_metrics=cfg.msprof["aic_metrics"],
        record_shapes=cfg.msprof["record_shapes"],
        with_stack=cfg.msprof["with_stack"],
    )

    # 关键: indexer / attn 必须分两次独立抓 trace, 不要混在一起 (避免分析时按时间戳切分)
    trace_idx = run_with_msprof(
        lambda: run_csa_indexer(t, cfg, meta),
        name=f"csa_indexer_seq{cfg.seq_len}", **kw,
    )
    indexer_stats = parse_op_summary(trace_idx, "QuantLightningIndexer")

    # 拿 topk 给 attn 用 (static), 避免每次 attn 调用前都触发 indexer
    topk_static = run_csa_indexer(t, cfg, meta)

    trace_attn = run_with_msprof(
        lambda: run_csa_attn(t, cfg, meta, topk_static),
        name=f"csa_attn_seq{cfg.seq_len}", **kw,
    )
    attn_stats = parse_op_summary(trace_attn, "SparseAttnSharedkv")

    out = {
        "kind": "csa",
        "mode": "msprof_hardware_only",
        "seq_len": cfg.seq_len,
        "indexer_hw": indexer_stats,
        "attn_hw": attn_stats,
        "trace_dirs": [str(trace_idx), str(trace_attn)],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
else:
    # 原 Event 计时路径不变
    ...
```

**关键约束**：

- `bench_csa --msprof` 必须**分两次独立抓** trace（indexer 一次、attn 一次），不要混在一次 profiler context 里
- `bench_swa --msprof` 只抓 SWA attn
- `bench_hca --msprof` 只抓 HCA attn
- 如果 `parse_op_summary` 找不到 op pattern，会自动 dump `op_types_seen.txt` 让你调 pattern

### 3.11 `report.py`（v4 — 加双轨对比表）

```markdown
# Attention Microbench Summary (v4)

> ⚠ Python 层数字反映 SGLang eager 模式真实端到端调用成本
> ⚠ msprof 数字为 NPU 硬件 device time, 不含 Python/driver/launch overhead
> ⚠ 两者差额 = launch overhead, 是 NPUGraph / kernel fusion 优化的目标量

## Python 层 (Event 计时)
| kind | seq_len | indexer (µs, mean±std) | attn (µs, mean±std) | isolated_sum |
|------|---------|------------------------|---------------------|--------------|
| swa  | 32768   | -                       | T_swa_py ± σ        | T_swa_py     |
| csa  | 32768   | T_li_py ± σ            | T_csa_py ± σ        | T_li_py + T_csa_py |
| hca  | 32768   | -                       | T_hca_py ± σ        | T_hca_py     |

## 硬件层 (msprof Level1)
| kind | op            | device_mean_us | p50  | p95  | p99  | max  | n  |
|------|---------------|----------------|------|------|------|------|-----|
| swa  | SparseAttnSharedkv (c1a) | T_swa_hw    | ...  | ...  | ...  | ...  | 10 |
| csa  | QuantLightningIndexer     | T_li_hw     | ...  | ...  | ...  | ...  | 10 |
| csa  | SparseAttnSharedkv (c4a)  | T_csa_hw    | ...  | ...  | ...  | ...  | 10 |
| hca  | SparseAttnSharedkv (c128a)| T_hca_hw    | ...  | ...  | ...  | ...  | 10 |

## Launch Overhead 拆解 (核心结论)
| op          | python_event_us | msprof_device_us | launch_overhead_us | overhead_pct |
|-------------|-----------------|------------------|--------------------|--------------|
| swa_attn    | T_swa_py        | T_swa_hw         | py - hw            | overhead/py  |
| csa_indexer | T_li_py         | T_li_hw          | py - hw            | overhead/py  |
| csa_attn    | T_csa_py        | T_csa_hw         | py - hw            | overhead/py  |
| hca_attn    | T_hca_py        | T_hca_hw         | py - hw            | overhead/py  |

> overhead_pct > 50% → Python launch 是主要瓶颈, 优化方向 = NPUGraph / op fusion
> overhead_pct < 30% → NPU kernel 是主要瓶颈, 优化方向 = kernel 内部 / 算法
> 中间区间 → 两边都要看

## Roofline 对照 (hw 层)
| op          | device_us | hw_lower_bound_us (effective) | util_vs_achievable |
|-------------|-----------|-------------------------------|---------------------|
| swa_attn    | T_swa_hw  | 0.1 (128 KB @ 1 TB/s)        | 极低, launch 主导   |
| csa_indexer | T_li_hw   | ~1 (1 MB int8 @ 1 TB/s)      | util = T_li_hw / lb |
| csa_attn    | T_csa_hw  | ~9 (8.4 MB bf16 @ 1 TB/s)    | ...                 |
| hca_attn    | T_hca_hw  | ~0.3 (262 KB @ 1 TB/s)       | ...                 |

Per-layer estimate (硬件层):
  2 × T_swa_hw + ~20 × (T_li_hw + T_csa_hw) + ~20 × T_hca_hw
  → 对照 msprof token#200 整层均值 ~26 µs, 可验证是否一致
```

---

## 4. 32k Synthetic 形状速查（v4，含 dtype）

| Tensor | Shape | dtype |
|--------|-------|-------|
| `q` | `[1, 64, 512]` | bf16 |
| `sinks` | `[64]` | fp32（默认 -inf） |
| `seqused_kv` | `[1]` = 32768 | int32 |
| `cu_seqlens_q_pa` | `[2]` = `[0,1]` | int32 |
| `ori_kv` | `[2, 128, 1, 512]` | bf16 |
| `swa_page_table` | `[1, 32768]` ∈ `[0,2)` | int32 |
| `cmp_kv_c4` | `[64, 128, 1, 512]` *(B4 TBD)* | bf16 |
| `c4_page_table` | `[1, 64]` strided physical page ids ∈ `[0,64)` (P1.4 修复) | int32 |
| `cmp_kv_c128` | `[2, 128, 1, 512]` | bf16 |
| `c128_page_table` | `[1, 2]` strided ∈ `[0,2)` | int32 |
| `li_query` | `[1, 64, 128]` | bf16 |
| `li_key` | `[64, 128, 1, 128]` | **int8** |
| `li_key_scale` | `[64, 128, 1, 1]` → `squeeze(-2)` → `[64, 128, 1]` | float16 |
| `li_weights` | `[1, 64]` | float16 |

---

## 5. 风险与 fallback（v4）

| 风险 | fallback |
|------|----------|
| B4 cmp_kv 第二维语义不对 | P1.3 dump waived 后，靠 msprof + sanity 间接判定；公式备选 `(pos // 32) % num_pages` for c4 |
| li_key int8 与 kernel 期望不一致 | `--li-key-dtype bf16` 退回，`--skip-indexer` 仅测 attn |
| metadata 字段升级 | 改 `yaml.metadata_keys`（单点） |
| repeat=1000 仍抖 | `--repeat 2000`；输出 p95+std |
| **P1.7 msprof op_summary 找不到 op pattern** | `parse_op_summary` 自动 dump `op_types_seen.txt`；调 pattern 重跑 |
| **P1.7 msprof trace 体积爆炸 (GB级)** | 用 `Level1` 而非 `Level2`；`with_stack=False`；`active=10` 而非更高 |
| **msprof 抓 trace 时性能开销 5–15%** | 不与 Python Event 数字直接比绝对值；只看「同一份 trace 内相对比例」和「python vs hw 差额」|
| HBM 带宽估错 | `roofline.measure_hbm_effective_tb_s` 校准回填 yaml |

---

## 6. 里程碑

| 阶段 | 主要交付 | 状态 |
|------|----------|------|
| **D0 文档同步** | PLAN v2; B1-B7 写入 | ✅ |
| **D1 修 page_table + synthetic** | B1 公式 / B3 int8 / S6 assert | ✅ |
| **D2 timing/sanity/version** | S2/S8/S9/S7 | ✅ |
| **D3 NPU smoke** | 三类 sanity + repeat=10 | ✅ |
| **D4 计时 + 报告** | 32k 三类 repeat=300 → §4 旧表 | ✅ 但 **数字已作废** |
| **P0 诊断** | seq sweep / 极端 seqused_kv / key_len 变体；§0.4 结论 | ✅ |
| **P1.0-P1.6 metadata 钉死** | F1/F2/F3 + c4_cols sweep + unique_pages + strided block_table + repeat=1000 | ✅（P1.3 dump waived） |
| **P1.7 纯硬件 msprof** | 见下方子里程碑 | 🟡 进行中 (v4 主交付) |
| **P2 同 seq msprof 对照** | seq_len≈200 同时跑 microbench + 新 msprof | ⏸ P1.7 之后 |
| **P3 sweep / prefill / bf16 baseline** | batch × seq × q_len | ⏸ P2 之后 |

### P1.7 子里程碑（v4 主交付，按顺序执行）

| 子项 | 内容 | 自检 | 输出物 |
|------|------|------|--------|
| **P1.7.0** | 落 `msprof_runner.py`；yaml 加 `msprof` 段；三类 bench CLI 加 `--msprof` | 单元: `python -c "from attn_bench.msprof_runner import run_with_msprof"` 不报错 | 代码 |
| **P1.7.1** | dry-run msprof：seq=1024 跑一次 `bench_swa --msprof --quick-mode`，确认 trace 落盘 + op_summary CSV 可解析 | `op_types_seen.txt` 列出至少 1 个 SparseAttnSharedkv 相关 op | trace + JSON |
| **P1.7.2** | 三类 32k msprof 正式跑：`bench_swa/csa/hca --msprof --seq-len 32768` | 三个 trace 目录非空；JSON 含 device_mean_us | `results/{swa,csa,hca}_msprof.json` |
| **P1.7.3** | 生成 comparison 报告：跑一次 Python Event 计时（已有 P1.5 结果可直接复用），跑一次 msprof，落 `report.py` 双轨表 | `results/msprof_vs_python_comparison.md` 含 4 行 op + launch_overhead 列 | comparison.md |
| **P1.7.4** | seq sweep msprof：`{1024, 4096, 8192, 16384, 32768}` 每个 seq 跑一次 msprof，看硬件层是否仍平坦 | 5 个 seq 的 indexer device_mean_us 列出；判定硬件 floor 是否存在 | `results/msprof_seq_sweep.md` |
| **P1.7.5** | Roofline 对照：硬件 device time vs `roofline.measure_hbm_effective_tb_s` 计算的下界 | `report.py` Roofline 表填上 util_vs_achievable | summary 更新 |
| **P1.7.6** | 改写审核报告 §12：基于 P1.7 数据明确结论（NPU kernel floor / Python launch floor / 两者都有） | §12 含 launch_overhead_pct 数字 + 优化方向建议 | 报告更新 |

**P1.7 终局判定矩阵**：

| msprof device time | 结论 | 下一步 |
|---|---|---|
| ~20–50 µs (远 < 250 µs) | **Python launch overhead 主导**；NPU kernel 可能已接近 roofline | 后续优化方向 = NPUGraph / kernel fusion |
| ~150–250 µs (接近 Python) | **NPU kernel 自身 floor 主导**；microbench Python 数字基本反映硬件 | 后续优化方向 = kernel 内部 / 算法压缩 |
| ~80–150 µs (中间) | 两边都要看；分项 sub-kernel breakdown | 进一步拆 op_summary 看 sub-kernel |

---

## 7. 首跑命令（v4）

### 7.1 P1.7.0–P1.7.1: msprof 路径冒烟

```bash
cd tools/attn_microbench
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
source env.sh

# 单元测试 msprof_runner 能 import
python -c "from attn_bench.msprof_runner import run_with_msprof, parse_op_summary, list_op_types_in_trace; print('OK')"

# P1.7.1: dry-run msprof, seq=1024 跑 SWA 快速冒烟
python -m attn_bench.bench_swa --seq-len 1024 --msprof --msprof-out ./npu_results --out results/swa_msprof_smoke.json

# 检查 trace 落盘 + op type 可解析
ls ./npu_results/swa_attn_seq1024/
cat results/swa_msprof_smoke.json | python -m json.tool
```

**若 `parse_op_summary` 报错找不到 pattern**：

```bash
# 看 op_types_seen.txt 选合适的 pattern
cat ./npu_results/swa_attn_seq1024/op_types_seen.txt
# 然后在 ops_runner.py 或 yaml 里把 pattern 改对
```

### 7.2 P1.7.2: 三类 32k msprof 正式跑

```bash
python -m attn_bench.bench_swa --seq-len 32768 --msprof --out results/swa_msprof.json
python -m attn_bench.bench_csa --seq-len 32768 --msprof --out results/csa_msprof.json
python -m attn_bench.bench_hca --seq-len 32768 --msprof --out results/hca_msprof.json
```

### 7.3 P1.7.3: comparison 报告

```bash
# Python Event 计时（如已有 P1.5 的 results/*_r1000.json 可跳过）
SEQ_LEN=32768 REPEAT=1000 bash run_all.sh

# 生成对比报告
python -m attn_bench.report --inputs \
    results/swa_msprof.json results/csa_msprof.json results/hca_msprof.json \
    results/swa.json results/csa.json results/hca.json \
    --out results/msprof_vs_python_comparison.md
```

### 7.4 P1.7.4: seq sweep msprof

```bash
for S in 1024 4096 8192 16384 32768; do
    python -m attn_bench.bench_csa --seq-len $S --msprof \
        --msprof-out ./npu_results/seq_$S \
        --out results/csa_msprof_seq_$S.json
done

python -m attn_bench.report --msprof-sweep \
    --inputs results/csa_msprof_seq_*.json \
    --out results/msprof_seq_sweep.md
```

### 7.5 一键脚本（推荐）

```bash
bash run_msprof.sh                                              # 默认 32k 三类
SEQ_LEN_SWEEP="1024 4096 8192 16384 32768" bash run_msprof.sh   # 含 seq sweep
```

---

## 8. 与 moe_microbench 的对照

| 维度 | attn_microbench v4 | moe_microbench v3 |
|------|---------------------|---------------------|
| 主要 op | `npu_sparse_attn_sharedkv` + `npu_quant_lightning_indexer` | `npu_grouped_matmul`(×2) + `npu_quant_matmul` |
| Synthetic 难点 | page_table 物理 id + sinks + indexer scale squeeze | weight scale + group_list cumsum |
| 共用模块 | timing.py / roofline.py / **msprof_runner.py**（NOTE 手动同步，不抽 _common/） |
| Python 层计时 | ✅ | ✅ |
| **纯硬件 msprof** | ✅ (v4 新增) | ⏸ 待 moe 也加（推荐复用 msprof_runner.py） |
| 当前阶段 | P1.7 (msprof 抓硬件时间) | D1 (yaml 落定，待开发) |

---

## 9. 历史决策与作废记录

### v1 → v2（B1-B7, S1-S10）

| ID | 决策 | 落点 |
|----|------|------|
| B1 | page_table 物理 page id | ✅ page_table.py 全部公式 |
| B2 | seqused_kv shape `[B]` | ✅ 文档 / synthetic |
| B3 | li_key int8 + scale 4D + squeeze(-2) | ✅ ops_runner / synthetic |
| B4 | cmp_kv 第二维语义 | ⚠️ TBD，P1.3 dump 钉死（已 waived，待 msprof 间接判定） |
| B5 | bf16 主 KV | 🟡 README 声明，不切 |
| B6 | RoPE 不分维 | 🟡 README 声明 |
| B7 | sinks 默认 -inf | ✅ bench_swa --with-sink 切换 |
| S1-S10 | report 衍生列 / repeat=300 / sanity / 等 | ✅ 全部落地 |

### v2 → v3（P0 + reviewer F1-F3 / J1-J3）

| ID | 决策 | 落点 |
|----|------|------|
| **§4 旧数字** | 503 µs indexer / 1.9× ratio | ❌ **作废**；§0.4 + §0.5 修订 |
| **F1** L43 nextn 猜测 | TBD，可能是 MTP head / padding | ✅ yaml.layer_layout_info 标 TBD |
| **F2** §5.3 outdated 预算检查 | 整段加 strike | ✅ 报告 §5.3 顶部 ⚠ |
| **F3** SWA sinks 假设性 | 默认 -inf 是 microbench 选择不是事实 | ✅ yaml.invariants + 报告 §3.1 footnote |
| **J1** "未按 KV 长度缩放" 结论过强 | 改为「三种假说待排除」 | ✅ §0.4 + P1.1-1.3 |
| **J2** kernel floor 假说没排除 | P1.1 极端 c4_cols sweep | ✅ yaml.diag.override_c4_cols + P1.1 |
| **J3** §4 vs §11.2 数字差 2× | warmup/repeat 调整解决，需 P0 报告说明 | ✅ 报告 §11.1 顶部声明 |
| **O1** mean±std | ✅ timing.py / report.py 已改 |
| **O2** P1 拆细 | ✅ P1.0-P1.6 子里程碑 |
| **Y1** softmax_scale 显式 | ✅ yaml.model |
| **Y2** invariants 段固化 | ✅ yaml.invariants |
| **Y3** diag 段暴露旋钮 | ✅ yaml.diag |
| **Y4** metadata_keys 注释源 | ✅ yaml.metadata_keys |
| **Y5** layers → layer_layout_info | ✅ yaml |
| **Y6-Y11** dtypes 拆分 / quick_mode / roofline / banner | ✅ yaml |

### v3 → v4（纯硬件 msprof 路径）

| ID | 决策 | 落点 |
|----|------|------|
| **P1.5 "kernel floor 确认"** | 改为 **"Python 层端到端 floor 确认"**；硬件层结论待 P1.7 验证 | §0.5 + §6 P1.7 + §9 |
| **P1.3 dump 阻塞** | waived（静态对照已 sufficient + P1.7 msprof 补足审计） | §6 标 waived |
| **P1.7 (新增)** | msprof Level1 抓纯硬件 device time | §0.6 / §3.9 / §6 / §7 全篇 |
| **launch_overhead 拆解** | report.py 加 comparison 表，量化 Python vs HW 差额 | §3.11 |
| **roofline 硬件对照** | 用 msprof device time 算 util_vs_achievable | §3.11 + P1.7.5 |
| **msprof_runner.py** | 与 timing.py 双轨并存，不替换 | §3.9 |
| **yaml.msprof 段** | profiler_level / skip_first / warmup / active / out_dir | §2 |

---

*v4 — 2026-05-26：在 v3 基础上加入纯硬件 msprof 测量路径（P1.7）；可能推翻 P1.5 "kernel floor" 结论，把 ~250 µs 拆解为 Python launch overhead vs NPU kernel device time。*