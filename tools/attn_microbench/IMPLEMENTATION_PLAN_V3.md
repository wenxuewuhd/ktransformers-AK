# Synthetic 32k Attention Microbench — 实施计划 v3

> **v2 → v3 更新（2026-05-26，整合 P0 诊断 + 外部 review F1-F3 / J1-J3）**
> - **§0.4**: 加入 P0 诊断结论；§4 旧数字（503 µs indexer / 1.9× ratio）**作废**
> - **§0.5**: 当前可/不可对外陈述边界（来自审核报告 §11.4，补 reviewer F1-F3 修正）
> - **§2 / yaml**: softmax_scale 显式、dtypes 拆分、metadata_keys 加注释、layer_layout_info 更正、invariants/diag 段固化
> - **§3.10 新增**: `bench_diag.py` + `reference_check.py` 已落，diag 旋钮全部从 yaml.diag 读取
> - **§7**: D0–D5 + P0 已完成；本版本主交付为 **P1（metadata 钉死）**，含 P1.1-P1.6 子里程碑
> - **§9 新增**: 历史决策与作废记录表（B1-B7、S1-S10、F1-F3、J1-J3）
>
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

### 0.1 微基准目标

用 Synthetic KV + page table + metadata，直调与 SGLang 生产相同的 `torch.ops.custom.*`：

| 层类型 | compress_ratio | 算子通路 |
|--------|----------------|----------|
| SWA (L0/L1, 2 层) | 1 | `npu_sparse_attn_sharedkv`（c1a metadata） |
| CSA (≈20 层) | 4 | `npu_quant_lightning_indexer` → `npu_sparse_attn_sharedkv`（c4a） |
| HCA (≈20 层) | 128 | `npu_sparse_attn_sharedkv`（c128a，`cmp_sparse_indices=None`） |

默认：**decode 单 token**，`seq_len=32768`，`batch_size=1`，主 KV bf16。

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
| **B4** | `synthetic.cmp_kv_c4` | 第二维语义二义 (128 token 域 vs 32 compressed 域) | ⚠️ TBD；P1 dump 生产 buffer 钉死 |
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

| 假说 | 排除方法 |
|------|----------|
| (i) metadata 字段语义错（seqused_kv 应传别的字段） | P1.3 dump 生产传参对照 |
| (ii) page_table 物理 page 复用，kernel 只读一份 KV | P1.2 page_table unique pages 重测 |
| (iii) kernel 有固定 launch floor (~240 µs) | P1.1 极端小 c4_cols 重测 |

**当前数据不能区分 (i)/(ii)/(iii)**，P1 必须按 P1.1 → P1.2 → P1.3 顺序排除。

### 0.5 当前可对外陈述

**可以说**：
- 三类 NPU attention 算子在 32k synthetic 布局下均可 forward（无 NaN，shape 正确）
- SWA sweep 平坦符合 sliding-window 预期，侧面印证测量框架可用
- 修复 B1/B2/B3 后 indexer 调用通过 kernel 校验

**不能说**（直至 P1 完成）：
- 32k indexer/attn 任何 **绝对值**
- 「indexer 比 attn 重 X 倍」「indexer 占主要瓶颈」
- 与 token#200 msprof 的数值对比（seq_len 不匹配，且 microbench 数字本身不可信）
- 「skip-indexer 176 µs ≈ 真实 CSA attn」（topk 分布不同）
- 「L43 是 nextn」（待 dump 确认，可能是 MTP head / padding）

---

## 1. 工作空间结构（v3）

```text
tools/attn_microbench/
├── README.md                       (含 §"已知偏离" 顶部 banner)
├── IMPLEMENTATION_PLAN.md          (本文 v3)
├── env.sh
├── run_all.sh
├── run_diag.sh                     (P0/P1 诊断, 走 yaml.diag 旋钮)
├── config/dsv4_flash.yaml          (含 invariants / diag / roofline 段)
└── attn_bench/
    ├── __init__.py
    ├── config.py                   (含 metadata_keys / invariants / diag dataclass)
    ├── init_npu.py                 (log_versions + print env)
    ├── page_table.py               (B1 修复; 支持 diag.page_table_unique_pages)
    ├── synthetic.py                (B3/B4 修复 + S6 assert + diag overrides)
    ├── metadata.py                 (S7: 从 yaml.metadata_keys 读)
    ├── ops_runner.py
    ├── timing.py                   (S2/S8: mean±std/p95/p99/max + host wall)
    ├── sanity.py                   (S9)
    ├── roofline.py                 (P1 加入; 与 moe_microbench 共用算法; 文件头 NOTE 同步)
    ├── bench_swa.py                (B7 + --batch-size + --sanity + --with-sink)
    ├── bench_csa.py                (+ --skip-indexer)
    ├── bench_hca.py
    ├── bench_diag.py               (P0 落地; 跑 yaml.diag 各 override 组合)
    ├── reference_check.py          (B4 小 case: page128 vs page32 + topk 粗对比)
    └── report.py                   (S1 衍生列 + mean±std + isolated_device_sum_us + util 双列)
```

---

## 2. 配置文件

详见 `config/dsv4_flash.yaml`。要点：

- `model.softmax_scale`：**显式** 1/√512
- `runtime.dtypes`：q/ori_kv/cmp_kv/out/indexer_kv/indexer_weight 分别声明
- `bench.repeat=1000`，`quick_mode_repeat=100`
- `layer_layout_info`：L0/L1=SWA，L2-L42 c4/c128 交替，L43 待 P1 dump
- `metadata_keys`：每行注释源（ascend_backend.py / nsa_indexer.py）
- `invariants`：B1-B7 修复点固化
- `diag`：P0/P1 诊断旋钮（默认全 null / false）
- `roofline`：与 moe_microbench 共用字段

---

## 3. 模块规格（v3 增改）

### 3.1 `config.py`

`BenchConfig` 新增字段：

```python
@dataclass(frozen=True)
class BenchConfig:
    # ... v2 字段 ...
    dtypes: dict                    # q/ori_kv/cmp_kv/out/indexer_kv/indexer_weight
    softmax_scale: float            # 显式 (v3)
    metadata_keys: dict
    invariants: dict                # 历史 bug 修复点 (v3)
    diag: dict                      # 诊断旋钮 (v3)
    roofline: dict                  # 含 hbm_peak/effective (v3)
    quick_mode_repeat: int = 100
    quick_mode_warmup: int = 10
```

### 3.2 `page_table.py`（B1 + P1.2 支持）

```python
def build_swa_page_table(spec, device, cfg):
    """[B, S] physical page ids ∈ [0, swa_num_pages)."""
    table = torch.zeros((spec.batch_size, spec.swa_cols), dtype=torch.int32, device=device)
    win = min(spec.page_size * spec.swa_num_pages, spec.seq_len)
    start = spec.seq_len - win
    rel = torch.arange(win, device=device, dtype=torch.int32)
    table[0, start:] = (rel // spec.page_size) % spec.swa_num_pages
    return table

def build_c4_page_table(spec, device, cfg):
    """[B, c4_cols] physical page ids ∈ [0, c4_num_pages).
    支持 cfg.diag.page_table_unique_pages: True 时禁用模运算复用 (P1.2)."""
    cols = spec.c4_cols
    pos = torch.arange(cols, device=device, dtype=torch.int32)
    if cfg.diag.get("page_table_unique_pages", False):
        # 每 logical column 单独占一个 physical page; 排除 page 复用假说
        assert spec.c4_num_pages >= cols // spec.page_size, \
            f"unique-page 模式需要 c4_num_pages >= {cols // spec.page_size}"
        return (pos // spec.page_size).clamp_max_(spec.c4_num_pages - 1).unsqueeze(0)
    return ((pos // spec.page_size) % spec.c4_num_pages).unsqueeze(0)
```

`build_c128_page_table` 同理。

### 3.3 `synthetic.py`（B3 + B4 + S6 + diag.override_c4_cols）

```python
def build_synthetic(cfg) -> SyntheticTensors:
    # 应用 diag overrides
    c4_cols = cfg.diag.get("override_c4_cols") or compute_c4_cols(cfg.seq_len)
    seqused_kv_val = cfg.diag.get("override_seqused_kv") or cfg.seq_len
    actual_key_len = cfg.diag.get("override_actual_seq_lengths_key") or cfg.seq_len
    ...
    # B3: li_key int8 + scale 4D
    li_key = torch.randint(-127, 128, (c4_pages, page_size, kv, idx_dim), dtype=torch.int8)
    li_key_scale = torch.full((c4_pages, page_size, kv, 1), 0.01, dtype=torch.float16)
    ...

def assert_shapes(cfg, t):
    # ... S6 全套 ...
    assert t.swa_page_table.max().item() < spec.swa_num_pages
    assert t.c4_page_table.max().item() < spec.c4_num_pages
    assert t.c128_page_table.max().item() < spec.c128_num_pages
    assert t.seqused_kv.shape == (cfg.batch_size,)
    assert t.li_key.dtype == torch.int8
    assert t.li_key_scale.shape == (spec.c4_num_pages, cfg.page_size, cfg.num_heads_kv, 1)
    # invariants 防回归
    assert cfg.invariants["page_table_value_domain"] == "physical_page_id"
    assert cfg.invariants["li_key_scale_squeeze_dim"] == -2
```

### 3.4 `ops_runner.py`（B3 squeeze）

```python
def run_csa_indexer(t, cfg, meta):
    q, q_scale = torch_npu.npu_dynamic_quant(t.li_query)
    topk, _ = torch.ops.custom.npu_quant_lightning_indexer(
        query=q, key=t.li_key,
        key_dequant_scale=t.li_key_scale.squeeze(cfg.invariants["li_key_scale_squeeze_dim"]),
        ...
    )
    return topk
```

squeeze 维度从 invariants 读，避免 hardcode。

### 3.5 `timing.py`（S2/S8 已落地）

输出 `device_mean/p50/p95/p99/max/std + host_mean + n`。文件头：

```python
# NOTE: keep in sync with tools/moe_microbench/moe_bench/timing.py
# Changes here should be mirrored manually (no shared module by design).
```

### 3.6 `roofline.py`（P1 新增，与 moe 共用算法）

```python
# NOTE: keep in sync with tools/moe_microbench/moe_bench/roofline.py.

def swa_kv_bytes(cfg):
    """SWA decode: window × num_heads_kv × head_dim × dtype_size."""
    return cfg.sliding_window_size * cfg.num_heads_kv * cfg.head_dim * 2  # bf16

def c4_kv_bytes(cfg, c4_cols):
    return c4_cols * cfg.num_heads_kv * cfg.head_dim * 2

def c128_kv_bytes(cfg):
    cols = cfg.seq_len // 128
    return cols * cfg.num_heads_kv * cfg.head_dim * 2

def utilizations(actual_us, bytes_, cfg):
    lb_peak = bytes_ / (cfg.roofline["hbm_peak_tb_s"] * 1e12) * 1e6
    lb_eff  = bytes_ / (cfg.roofline["hbm_effective_tb_s"] * 1e12) * 1e6
    return {"lb_peak_us": lb_peak, "lb_effective_us": lb_eff,
            "util_vs_peak": actual_us / lb_peak if lb_peak else None,
            "util_vs_achievable": actual_us / lb_eff if lb_eff else None}

def measure_hbm_effective_tb_s(device, size_mb=256):
    # 同 moe_bench.roofline.measure_hbm_effective_tb_s
    ...
```

**关键**：SWA bytes = 128 × 1 × 512 × 2B = **128 KB**。这点流量在 240 µs 内能传 ~400 MB。
SWA 测出 ~125 µs 远高于 roofline (~0.1 µs) → 强证据是 **kernel launch floor 主导**，与 P0/P1.1 假说 (iii) 一致。

### 3.7 `bench_diag.py`（P0 已落地，P1 复用）

走 yaml.diag 旋钮，分别跑：

```python
# 已支持组合 (P0)
- seq_len sweep: {1024, 4096, 8192, 16384, 32768}
- override_seqused_kv: {seq_len, 128}
- override_actual_seq_lengths_key: {token_len, c4_len}

# P1 新增组合
- override_c4_cols: {4, 16, 64, 256, 1024, 8192}    # P1.1
- page_table_unique_pages: {false, true}             # P1.2
```

输出 `results/diag_<scenario>.json`，每条记录附 `yaml.diag` snapshot 便于回看。

### 3.8 `report.py`（v3）

```markdown
# Attention Microbench Summary

> ⚠ 数字为 isolated eager 上界, **不能直接外推生产 per-layer**
> ⚠ csa/hca 已含 SWA branch; attn_compressed_only = csa.attn - swa.attn
> ⚠ total → isolated_device_sum_us (非端到端 pipeline latency)

| kind | seq_len | indexer (µs, mean±std) | attn (µs, mean±std) | attn_compressed_only | isolated_sum | lb_eff | util_vs_achievable |
|------|---------|------------------------|---------------------|----------------------|--------------|--------|---------------------|
| swa  | 32768   | -                       | T_swa ± σ          | -                    | T_swa        | 0.1 µs | (n/a, floor 主导)   |
| csa  | 32768   | T_li ± σ               | T_csa ± σ          | T_csa - T_swa        | T_li + T_csa | ...    | ...                 |
| hca  | 32768   | -                       | T_hca ± σ          | T_hca - T_swa        | T_hca        | ...    | ...                 |

Per-layer estimate (informational only, not validated against msprof):
  2 × T_swa + ~20 × (T_li + T_csa) + ~20 × T_hca
```

---

## 4. 32k Synthetic 形状速查（v3，含 dtype）

| Tensor | Shape | dtype |
|--------|-------|-------|
| `q` | `[1, 64, 512]` | bf16 |
| `sinks` | `[64]` | fp32（默认 -inf） |
| `seqused_kv` | `[1]` = 32768 | int32 |
| `cu_seqlens_q_pa` | `[2]` = `[0,1]` | int32 |
| `ori_kv` | `[2, 128, 1, 512]` | bf16 |
| `swa_page_table` | `[1, 32768]` ∈ `[0,2)` | int32 |
| `cmp_kv_c4` | `[64, 128, 1, 512]` *(B4 TBD)* | bf16 |
| `c4_page_table` | `[1, 8192]` ∈ `[0,64)` | int32 |
| `cmp_kv_c128` | `[2, 128, 1, 512]` | bf16 |
| `c128_page_table` | `[1, 256]` ∈ `[0,2)` | int32 |
| `li_query` | `[1, 64, 128]` | bf16 |
| `li_key` | `[64, 128, 1, 128]` | **int8** |
| `li_key_scale` | `[64, 128, 1, 1]` → `squeeze(-2)` → `[64, 128, 1]` | float16 |
| `li_weights` | `[1, 64]` | float16 |

---

## 5. 风险与 fallback（v3）

| 风险 | fallback |
|------|----------|
| B4 cmp_kv 第二维语义不对 | P1.3 dump 生产 buffer 后改 `page_table.py` + `synthetic.py` 单点；公式备选 `(pos // 32) % num_pages` for c4 |
| li_key int8 与 kernel 期望不一致 | `--li-key-dtype bf16` 退回，`--skip-indexer` 仅测 attn |
| metadata 字段升级 | 改 `yaml.metadata_keys`（单点） |
| repeat=1000 仍抖 | `--repeat 2000`；输出 p95+std |
| P1.1-1.2 仍平坦 | 落 P1.3 dump 生产；若 dump 后仍未拉开则归因 kernel floor (假说 iii)，对外只报 isolated 上界 |
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
| **P1 metadata 钉死** | 见下方 P1.1-P1.6 | 🟡 进行中 |
| **P2 同 seq msprof 对照** | seq_len≈200 同时跑 microbench + 新 msprof | ⏸ P1 之后 |
| **P3 sweep / prefill / bf16 baseline** | batch × seq × q_len | ⏸ P2 之后 |

### P1 子里程碑（按顺序执行）

| 子项 | 内容 | 自检 | 假说排除 |
|------|------|------|----------|
| **P1.0** | 修 F1（L43 标 TBD 不写 nextn）/ F2（§5.3 outdated 加 strike）/ F3（sinks footnote） | 文档 grep "nextn" 无；§5.3 顶部有 ⚠ | — |
| **P1.1** | `override_c4_cols` sweep `{4,16,64,256,1024,8192}` | indexer 是否随 c4_cols 变化 | 排除 (iii) kernel floor |
| **P1.2** | `page_table_unique_pages=true` 重测 32k | indexer 是否拉开 | 排除 (ii) page 复用 |
| **P1.3** | 用 monkey-patch print 脚本 dump 一次生产 `forward_npu_dsv4_fusion` 完整 kwargs | 拿到 dump JSON | — |
| **P1.4** | 字段对照 microbench vs dump，修 metadata.py / synthetic.py | diff 结果落入 `results/p1_field_diff.md` | (i) metadata 语义 |
| **P1.5** | P1.1-1.4 修复后重跑 seq sweep `{1k, 4k, 8k, 16k, 32k}`，repeat=1000 | indexer 应见单调 / 拐点 | — |
| **P1.6** | 更新 `results/summary_32768_r1000.md` + 审核报告 §11.4 改写 | 数字带 mean±std + util_vs_achievable | — |

---

## 7. 首跑命令（v3）

```bash
cd tools/attn_microbench
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
source env.sh

# 形状校验 (无 NPU)
DRY_RUN=1 SEQ_LEN=32768 bash run_all.sh

# P1.1: c4_cols 极端 sweep (排除 kernel floor)
DIAG_OVERRIDE_C4_COLS="4 16 64 256 1024 8192" bash run_diag.sh

# P1.2: unique page 重测
DIAG_PAGE_TABLE_UNIQUE=1 bash run_diag.sh

# P1.5: 修完后正式 sweep
SEQ_LEN_SWEEP="1024 4096 8192 16384 32768" REPEAT=1000 bash run_all.sh
```

---

## 8. 与 moe_microbench 的对照

| 维度 | attn_microbench v3 | moe_microbench v3 |
|------|---------------------|---------------------|
| 主要 op | `npu_sparse_attn_sharedkv` + `npu_quant_lightning_indexer` | `npu_grouped_matmul`(×2) + `npu_quant_matmul` |
| Synthetic 难点 | page_table 物理 id + sinks + indexer scale squeeze | weight scale + group_list cumsum |
| 共用模块 | timing.py / roofline.py（NOTE 手动同步，不抽 _common/） |
| 当前阶段 | P1 (P0 后 metadata 钉死) | D1 (yaml 落定，待开发) |

---

## 9. 历史决策与作废记录

### v1 → v2（B1-B7, S1-S10）

| ID | 决策 | 落点 |
|----|------|------|
| B1 | page_table 物理 page id | ✅ page_table.py 全部公式 |
| B2 | seqused_kv shape `[B]` | ✅ 文档 / synthetic |
| B3 | li_key int8 + scale 4D + squeeze(-2) | ✅ ops_runner / synthetic |
| B4 | cmp_kv 第二维语义 | ⚠️ TBD，P1.3 dump 钉死 |
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

---

*v3 — 2026-05-26：P0 诊断完成；本版主交付为 P1 (metadata 钉死)。*