# DeepSeek-V4-Flash NPU Attention Microbench — 审核报告

> **用途**：供外部 reviewer（Claude 等）评估「Synthetic 32k attention microbench + 生产 msprof 对照」的方法论与结论是否合理。  
> **工作区**：`tools/attn_microbench/`（独立小工程，**不修改** `third_party/sglang` / `kt-kernel` 主干）  
> **环境**：Ascend 910B1，CANN + `torch_npu 2.8.0.post2`，`custom_ops`（CANN DSv4 算子包）  
> **日期**：2026-05-26

---

## 1. 背景与目标

### 1.1 动机

生产 decode profiling（msprof，token #200 eager）显示：

- 整步 NPU busy **~28 ms**（43 层 forward）
- `SparseAttnSharedkv` 合计 **1.12 ms / 43 次** → 折算 **~26 µs/层**
- Attention 大类占 NPU **~4.6%**，远低于 MatMul（~45%）

无法在 msprof 中单独拆解 **SWA / CSA（c4+indexer）/ HCA（c128）** 三类 attention 子路径，也无法在 32k 上下文下隔离 **Lightning Indexer vs Sparse Attn** 耗时。

### 1.2 Microbench 目标

用 **Synthetic KV + page table + metadata**，直调与 SGLang 生产相同的 `torch.ops.custom.*` 算子：

| 层类型 | compress_ratio | 算子通路 |
|--------|----------------|----------|
| SWA | 1 | `npu_sparse_attn_sharedkv`（c1a metadata） |
| CSA | 4 | `npu_quant_lightning_indexer` → `npu_sparse_attn_sharedkv`（c4a） |
| HCA | 128 | `npu_sparse_attn_sharedkv`（c128a，`cmp_sparse_indices=None`） |

默认：**decode 单 token**，`seq_len=32768`，`batch_size=1`，主 KV **bf16**（FP8 主 KV 路径搁置）。

---

## 2. 生产 Attention 通路（对照基准）

### 2.1 融合 Sparse Attention

SGLang NPU decode（`USE_PA_DECODE=1`）在 `ascend_backend.py` 中调用：

```python
o, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
```

`attn_kwargs` 含：`q`, `ori_kv`, `ori_block_table`, `cmp_kv`, `cmp_block_table`, `cmp_sparse_indices`, `seqused_kv`, `metadata`, sliding window 等。

### 2.2 CSA Lightning Indexer

`nsa_indexer.py::forward_npu_dsv4_fusion`（`LI_KV_DTYPE_INT8=1` 路径）：

```python
q, q_scale = torch_npu.npu_dynamic_quant(q)
topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(
    query=q,
    key=k,                              # c4_index int8 paged KV
    key_dequant_scale=k_scale.squeeze(-2),
    actual_seq_lengths_query=actual_seq_lengths_q,
    actual_seq_lengths_key=actual_seq_lengths_kv,
    block_table=c4_page_table,
    layout_query="TND",
    layout_key="PA_BSND",
    weights=weights.to(torch.float16),
    query_dequant_scale=q_scale.to(torch.float16),
    cmp_ratio=4,
    query_quant_mode=0,
    key_quant_mode=0,
    sparse_mode=3,
    sparse_count=self.index_topk,
    metadata=li_quant_metadata,
)
```

**关键对齐点**：`key_dequant_scale` 必须对 **`[pages, page_size, 1, 1]`** scale buffer 做 **`squeeze(-2)`**，得到 **`[pages, page_size, 1]`**。Microbench 初版误用 `squeeze(-1)` 导致 `aclnnQuantLightningIndexer failed`。

### 2.3 算子注册

DSv4 sparse attn / quant lightning indexer **不在** `sgl_kernel_npu.so`，而在 CANN **`custom_ops`** 包（与 `deepseek_v4.py` 中 `import custom_ops` 一致）。

环境需对齐 `tools/p27_launch_ds4flash_npu.sh`：

- `IS_DEEPSEEK_V4=1`, `USE_PA_DECODE=1`, `ASCEND_USE_FIA=1`, `LI_KV_DTYPE_INT8=1` 等
- `LD_LIBRARY_PATH=/usr/local/kml/lib` + `$PYTHON/site-packages/torch/lib`
- `import` 顺序：**先** `torch_npu`，**再** `custom_ops`

---

## 3. Microbench 核心代码结构

```text
tools/attn_microbench/
├── env.sh                 # NPU 环境 + 卡选择（对齐 p27 launch）
├── run_all.sh             # SWA / CSA / HCA 一键跑 + report
├── config/dsv4_flash.yaml # 模型维度、page_size=128、index_topk=512
└── attn_bench/
    ├── init_npu.py        # custom_ops 注册 + 版本日志
    ├── synthetic.py       # Synthetic 张量 + shape assert
    ├── page_table.py      # SWA/c4/c128 page table（B1 修复：物理 page id）
    ├── metadata.py        # npu_*_metadata 构建
    ├── ops_runner.py      # 三类算子 forward（生产 kwargs 对齐）
    ├── timing.py          # NPU Event device_us + host wall
    ├── bench_swa.py / bench_csa.py / bench_hca.py
    └── report.py          # summary 表 + attn_compressed_only 衍生列
```

### 3.1 三类算子入口（`ops_runner.py`）

**SWA** — 仅 c1a，sinks 默认 `-inf`（无 sink）[^swa-sinks]：

[^swa-sinks]: 默认 `-inf` 是 microbench 选择（≡ 无 sink），**非**论文或生产 ckpt 的既定事实；生产 SWA sinks 待 dump 确认。可用 `--with-sink` 切回 0。

```python
torch.ops.custom.npu_sparse_attn_sharedkv(metadata=meta.c1a, **_attn_common(...))
```

**HCA** — c128a，`cmp_sparse_indices=None`：

```python
torch.ops.custom.npu_sparse_attn_sharedkv(
    metadata=meta.c128a, cmp_ratio=128, cmp_kv=t.cmp_kv_c128,
    cmp_block_table=t.c128_page_table, cmp_sparse_indices=None, ...
)
```

**CSA Indexer** — 对齐 `forward_npu_dsv4_fusion`：

```python
q, q_scale = torch_npu.npu_dynamic_quant(t.li_query)
topk, _ = torch.ops.custom.npu_quant_lightning_indexer(
    query=q, key=t.li_key,
    key_dequant_scale=t.li_key_scale.squeeze(-2),  # scale: [P,128,1,1] → [P,128,1]
    actual_seq_lengths_query=t.actual_seq_lengths_q,   # decode: [1]
    actual_seq_lengths_key=t.seqused_kv,                 # [32768]
    block_table=t.c4_page_table, metadata=meta.li_quant, ...
)
```

**CSA Attn** — c4a + topk：

```python
torch.ops.custom.npu_sparse_attn_sharedkv(
    metadata=meta.c4a, cmp_ratio=4,
    cmp_sparse_indices=topk.view(-1, 1, 512), ...
)
```

### 3.2 Synthetic 32k 关键形状

| 张量 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `q` | `[1, 64, 512]` | bf16 | decode 单 token |
| `ori_kv` | `[2, 128, 1, 512]` | bf16 | SWA paged（2 pages × window） |
| `swa_page_table` | `[1, 32768]` | int32 | 末 256 位置有效 page id |
| `seqused_kv` | `[1]` = 32768 | int32 | 非 `[32768]` 向量 |
| `cmp_kv_c4` | `[64, 128, 1, 512]` | bf16 | c4 压缩 KV（B4：第二维语义 TBD） |
| `c4_page_table` | `[1, 8192]` | int32 | seq/4 逻辑列 |
| `li_key` | `[64, 128, 1, 128]` | **int8** | indexer paged key |
| `li_key_scale` | `[64, 128, 1, 1]` | fp16 | **squeeze(-2)** 传入 indexer |
| `li_query` | `[1, 64, 128]` | bf16 | indexer query（forward 前 dynamic quant） |

### 3.3 计时方法（`timing.py`）

- 每个 `fn` **独立** warmup + repeat（indexer / attn **分开**计时，非单次端到端链）
- **device_us**：`torch.npu.Event` `elapsed_time` × 1000
- **host_us**：`perf_counter` 墙钟（含 launch/sync 开销）
- **`total_us`（CSA）**：indexer 与 attn 的 **device 均值相加**，不是 fused pipeline latency

---

## 4. 实验结果

**硬件**：NPU 0（`ASCEND_RT_VISIBLE_DEVICES=0`）  
**采样**：多数实验 `warmup=5, repeat=10`（正式配置 yaml 默认 repeat=300）

### 4.1 32k 三类 Attention（device 均值，µs）

| 测试 | indexer | attn | 合计 | 备注 |
|------|---------|------|------|------|
| **SWA** | — | **143** | 143 | sinks=-inf，纯 sliding window |
| **HCA** | — | **165** | 165 | c128，无 indexer |
| **CSA skip-indexer** | — | **176** | 176 | 随机 topk，仅测 c4 attn |
| **CSA 全路径** | **503** | **271** | **774** | 真实 indexer + c4 attn |

来源：

- `tools/attn_microbench/results/swa.json`, `hca.json`, `csa_attn_only.json`
- `tools/attn_microbench/results/csa_full.json`

**CSA 全路径 sanity**（修复 scale 后）：

- indexer topk：`[1, 512]` int32，无 NaN
- attn out：`[1, 64, 512]` bf16，无 NaN

### 4.2 CSA 子路径拆解（32k）

| 对比 | attn device | 解读 |
|------|-------------|------|
| skip-indexer vs SWA | 176 − 143 ≈ **33 µs** | c4 压缩段增量（粗估，report 列 `attn_compressed_only`） |
| 全路径 vs skip-indexer attn | 271 − 176 ≈ **95 µs** | 真实 topk 改变 sparse attn 访存路径 |
| indexer / attn 比值 | 503 / 271 ≈ **1.9×** | microbench 内 indexer 重于 attn |

### 4.3 seq_len  scaling sweep（CSA 全路径，repeat=20）

| seq_len | c4_cols | c4 pages | indexer (µs) | attn (µs) |
|---------|---------|----------|----------------|-----------|
| 1024 | 256 | 2 | 478 | 276 |
| 4096 | 1024 | 8 | 453 | 262 |
| 8192 | 2048 | 16 | 481 | 275 |
| 16384 | 4096 | 32 | 453 | 262 |
| 32768 | 8192 | 64 | 491 | 270 |

**观察**：c4 域 token 数 **256 → 8192（32×）**，indexer/attn device 时间 **几乎平坦（~450–500 / ~260–275 µs）**。

---

## 5. 与生产 msprof 对照

来源：`doc/zh/DeepSeek-V4-Flash_eager_decode_profiling_分析报告_20260525.md`（token #200，eager，单卡 KT+SGLang）

### 5.1 生产整步 NPU 构成（摘要）

| 指标 | 数值 |
|------|------|
| 墙钟（ProfilerStep） | ~338 ms |
| NPU busy（Computing） | **~28 ms** |
| NPU idle（Free，含 CPU MoE overlap） | ~309 ms |
| `aclnnSparseAttnSharedkv` | **1.12 ms / 43 层** → **~26 µs/层** |
| Attention 大类 | ~1.3 ms（4.5% NPU） |
| MatMul/GEMM | ~12.6 ms（45%） |

### 5.2 绝对值对比（microbench vs 生产）

| 维度 | Microbench（单算子 eager） | 生产 msprof（整图一层均值） | 倍率 |
|------|---------------------------|----------------------------|------|
| Sparse attn | 143–271 µs | ~26 µs/层 | **~5–10×** |
| Indexer | ~503 µs | **未单独出现在 Top kernel** | 无法直接对比 |

### 5.3 预算一致性检查

> **⚠ outdated（2026-05-26）**：本节基于 **已作废的 §4 数字**（503 µs indexer、repeat=10）与 **错误的层数口径** 推算，**不得引用**。P1 完成后重写或删除。

~~DeepSeek-V4-Flash `compress_ratios`（43 层）：**c4=21 层**，c128=20 层，SWA(c1)=3 层。~~

~~若 microbench 的 **503 µs indexer × 21 c4 层 ≈ 10.6 ms**……~~

**层数事实（`DeepSeek-V4-Flash-W8A8/config.json`）**：`len(compress_ratios)=44`；L0–L1 为 SWA(c1)；L2–L42 为 c4/c128 交替；**L43 待 dump 确认（MTP head / padding / nextn 都可能）**，msprof 43 层 forward 是否含 L43 亦待确认。主 stack 预算应使用 **c4≈20 层（L2–L41 中 ratio=4）**，勿混用 21。

**结论（仍有效）**：microbench isolated eager 数字 **不能**直接乘层数解释生产 NPU busy；且须与 msprof **同 seq_len** 对照（token #200 ≈ 几百，非 32k）。

### 5.4 生产 profiling 未显式列出 Indexer 的可能原因（待验证）

1. Indexer 耗时被归入 `Quant/Cast`、`Memory/Index` 等大类，未单独命名
2. NPUGraph / 流水线 overlap 降低可见 device 时间
3. msprof 分类规则未匹配 `QuantLightningIndexer` 字符串
4. Microbench synthetic/metadata 未正确传递「有效 KV 长度」，导致 **seq scaling 平坦**（见 §6.3）

---

## 6. 方法论评估（供 reviewer 重点审）

### 6.1 已做对的部分 ✅

1. **算子通路对齐**：kwargs 与 `ascend_backend.py` / `nsa_indexer.py` 一致（修复 scale 后 indexer 可跑通）
2. **独立可复现**：`tools/attn_microbench/` 不碰主干；`env.sh` 对齐 `p27_launch_ds4flash_npu.sh`
3. **三类 layer 分离**：SWA / CSA / HCA 可独立 benchmark
4. **双计时**：device Event + host wall，区分 launch 开销
5. **Sanity 检查**：输出 shape/dtype/NaN
6. **已知局限文档化**：bf16-only、decode-only、synthetic、csa 含 SWA branch 等（`README.md`）

### 6.2 已知缺陷与修复记录

| ID | 问题 | 状态 |
|----|------|------|
| B1 | page_table 写入 logical index 越界 | ✅ 改为物理 page id |
| B2 | 文档 `seqused_kv` shape 笔误 | ✅ 实际为 `[B]` |
| B3 | `li_key` int8 + `li_key_scale` 四维 + squeeze(-2) | ✅ 已修复 |
| B4 | `cmp_kv_c4` 第二维 128 vs 32 语义 | ⚠️ TBD，attn 已跑通 |
| B5 | FP8 主 KV | 搁置，README 声明 |
| B7 | SWA sinks | 默认 -inf，`--with-sink` 可选 |

**Indexer 失败根因（已修复）**：

```text
错误：li_key_scale [P,128,1] + squeeze(-1) → [P,128]
正确：li_key_scale [P,128,1,1] + squeeze(-2) → [P,128,1]  （与生产一致）
```

### 6.3 科学性疑点 ⚠️（请 reviewer 重点判断）

#### (A) seq_len scaling 平坦

indexer/attn 在 **1k–32k** 几乎不变，与「对 growing KV 做 top-512」的直觉不符。

**可能解释**：

- 算子 launch + metadata tiling **固定成本**主导（~450 µs floor）
- Synthetic `seqused_kv` / `block_table` / metadata 未让 kernel 感知有效长度变化（**bug 或未对齐生产语义**）
- 910B 上 256–8192 c4 slot 的访存仍远小于 compute floor

**影响**：不宜用本 microbench 回答「32k 比 8k indexer 慢多少」。

#### (B) 绝对值 vs 生产差距 ~10×

Microbench sparse attn **143–271 µs** vs msprof **~26 µs/层**。

**可能解释**：

- Microbench = **单算子 eager**，无 NPUGraph
- 生产 msprof = **43 层整 forward 分摊**，且有 overlap / fusion
- Microbench 含 **完整 SWA+c4/c128 融合 kernel**，生产 per-layer 26 µs 是 **所有层类型混合均值**（含大量 c1 轻量层）

#### (C) CSA total_us 非端到端

`indexer_us + attn_us` 是 **两次独立计时之和**，生产可能在同一 stream 内背靠背执行，cache 行为不同。

#### (D) Synthetic 数据

随机 KV / 随机 topk（skip-indexer）/ 无 RoPE 分维 → **数值路径**可能与真 KV 不同；**计时**在算子已跑通前提下仍可能有代表性，但未验证。

---

## 7. 结论（当前可对外陈述的版本）

### 7.1 可以较有信心地说

1. **三类 NPU attention 算子在 32k synthetic 布局下均可 forward**（无 NaN，shape 正确）。
2. **在 isolated eager microbench 中**，CSA 的 **Lightning Indexer（~500 µs）重于 c4 Sparse Attn（~270 µs）**，比值约 **1.9×**。
3. **相对 SWA baseline（~143 µs）**，c4 压缩 attn 增量约 **30–130 µs**（取决于 topk 来源）。
4. **生产 msprof** 仍表明 **Attention 大类仅占 NPU ~4.6%**，MatMul + Compressor 是更大头；microbench **不支持**「indexer 是整步 decode 主要瓶颈」的推论。

### 7.2 目前不应说

1. ❌ 「生产 decode 每层 indexer 需要 500 µs」
2. ❌ 「32k 上下文 indexer 随序列线性变慢」（scaling 数据不支持）
3. ❌ 「microbench attn 143 µs 等于 msprof 26 µs/层」（测量口径不同）
4. ❌ 「skip-indexer 的 176 µs 等于真实 CSA attn」（topk 分布不同）

### 7.3 建议的下一步（若 reviewer 认为方向 OK）

1. 对 `bench_csa` 跑 **msprof**，确认 `aclnnQuantLightningIndexer` 在生产环境下的单层 device 时间
2. 在 **SGLang server 内 hook** `forward_npu_dsv4_fusion`，与 microbench 同 seq_len 对比
3. 排查 **seq scaling 平坦** 是算子特性还是 synthetic/metadata bug（对照 `actual_seq_lengths_key` 是否应传 c4 域长度）
4. 钉死 **B4** `cmp_kv` page 第二维语义（128 vs 32）

---

## 8. 复现命令

```bash
# 选空闲卡
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=0

cd tools/attn_microbench
source env.sh

# 形状校验（无 NPU 计算）
DRY_RUN=1 SEQ_LEN=32768 bash run_all.sh

# 三类 + report（skip-indexer CSA）
REPEAT=10 WARMUP=5 SEQ_LEN=32768 bash run_all.sh

# CSA 全路径（含 indexer）
"${PYTHON_BIN}" -m attn_bench.bench_csa \
  --seq-len 32768 --sanity --repeat 10 --warmup 5 \
  --out results/csa_full.json
```

---

## 9. 请 reviewer 回答的问题

1. **seq scaling 平坦** 是否足以否定 microbench 对 long-context 的结论？还应做哪些对照实验？
2. **indexer 503 µs vs msprof 无独立条目**：更可能是测量口径问题，还是 synthetic 布局问题？
3. **`total_us = indexer + attn` 分开计时** 是否应改为单循环端到端计时？
4. **`cmp_kv_c4` 第二维 128** 在缺少生产 buffer dump 的情况下，当前 attn 跑通是否足够证明 layout 正确？
5. 本报告 §7 的「可以说 / 不应说」边界是否合理？

---

## 10. 参考文件索引

| 文件 | 说明 |
|------|------|
| `tools/attn_microbench/attn_bench/ops_runner.py` | 三类算子 forward |
| `tools/attn_microbench/attn_bench/synthetic.py` | Synthetic 张量 |
| `tools/attn_microbench/attn_bench/timing.py` | 计时 |
| `tools/attn_microbench/env.sh` | NPU 环境 |
| `tools/attn_microbench/results/csa_full.json` | CSA 全路径结果 |
| `tools/attn_microbench/IMPLEMENTATION_PLAN.md` | 实现计划 v2 |
| `doc/zh/DeepSeek-V4-Flash_eager_decode_profiling_分析报告_20260525.md` | 生产 msprof |
| `third_party/sglang/.../nsa_indexer.py` | 生产 indexer 通路 |
| `third_party/sglang/.../ascend_backend.py` | 生产 sparse attn 通路 |
| `tools/p27_launch_ds4flash_npu.sh` | 环境变量基准 |

---

## 11. P0 诊断结果（Claude review 跟进，2026-05-26）

> **§4 vs §11.2 数字差 ~2× 的原因**：§4 为 **warmup=5, repeat=10**（indexer ~503 µs）；§11.2 为 **warmup=30, repeat=100**（indexer ~245 µs）。差异来自 **预热不足 + 小样本方差**，非代码回退。正式数据以 **repeat=1000** 为准。

> **状态**：§4 旧数字（repeat=10）**作废**；正式数据待 `repeat=1000` 重跑（`results/*_r1000.json`）。

### 11.1 新增工具

| 脚本 | 作用 |
|------|------|
| `bash run_diag.sh` | seq sweep + 极端 seqused_kv + key_len 变体 |
| `python -m attn_bench.reference_check --seq-len 512` | B4 page128 vs page32 + indexer topk 粗对比 |
| `attn_bench/report.py` | 输出 **mean ± std**；`isolated_device_sum_us` |

### 11.2 诊断结论（repeat=100, warmup=30, NPU 0）

#### (A) seq_len sweep — indexer 仍平坦 ⚠️

| seq_len | c4_cols | indexer (µs) | SWA attn (µs) |
|---------|---------|--------------|---------------|
| 1024 | 256 | 250.3 ± 19.4 | 129.2 ± 5.1 |
| 4096 | 1024 | 251.7 ± 6.8 | 127.8 ± 5.0 |
| 32768 | 8192 | 244.5 ± 4.6 | 124.1 ± 3.1 |

SWA 平坦 **符合预期**（window=128）；indexer 在 256 vs 8192 candidates 下仍 ~245 µs **不符合物理预期**。

#### (B) 极端 seqused_kv @ 32k — **未通过** ⚠️

| 设置 | seqused_kv | indexer (µs) |
|------|------------|--------------|
| baseline | 32768 | 244.4 ± 3.8 |
| extreme | **128** | 240.8 ± 3.3 |

32768→128 **无显著下降** → **强烈支持 Claude 判断**：当前 synthetic/metadata 下 kernel **未按有效 KV 长度缩放**（固定上界路径或字段未生效）。

#### (C) actual_seq_lengths_key 变体 — 无差异

| 语义 | key 长度 | indexer (µs) |
|------|----------|--------------|
| token_len | 32768 | 242.7 ± 3.7 |
| c4_len | 8192 | 242.8 ± 3.7 |

单独改 `seqused_kv`/metadata 的 key 长度 **不能**拉开耗时。

### 11.3 compress_ratios 更正（config.json）

`DeepSeek-V4-Flash-W8A8/config.json`：`len(compress_ratios)=44`

- L0,L1：ratio=1（**主 stack pure SWA = 2 层**）
- L2–L42：4/128 交替 → **c4×21, c128×20**（L2–L41 中 ratio=4 为 **20 层**）
- L43：ratio=1，**待 dump 确认（MTP head / padding / nextn 都可能）**；msprof 43 层 forward 是否含 L43 待确认

§5.3 预算应用 **20 层 c4**（L2–L41 中 ratio=4），勿混用 21 与「43 层 forward」。

### 11.4 当前可对外陈述（P1.7 后修订）

**可以说**：

- **Python Event 层**（eager 单次 op 调用）：indexer **~265 µs**，CSA attn **~157 µs** @ 32k（repeat=1000）；seq sweep **平坦**（Python 层 floor）
- **msprof 硬件层**（Level1 kernel_details）：indexer **~36 µs**，CSA attn **~39 µs**，SWA **~22 µs** @ 32k；**launch overhead 75–86%**
- **判定 (A) Python launch 主导**：P1.5「kernel floor」实为 **driver/Python 调度 floor**；NPU kernel 本体 ~20–40 µs
- 硬件层 CSA indexer seq sweep **1k→32k 约 2.1×**（16→34 µs）；Python 层仍 ~1.0× — **平坦来自 launch，不是 kernel**
- `c4_page_table` strided 修复已完成（P1.4）

**不能说**：

- Python Event 数字 = NPU kernel 硬件时间（需看 msprof 列）
- 32k **生产 E2E latency**（需 P2 同 seq msprof）
- 与 token#200 msprof 直接数值对比（seq_len 仍不匹配；P2 完成后再比）

### 11.5 下一步（P2）

1. 同 seq_len（~200）microbench hw_us vs 新 msprof 对照
2. NPUGraph / op fusion 验证 launch overhead 可否压掉
3. P1.3 生产 dump（可选审计项，已 waived）

---

## 12. P1.7 终局结论（硬件层 msprof 数据）

### 12.1 硬件 device time vs Python end-to-end (32k decode, repeat=1000 Event)

| op | python_us | hw_us | launch_overhead_us | overhead_pct |
|----|-----------|-------|--------------------|--------------|
| swa_attn | 133.8 ± 6.7 | 21.6 ± 1.2 | 112.3 | **83.9%** |
| csa_indexer | 265.0 ± 13.5 | 36.1 ± 1.6 | 228.9 | **86.4%** |
| csa_attn | 157.2 ± 10.1 | 38.8 ± 1.3 | 118.3 | **75.3%** |
| hca_attn | 136.1 ± 5.7 | 26.5 ± 1.2 | 109.6 | **80.5%** |

数据来源：`results/msprof_vs_python_comparison.md`（Event=`seq_32768/*.json`，msprof=`*_msprof.json`）

### 12.2 判定

**结论 (A) Python launch 主导** — 四类 op overhead_pct 均 **>75%**。

- P1.5 的 ~250 µs「floor」主要来自 **Python/driver/launch overhead**，不是 NPU kernel 硬件 floor
- NPU kernel 实际：**indexer ~36 µs，attn ~22–39 µs**（msprof active=10 均值）
- Roofline：`util_vs_achievable` **0.03–0.13**（hw 层远未打满 HBM 带宽）→ 瓶颈在 launch/调度，非 memory bound
- **后续优化方向**：NPUGraph / kernel fusion / 减少 eager op dispatch

### 12.3 硬件 seq sweep（CSA indexer, msprof）

| seq_len | indexer_hw (µs) | scaling_vs_1k |
|---------|-----------------|---------------|
| 1024 | 16.2 | 1.00 |
| 4096 | 24.1 | 1.49 |
| 32768 | 33.9 | **2.09** |

硬件层 **随 seq_len 缩放**（max 2.1×）；Python Event 层仍平坦 → **P1.5 结论被 P1.7 推翻**。

### 12.4 microbench 可对外陈述边界（最终版）

**可以说**：

- isolated eager 单次 op：**Python ~130–265 µs**；**NPU kernel ~20–40 µs**；差额 = launch overhead
- long-context 下 Python 层平坦 **不意味着** kernel 不随 seq 缩放
- 优化 ROI 在 **减少 Python launch**（NPUGraph），不在 kernel 内部算法（当前 hw 已较快）

**不能说**：

- 「indexer kernel 需要 250 µs」（那是 Python end-to-end）
- 32k 生产 decode latency（需 P2 in-server msprof @ 同 seq_len）
- msprof microbench × 43 层 ≈ **~2 ms** 可直接对标 token#200 Attention ~1.3 ms（seq_len / 图捕获 / 融合路径不同，仅作数量级参考）

---

*报告生成：ktransformers-AK attn microbench 工作流；P1.7 数据见 `tools/attn_microbench/results/msprof_*.json` + `msprof_vs_python_comparison.md`。*
