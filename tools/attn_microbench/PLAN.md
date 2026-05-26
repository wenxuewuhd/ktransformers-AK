# 方案 A：32k Decode Attention Microbench 详细计划

> **目标**：独立小工程，直调当前生产通路 `torch.ops.custom.*`，分别 profiling **SWA / CSA / HCA** 三类 attention 层在 **seq_len=32768** decode 步的耗时。  
> **不跑全模型 forward**，但 KV / page table 需与真实 prefill 一致或可追溯。  
> **代码锚点**：`third_party/sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`

---

## 1. 三类 Layer 定义（与生产对齐）

DeepSeek-V4-Flash 43 层 decode 共用 `npu_sparse_attn_sharedkv`，按 `compress_ratios[layer_id]` 分三种 metadata 路径：

| 代号 | 含义 | compress_ratio | 代表 layer_id | metadata key | 额外算子 |
|------|------|----------------|---------------|--------------|----------|
| **SWA** | Sliding Window Attention only | **1** | **0, 1** | `c1a_metadata` | 无 cmp 分支 |
| **CSA** | Compressed Sparse Attention (c4 + LI topk) | **4** | **2, 4, 6, …**（21 层） | `c4a_metadata` | **`npu_quant_lightning_indexer`** → attn |
| **HCA** | Highly Compressed Attention (c128 全量 cmp) | **128** | **3, 5, 7, …**（20 层） | `c128a_metadata` | 无 indexer（`cmp_sparse_indices=None`） |

`config.json` 中 `compress_ratios` 模式（前 10 层）：`[1, 1, 4, 128, 4, 128, …]`，L42=`4`。

**共同点（三类都有）**：
- SWA 本地窗：`ori_kv` + `ori_block_table`，`ori_mask_mode=4`，`ori_win_left=127`
- decode 单 token：`q` shape `[1, 64, 512]`（TND），`cu_seqlens_q=[0,1]`，`seqused_kv=[32768]`

**差异点**：

| 字段 | SWA | CSA | HCA |
|------|-----|-----|-----|
| `has_cmp_kv` | False | True | True |
| `cmp_ratio` | — | 4 | 128 |
| `cmp_kv` | — | paged compress buffer | paged compress buffer |
| `cmp_block_table` | — | `[1, ⌈32768/4/128⌉]` = `[1,64]` pages | `[1, ⌈32768/128/128⌉]` = `[1,2]` pages |
| `cmp_sparse_indices` | — | `[1,1,512]` from indexer | **None** |
| 逻辑 cmp token 数 | — | ~8192 | ~256 |

---

## 2. 工程目录结构（建议）

```text
tools/attn_microbench/
├── PLAN.md                          # 本文
├── README.md                        # 用法与 env
├── run_all.sh                       # 一键：dump + bench + report
├── env.sh                           # CANN / DSv4 环境变量（与 p27 launch 对齐）
├── config/
│   └── dsv4_flash_32k.yaml          # seq_len, page_size, layer_ids, head 数
├── dump/
│   ├── hook_dump_kv.py              # 可选：server prefill 后 dump 一层 KV
│   └── snapshot_format.md           # .pt 字段说明
├── build/
│   ├── metadata_builder.py          # 从 ascend_backend.compute_kernel_metadata 抽取
│   ├── tensor_builder.py            # 由 snapshot 或 synthetic 构造 PA buffers
│   └── page_table_utils.py          # swa/c4/c128 page table 与 seq_len 关系
├── bench/
│   ├── bench_swa.py                 # 只调 npu_sparse_attn_sharedkv (c1a)
│   ├── bench_csa.py                 # indexer + attn (c4a)
│   ├── bench_hca.py                 # 只调 attn (c128a)
│   └── common.py                    # warmup, Event 计时, profiler 封装
├── profile/
│   ├── run_level0_profiler.sh       # torch_npu.profiler Text 导出
│   └── parse_operator_csv.py      # 从 operator_details 抽 Device Self
└── report/
    └── summarize.py                 # 合并 SWA/CSA/HCA 表格输出
```

**依赖边界**：
- 必须：`torch_npu`、`sgl_kernel_npu`（注册 `torch.ops.custom.*`）
- 可选引用：`sglang.srt.hardware_backend.npu.attention.ascend_backend` 中常量/逻辑（不启动 server）
- 不需要：KT MoE、全量 W8A8 权重、scheduler

---

## 3. 实施阶段

### Phase 0：环境与准入（0.5 天）

**任务**：
1. `env.sh` 对齐 `p27_launch_ds4flash_npu_num_expert_0.sh`：
   ```bash
   export ASCEND_USE_FIA=1
   export USE_PA_DECODE=1
   export USE_PA_PREFILL=1
   export IS_DEEPSEEK_V4=1
   export LI_KV_DTYPE_INT8=1
   export USE_FUSED_COMPRESSOR=1
   ```
2. 验证算子可 import：
   ```python
   import sgl_kernel_npu  # noqa
   assert hasattr(torch.ops.custom, "npu_sparse_attn_sharedkv")
   assert hasattr(torch.ops.custom, "npu_quant_lightning_indexer")
   ```
3. 单卡 NPU 可见，`page_size=128`。

**准入标准**：三类 metadata op 各调用一次不 crash（可用 synthetic 小 seq_len=512 冒烟）。

---

### Phase 1：32k KV 状态准备（1–2 天）

Attention 计时的前提是 **decode 步的 KV / page table 与 seq_len=32768 一致**。两种路径（二选一或并存）：

#### 路径 1：Snapshot（推荐，生产 faithful）

1. 用现有 `p27_long_context_decode_test.sh` 或 32k prefill workload 跑到 **profile decode token**：
   ```bash
   TARGET_SEQ_LEN=32768 PREFILL_TOKENS=32760 PROFILE_DECODE_TOKEN=8 ...
   ```
2. 在 `AscendAttnBackend.forward_sparse` 或 `token_to_kv_pool.get_*_buffer` 加 **一次性 dump hook**（环境变量 `ATTN_MICROBENCH_DUMP=1`）：
   - 对代表层 **L0(SWA), L2(CSA), L3(HCA)** 保存：
     - `ori_kv`, `cmp_kv`（若有）
     - `swa_page_table`, `c4_page_table`, `c128_page_table`
     - `seqused_kv`, `cu_seqlens_q`
     - `sinks`（fp32 `[64]`）
     - CSA 额外：`indexer` 的 `q,k,k_scale,weights` 或 dump 后的 `topk_idxs`
3. 存为 `fixtures/seq32768_layer{L}.pt`。

#### 路径 2：Synthetic（快速打通，精度次之）

1. 按 `HybridSWAC4C128PoolConfigurator` 的 **paged shape** 分配 empty/random tensor：
   - `ori_kv`: `[num_swa_blocks, 128, 1, 512]` bf16
   - `cmp_kv` c4: `[num_c4_blocks, 128, 1, 512]`
   - `cmp_kv` c128: `[num_c128_blocks, 128, 1, 512]`
2. page table 填 **单调递增物理页号**（通过 `init_forward_metadata` decode 分支逻辑复现）。
3. CSA 的 `topk_idxs` 可先 random valid indices，或通过 **小 seq prefill 一次** 生成后 freeze。

**准入标准**：三类层 `npu_sparse_attn_sharedkv` forward 数值不 NaN，shape 与 CANN 一致。

---

### Phase 2：Metadata 与参数构造（1 天）

从 `ascend_backend.compute_kernel_metadata()` **原样抽取**（避免手写参数漂移）：

```python
# metadata_builder.py 伪代码
fa_common = {
    "cu_seqlens_q": tensor([0, 1], int32, npu),
    "seqused_kv": tensor([32768], int32, npu),
    "cmp_ratio": 1,  # 会被 c4/c128 覆盖
    "ori_mask_mode": 4,
    "cmp_mask_mode": 3,
    "ori_win_left": 127,
    "ori_win_right": 0,
    "layout_q": "TND",
    "layout_kv": "PA_ND",
}
c1a = npu_sparse_attn_sharedkv_metadata(
    batch_size=1, num_heads_q=64, num_heads_kv=1, head_dim=512,
    has_ori_kv=True, has_cmp_kv=False, **fa_common,
)
c4a = ... has_cmp_kv=True, cmp_ratio=4, cmp_topk=512 ...
c128a = ... has_cmp_kv=True, cmp_ratio=128 ...
li_quant_metadata = npu_quant_lightning_indexer_metadata(
    actual_seq_lengths_query=..., actual_seq_lengths_key=...,
    sparse_count=512, cmp_ratio=4, num_heads_q=64, num_heads_k=1, head_dim=128, ...
)
```

**decode attn 公共 kwargs**（`forward_sparse` decode + `USE_PA_DECODE` 分支）：

```python
attn_kwargs = {
    "q": q,  # [1,64,512]
    "ori_kv": ori_kv,
    "ori_block_table": swa_page_table,
    "sinks": sinks,
    "metadata": metadata,  # c1a / c4a / c128a
    "softmax_scale": 512**-0.5,
    "cu_seqlens_q": [0,1],
    "seqused_kv": [32768],
    "ori_mask_mode": 4,
    "ori_win_left": 127,
    "ori_win_right": 0,
    "layout_q": "TND",
    "layout_kv": "PA_ND",
}
# CSA 追加:
# cmp_ratio=4, cmp_kv, cmp_block_table, cmp_sparse_indices=topk.view(-1,1,512), cmp_mask_mode=3
# HCA 追加:
# cmp_ratio=128, cmp_kv, cmp_block_table, cmp_sparse_indices=None, cmp_mask_mode=3
```

**代表层选取**（microbench 默认）：

| 类型 | layer_id | 说明 |
|------|----------|------|
| SWA | 0 | 无 compressor |
| CSA | 2 | 第一个 c4 + indexer |
| HCA | 3 | 第一个 c128 |

---

### Phase 3：三类 Benchmark 实现（1–2 天）

#### 3.1 SWA (`bench_swa.py`)

**调用链**：
```text
npu_sparse_attn_sharedkv(metadata=c1a) × repeat
```

**计时**：
- `torch.npu.Event`：`enable=True` → op → `synchronize` → `elapsed_time`
- warmup 20，measure 100，报 mean / p50 / p99

**Profiling 输出字段**：
- `swa_attn_device_us`（Event）
- 可选 Level0：`aclnnSparseAttnSharedkv` Device Self

---

#### 3.2 CSA (`bench_csa.py`)

**调用链**（与生产一致，**两段计时 + 合计**）：
```text
1) npu_dynamic_quant(q)  # 若 snapshot 已含量化 q 可跳过
2) npu_quant_lightning_indexer → topk_idxs [1,512]
3) npu_sparse_attn_sharedkv(cmp_sparse_indices=topk.view(-1,1,512))
```

**Indexer 输入**（`nsa_indexer.forward_npu_dsv4_fusion`）：
- `query`, `key`, `key_dequant_scale`, `weights`
- `block_table=c4_page_table`
- `actual_seq_lengths_query`, `actual_seq_lengths_key`
- `metadata=li_quant_metadata`

**Profiling 输出字段**：
- `csa_indexer_device_us`
- `csa_attn_device_us`
- `csa_total_device_us`（indexer + attn，中间不插 host 大开销）
- Level0 算子：`aclnnQuantLightningIndexer*` + `aclnnSparseAttnSharedkv`

---

#### 3.3 HCA (`bench_hca.py`)

**调用链**：
```text
npu_sparse_attn_sharedkv(metadata=c128a, cmp_sparse_indices=None) × repeat
```

**Profiling 输出字段**：
- `hca_attn_device_us`
- Level0：`aclnnSparseAttnSharedkv`（HCA 路径 kernel 名可能含 Compressor 相关）

---

### Phase 4：Profiling 方法论（0.5 天）

三层 profiling，由轻到重：

| 层级 | 工具 | 用途 | 三类是否共用 |
|------|------|------|--------------|
| **L1 设备 Event** | `torch.npu.Event` | 纯 NPU op 窗口，**默认验收** | SWA/CSA/HCA 各自 loop |
| **L2 inline profiler** | `torch_npu.profiler` Level0, `analyse_flag=0` | `operator_details` Device Self | 每类单独采 1 step |
| **L3 msprof** | `msprof` 外挂 | kernel_details Duration 分类 | 可选，与 L1 交叉验证 |

**注意**：
- `Duration` / Device Self = **设备 busy，不含 launch head**
- CSA 必须 **分开报 indexer 与 attn**，与生产路径一致
- inline profiler 有 `aten::item` 噪音；microbench 无 scheduler，噪音远小于 server

**seq_len 扫描（可选扩展）**：
```text
seq_lens = [512, 2048, 8192, 16384, 32768]
```
固定 snapshot 或按 seq 各 dump 一套 fixture，画 **SWA/CSA/HCA attn_us vs seq_len** 曲线。

---

### Phase 5：验证与验收（0.5 天）

| 检查项 | 方法 |
|--------|------|
| 形状 | 与 `forward_sparse` decode 分支 assert 一致 |
| 层类型 | metadata key 与 `compress_ratio` 匹配 |
| 32k page table | `c4_page_table` width ≥ 8192/128；`c128` ≥ 256/128 |
| 数值 | 输出 `o` 无 NaN/Inf |
| 与 server 交叉 | 同 seq_len 下 microbench L1 与 server profiler 中 **单层** `SparseAttnSharedkv` Device Self 量级接近（±20%） |
| CSA 完整性 | total ≈ indexer + attn（Event 口径） |

**交付物**：
- `report/summary_32k.json`：`{swa, csa_indexer, csa_attn, csa_total, hca}` 微秒
- `report/summary_32k.md`：表格 + 占 decode 单步 NPU busy 比例（可选）

---

## 4. 32k 关键尺寸速查（bs=1, page_size=128）

| 量 | 值 |
|----|-----|
| `seqused_kv` | 32768 |
| SWA 逻辑 pages | 32768 / 128 = **256** |
| c4 逻辑 cmp tokens | 32768 / 4 = **8192** → **64** pages |
| c128 逻辑 cmp tokens | 32768 / 128 = **256** → **2** pages |
| SWA 物理池（生产） | **128 tokens**（sliding window），非 32k 全量 |
| `q` | `[1, 64, 512]` bf16 |
| `index_topk` | 512 |

---

## 5. 风险与缓解

| 风险 | 缓解 |
|------|------|
| PA_ND layout 不对 | 优先 snapshot；synthetic 对照 `get_swa_buffer` 返回 tensor 的 `.shape` |
| metadata 参数漂移 | 直接复用 `compute_kernel_metadata`，不手写 |
| CSA 缺 indexer 输入 | snapshot 同时 dump indexer 的 k/k_scale/weights 或 dump 最终 topk 做 A/B |
| 32k prefill 耗时长 | 先 `seq=8192` 打通，再升到 32768；或复用 `REUSE_PROMPT` |
| `sgl_kernel_npu` 版本 | `env.sh` 固定 CANN / 包版本，README 记录 |
| 与 wall ITL 混淆 | 报告明确：**本 bench 仅 NPU attention 算子，不含 QKV 投影 / MoE / HC** |

---

## 6. 排期建议

| 阶段 | 工时 | 产出 |
|------|------|------|
| Phase 0 环境 | 0.5d | `env.sh` + smoke |
| Phase 1 KV | 1–2d | `fixtures/seq32768_layer*.pt` |
| Phase 2 Metadata | 1d | `metadata_builder.py` |
| Phase 3 Bench ×3 | 1–2d | `bench_swa/csa/hca.py` |
| Phase 4 Profiling | 0.5d | Event + Level0 脚本 |
| Phase 5 验证 | 0.5d | report + server 交叉 |
| **合计** | **4–6d** | 可复用 microbench 工程 |

---

## 7. 最小首跑命令（目标态）

```bash
# 1) 可选：从 server dump 32k KV
ATTN_MICROBENCH_DUMP=1 TARGET_SEQ_LEN=32768 ... bash tools/p27_long_context_decode_test.sh

# 2) microbench
source tools/attn_microbench/env.sh
python tools/attn_microbench/bench/bench_swa.py --seq-len 32768 --fixture fixtures/seq32768_layer0.pt
python tools/attn_microbench/bench/bench_csa.py --seq-len 32768 --fixture fixtures/seq32768_layer2.pt
python tools/attn_microbench/bench/bench_hca.py --seq-len 32768 --fixture fixtures/seq32768_layer3.pt

# 3) 汇总
python tools/attn_microbench/report/summarize.py --out report/summary_32k.md
```

---

## 8. 与现有通路关系

```text
生产 decode（43 层）
├── SWA 层 ×3   → bench_swa
├── CSA 层 ×21  → bench_csa（indexer + attn）
└── HCA 层 ×20  → bench_hca

单步 decode NPU attention 粗算：
  3×T_swa + 21×(T_li + T_c4a) + 20×T_c128a
```

本 microbench **不测** QKV 线性、RMSNorm、RoPE、HC、MoE；若需「单层 full attention block」需另开 Phase 6（扩展 scope）。

---

*计划版本：2026-05-25，对应 SGLang ascend_backend + DSv4-Flash W8A8 num_expert_0 通路。*
