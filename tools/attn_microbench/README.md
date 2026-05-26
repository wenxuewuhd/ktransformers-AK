# Synthetic NPU Attention Microbench（独立小工程）

> **已知局限（读数前必看）**
> 1. 主 KV = **bf16**（NPU 当前仅支持 bf16 attention；论文 bf16+fp8 mixed 路径**未测**）
> 2. Synthetic 数据不区分 RoPE / 非 RoPE 维
> 3. 默认 **decode-only**（`q_len=1`），batch=1
> 4. 正式计时默认 **`repeat=1000`**；报告看 **mean ± std**，不单看 mean
> 5. **`isolated_device_sum_us`** = indexer + attn **分开计时之和**，不是 fused 端到端
> 6. `csa`/`hca` 的 `attn_us` **已含 SWA branch**，不能与 `swa` 简单相加
> 7. ⚠️ **seq_len scaling 待诊断**：跑 `bash run_diag.sh` 看 `results/diag_seq_scaling.json`；在诊断通过前勿用 §4 绝对值做 long-context 结论
> 8. B4 `cmp_kv` page 第二维语义（128 vs 32）未完全钉死；小 case：`python -m attn_bench.reference_check --seq-len 512`

在 **不修改仓库主干** 的前提下，用 Synthetic KV / page table 直调生产通路算子：

- **SWA**：`npu_sparse_attn_sharedkv`（c1a）
- **CSA**：`npu_quant_lightning_indexer` + `npu_sparse_attn_sharedkv`（c4a）
- **HCA**：`npu_sparse_attn_sharedkv`（c128a）

默认 **seq_len=32768** decode 单 token。

## NPU 卡选择

```bash
npu-smi info                    # 看 HBM-Usage 与 Process，避开已占用卡（如 NPU 2）
export ASCEND_RT_VISIBLE_DEVICES=0   # 物理卡号；进程内为 npu:0
cd tools/attn_microbench
source env.sh
```

详见 [`IMPLEMENTATION_PLAN.md` §0](./IMPLEMENTATION_PLAN.md#0-npu-环境与卡选择)。

## 快速开始

```bash
# 形状校验（CPU）
DRY_RUN=1 SEQ_LEN=32768 bash run_all.sh

# NPU sanity + 正式计时
export ASCEND_RT_VISIBLE_DEVICES=0
source env.sh
SEQ_LEN=32768 bash run_all.sh
# → results/summary_32768.md

# 单独跑
python -m attn_bench.bench_swa --seq-len 32768 --sanity --repeat 50
python -m attn_bench.bench_csa --seq-len 32768 --skip-indexer --sanity
```

## seq_len / batch 扫描

```bash
for S in 512 8192 32768; do
  SEQ_LEN=$S OUT_DIR=results/seq_$S bash run_all.sh
done
for B in 1 8 32; do
  BATCH_SIZE=$B SEQ_LEN=32768 OUT_DIR=results/batch_$B bash run_all.sh
done
```

## 依赖

- `torch_npu` + CANN **`custom_ops`**（`torch.ops.custom.npu_sparse_attn_sharedkv*` / `npu_quant_lightning_indexer*`）
- 只读 `../../third_party/sglang/python`（不改主干）

## P0 诊断（Claude review 后必跑）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
source env.sh
bash run_diag.sh          # → results/diag_seq_scaling.json
python -m attn_bench.reference_check --seq-len 512
```

## CSA fallback

```bash
SKIP_INDEXER=1 bash run_all.sh
# 或
python -m attn_bench.bench_csa --seq-len 32768 --skip-indexer
```

详细模块 API 见 [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)。
