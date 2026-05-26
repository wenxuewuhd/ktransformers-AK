# P1.4 indexer kwargs diff — **未完成（缺生产 dump）**

> 生产侧 `results/production_indexer_dump.json` **未生成**（见 `p1_3_blocked.md`）。以下为 microbench 侧已钉死字段 + 代码审查修复。

## 已修复（代码审查 `ascend_backend.py` + `nsa_indexer.py`）

| 字段 | 修复前 microbench | 修复后 microbench | 生产预期（静态） |
|------|-------------------|-------------------|------------------|
| `block_table` | `[1, 8192]` int32 | `[1, 64]` strided page id | `c4_page_table[:,::128]//128` → `[B, c4_num_pages]` |
| `key_dequant_scale` | squeeze(-2) ✅ | 同左 | `k_scale.squeeze(-2)` ✅ |
| 标量 kwargs | cmp_ratio=4, sparse_mode=3, … | 同 `ops_runner.py` | 同 `nsa_indexer.py` li_input_kwargs ✅ |

## Microbench dump（`microbench_indexer_dump.json`, seq=32768）

- `block_table`: shape `[1, 64]`, dtype int32
- `key`: `[64, 128, 1, 128]` int8
- `key_dequant_scale`: `[64, 128, 1]` fp16
- `actual_seq_lengths_key`: `[1]` = 32768
- `sparse_count`: 512

## 待生产 dump 确认

- `block_table` 运行时 sample 值域
- `actual_seq_lengths_key` 是否恒为 token_len（非 c4_len）
- `metadata` tensor 内部 layout

**正式 diff 命令**（P1.3 完成后）：

```bash
python scripts/diff_indexer_kwargs.py
grep "❌" results/p1_field_diff.md  # 目标 0 行
```
