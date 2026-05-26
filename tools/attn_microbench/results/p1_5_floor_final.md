# P1.5 结论：indexer seq sweep 仍平坦（repeat=1000, warmup=30）

## 数据（`results/summary_seq_sweep_r1000.md`）

| seq_len | c4_num_pages | indexer mean±std (µs) | vs 1024 ratio |
|---------|--------------|-------------------------|---------------|
| 1024 | 2 | 273.7 ± 62.0 | 1.00 |
| 4096 | 8 | 261.3 ± 10.0 | 0.95 |
| 8192 | 16 | 288.8 ± 11.7 | 1.06 |
| 16384 | 32 | 299.8 ± 10.9 | 1.10 |
| 32768 | 64 | 265.0 ± 13.5 | 0.97 |

最大比值 **1.10 ≪ 1.5** → **仍平坦**。

## 已做修复

- **P1.4 部分**：`c4_page_table` / `c128_page_table` 对齐 `ascend_backend.py` decode 路径的 **strided page id**（`[B, c4_num_pages]` 而非 `[B, c4_cols]`）。见 `page_table.py::_strided_page_table`。
- microbench dump：`results/microbench_indexer_dump.json` 显示 `block_table.shape=[1,64]` @ seq=32768。

## 结论

在 strided page table 修复后，**indexer 仍不随 seq_len / c4 候选规模单调缩放** → **假说 (iii) kernel floor / isolated 上界** 为当前最可信解释。

**对外**：本 microbench **不适合** long-context indexer 性能分析；只能报告 **isolated 上界 ~270–300 µs**（repeat=1000, NPU decode, batch=1）。

---

> **已被 P1.7 推翻**（2026-05-26）：msprof 硬件路径显示 indexer @32k 仅 **~34 µs**，1024→32768 硬件缩放 **2.1×**；平坦来自 Python launch overhead。见 [`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md)。
