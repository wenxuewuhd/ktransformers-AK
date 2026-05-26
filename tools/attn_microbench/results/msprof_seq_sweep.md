# msprof CSA seq sweep (hardware only)

> **完整复现 / 数据索引 / 解析说明**：见 [`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md)  
> **数据来源**：`results/csa_msprof_seq_{1024,4096,8192,16384,32768}.json`  
> **Trace 目录**：`npu_results/seq_{S}/csa_{indexer,attn}_seq{S}/`  
> **生成命令**：`SEQ_LEN_SWEEP="1024 4096 8192 16384 32768" bash run_msprof.sh`

| seq_len | indexer_hw_us (mean±std) | attn_hw_us (mean±std) | scaling_vs_1k |
|---------|--------------------------|------------------------|---------------|
| 1024 | 16.2 ± 0.8 | 30.5 ± 1.3 | 1.00 |
| 4096 | 24.1 ± 0.7 | 38.3 ± 1.1 | 1.49 |
| 8192 | 20.7 ± 0.4 | 36.2 ± 2.1 | 1.28 |
| 16384 | 27.9 ± 1.0 | 38.5 ± 0.8 | 1.72 |
| 32768 | 33.9 ± 0.7 | 37.5 ± 0.8 | 2.09 |

判定:
- 硬件层随 seq_len 单调 — P1.5 平坦结论被推翻（Python overhead 平坦）
