# msprof vs Python Event Comparison

> **完整复现 / 数据索引 / 解析说明**：见 [`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md)  
> **数据来源**：`swa_msprof.json` / `csa_msprof.json` / `hca_msprof.json` + `seq_32768/*.json`  
> **生成命令**：`bash run_msprof.sh` 或 §7.3 in guide

> Python Event = eager end-to-end per op call  
> msprof device time = NPU hardware kernel time（`kernel_details.csv`，非旧版 op_summary）

## Launch Overhead 拆解

| op | python_event_us (mean±std) | msprof_device_us (mean±std) | launch_overhead_us | overhead_pct |
|----|----------------------------|------------------------------|--------------------|--------------|
| swa_attn | 133.8 ± 6.7 | 21.6 ± 1.2 | 112.3 | 83.9% |
| csa_indexer | 265.0 ± 13.5 | 36.1 ± 1.6 | 228.9 | 86.4% |
| csa_attn | 157.2 ± 10.1 | 38.8 ± 1.3 | 118.3 | 75.3% |
| hca_attn | 136.1 ± 5.7 | 26.5 ± 1.2 | 109.6 | 80.5% |

判定:
- overhead_pct > 50% → Python launch 主导, 优化方向 = NPUGraph / op fusion
- overhead_pct < 30% → NPU kernel 主导, 优化方向 = kernel 内部 / 算法
- 中间 → 两边都要看
- **本次判定: Python launch 主导**

## Roofline 对照 (hw 层, effective 1.0 TB/s)

| op | device_us (msprof) | hw_lower_bound_us | util_vs_achievable |
|----|--------------------|-------------------|---------------------|
| swa_attn | 21.6 | 0.13 | 0.006 |
| csa_indexer | 36.1 | 1.05 | 0.029 |
| csa_attn | 38.8 | 8.39 | 0.216 |
| hca_attn | 26.5 | 0.26 | 0.010 |

> util_vs_achievable < 0.1 → kernel 远未打满带宽
> util_vs_achievable > 0.5 → kernel 已接近带宽极限

## 顶部速览 (overhead_pct)
- swa_attn: 83.9%
- csa_indexer: 86.4%
- csa_attn: 75.3%
- hca_attn: 80.5%
