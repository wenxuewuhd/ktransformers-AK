# NPU MoE Microbench Summary

> 仅 NPU 段，不含 gating / dispatch / combine / CPU MoE。  
> **分析结论与复现**：[`ANALYSIS_AND_REPRODUCTION.md`](./ANALYSIS_AND_REPRODUCTION.md) · D5 launch 拆解：[`msprof_vs_python_comparison.md`](./msprof_vs_python_comparison.md)

| segment | dev_mean_us | p99 | host_mean_us | dispatch_overhead_us | lb_peak | util_peak | lb_eff | util_eff |
|---------|-------------|-----|--------------|----------------------|---------|-----------|--------|----------|
| act_quant_post | 166.41 | 206.30 | 300.24 | 133.83 | (n/a) | (n/a) | (n/a) | (n/a) |
| act_quant_pre | 186.41 | 204.28 | 313.08 | 126.66 | (n/a) | (n/a) | (n/a) | (n/a) |
| gemm_up | 4274.02 | 4587.58 | 4438.25 | 164.24 | 62.91 | 67.93 | 83.89 | 50.95 |
| silu_mul_unfused | 349.99 | 406.55 | 502.34 | 152.35 | (n/a) | (n/a) | (n/a) | (n/a) |
| gemm_down | 4184.89 | 4374.79 | 4350.63 | 165.74 | 31.46 | 133.03 | 41.94 | 99.78 |
| routed_full | 9142.27 | 9266.39 | 9311.31 | 169.04 | 94.37 | 96.87 | 125.83 | 72.66 |
| shared_expert | 978.65 | 1054.73 | 1131.99 | 153.34 | 15.73 | 62.22 | 20.97 | 46.67 |

- routed_full_compute_only_us: **9104.782190322876**
- routed_full_post_dispatch_us: **9142.269401550293**

### Grouped vs Loop (S2)

| target | grouped_us | loop_us | grouped_dispatch | loop_dispatch |
|--------|------------|---------|------------------|---------------|
| up | 4508.39 | 2070.56 | 185.35 | 143.42 |
| down | 4459.61 | 2038.41 | 155.90 | 148.33 |

### n_active sweep (A5)

| n_active | tpe | N | gemm_up_us | gemm_down_us | routed_full_us | util_eff |
|---------:|----:|--:|-----------:|-------------:|---------------:|---------:|
| 1 | 1 | 1 | 4506.95 | 4113.66 | 8472.43 | 404.00 |
| 2 | 1 | 2 | 4229.11 | 4243.94 | 9651.29 | 230.10 |
| 3 | 1 | 3 | 4594.62 | 4347.69 | 9299.77 | 147.82 |
| 4 | 1 | 4 | 4625.39 | 4237.99 | 8507.60 | 101.42 |
| 5 | 1 | 5 | 4413.86 | 4150.83 | 8663.74 | 82.62 |
| 6 | 1 | 6 | 4392.78 | 4450.59 | 8981.26 | 71.38 |

> act_quant(pre) 不随 n_active 变化（固定 [num_tokens, H]）。

> ⚠ **shared_expert ‖ routed_full 并行** (A8)：端到端 ≈ max(shared, routed)，不能相加。
> ⚠ **util_vs_achievable > 0.75** 算打满 (N4)；表中 util_eff 列即 util_vs_achievable。
> ⚠ **act_quant pre vs post** (N5)：post 是上限；pre ≈ post / n_active。

### HBM 带宽实测 (N4)
- hbm_peak_tb_s = 1.6
- hbm_effective_tb_s = 1.2

### FP4 vs W8 roofline (N8)
| path | routed weight | @peak 1.6TB/s | @eff 1.0TB/s |
|------|---------------|--------------|--------------|
| W8 (本计划, n_active=6) | 144 MB | 90 μs | 144 μs |
| FP4 (论文估计) | ~76 MB | ~48 μs | ~76 μs |
| gap | ~1.9× HBM traffic | | |
