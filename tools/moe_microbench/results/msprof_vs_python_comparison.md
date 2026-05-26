# msprof vs Python Event Comparison (MoE)

> Python Event = eager end-to-end per op call (D4)  
> msprof device time = NPU hardware kernel time (Level1, D5)  
> 两者差额 = launch overhead  
> **完整结论与复现**：[`ANALYSIS_AND_REPRODUCTION.md`](./ANALYSIS_AND_REPRODUCTION.md)

## Launch Overhead 拆解

| segment | python_event_us | msprof_device_us | launch_overhead_us | overhead_pct |
|---------|-----------------|------------------|--------------------|--------------|
| act_quant_post | 166.41 | 3.29 | 163.13 | 98.03% |
| act_quant_pre | 186.41 | 2.18 | 184.24 | 98.83% |
| gemm_up | 4274.02 | 107.42 | 4166.60 | 97.49% |
| silu_mul | 349.99 | 6.71 | 343.27 | 98.08% |
| gemm_down | 4184.89 | 84.20 | 4100.69 | 97.99% |
| routed_full | 9142.27 | 202.01 | 8940.26 | 97.79% |
| shared_expert | 978.65 | 46.51 | 932.13 | 95.25% |

## Roofline 对照 (硬件层, measured @ 1.20 TB/s)

| segment | device_us (hw) | lb @ measured | util_vs_measured |
|---------|----------------|---------------|------------------|
| gemm_up | 107.42 | 83.89 | 78.09% |
| gemm_down | 84.20 | 41.94 | 49.81% |
| shared_expert | 46.51 | 20.97 | 45.09% |

## Grouped vs Loop 硬件层确认 (D4 谜团)

| path | python_event_us | hw_device_us | overhead_us | overhead_pct |
|------|-----------------|--------------|-------------|--------------|
| grouped | 4508.39 | 104.04 | 4404.35 | 97.69% |
| loop | 2070.56 | 102.21 | 1968.35 | 95.06% |

判定: hw_grouped ≈ hw_loop：差异主要来自 grouped fallback 的 Python 路径开销

## 顶部速览 (overhead_pct)
- act_quant_post: 98.0%
- act_quant_pre: 98.8%
- gemm_up: 97.5%
- silu_mul: 98.1%
- gemm_down: 98.0%
- routed_full: 97.8%
- shared_expert: 95.2%

gemm_down util_vs_measured = 49.8% (D4 Event util_eff=99.78%)；判定: D4 Event 99.78% 含大量 Python overhead，硬件 util 远低于 Event 报数
