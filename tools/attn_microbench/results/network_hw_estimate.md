# Network Attention Hardware Estimate (auto-generated)

> 由 `scripts/summarize_network_hw.py` 从 `results/*_msprof*.json` 生成。  
> **完整复现 / 解析说明**：见 [`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md)  
> **重新生成**：`python scripts/summarize_network_hw.py --out results/network_hw_estimate.md`  
> 层数：SWA×2 + CSA×20 + HCA×20 = 42（不含 L43）。  
> **@32k CSA 用 seq sweep 的 `csa_msprof_seq_32768.json`**（与 `csa_msprof.json` 同口径、独立两次 run，数值差 ~5% 正常）。

## 单层硬件 @ 32k（msprof kernel_details）

| 层类型 | Indexer (µs) | Attn (µs) | 层合计 (µs) |
|--------|--------------|-----------|-------------|
| SWA | — | 21.6 ± 1.2 | 21.6 |
| CSA | 33.9 | 37.5 | 71.4 |
| HCA | — | 26.5 ± 1.2 | 26.5 |

## 整网 @ 32k

```
Total = 2×SWA + 20×CSA + 20×HCA
      = 2×21.6 + 20×71.4 + 20×26.5
      ≈ 2.00 ms
```

| 组成部分 | 耗时 | 占比 |
|----------|------|------|
| SWA×2 | 43 µs | 2.2% |
| CSA indexer×20 | 678 µs | 33.9% |
| CSA attn×20 | 751 µs | 37.5% |
| HCA attn×20 | 530 µs | 26.5% |
| **合计** | **2002 µs** | 100% |

## Seq sweep（CSA 可变，SWA/HCA 用 32k 常数）

| seq_len | indexer_hw | attn_hw | CSA×20 | 整网 hw | vs 1024 |
|---------|------------|---------|--------|---------|---------|
| 1024 | 16.2±0.8 | 30.5±1.3 | 933 µs | 1.51 ms | 1.00× |
| 4096 | 24.1±0.7 | 38.3±1.1 | 1250 µs | 1.82 ms | 1.21× |
| 8192 | 20.7±0.4 | 36.2±2.1 | 1139 µs | 1.71 ms | 1.14× |
| 16384 | 27.9±1.0 | 38.5±0.8 | 1327 µs | 1.90 ms | 1.26× |
| 32768 | 33.9±0.7 | 37.5±0.8 | 1429 µs | 2.00 ms | 1.33× |

## Python Event @ 32k（对比）

- 硬件：2.00 ms
- Python Event（repeat=1000）：11.43 ms
- Launch overhead：82.5%
