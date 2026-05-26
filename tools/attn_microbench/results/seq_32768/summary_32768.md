# Synthetic Attention Microbench Summary

> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.
> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.
> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).

| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |
|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|
| csa | 32768 | 1 | 1000 | 265.0 ± 13.5 | 157.2 ± 10.1 | 23.4 | 422.2 ± 16.9 | 248.53789899498224 |
| hca | 32768 | 1 | 1000 | - | 136.1 ± 5.7 | 2.3 | - | 231.60381522029638 |
| swa | 32768 | 1 | 1000 | - | 133.8 ± 6.7 | - | - | 231.86018131673336 |
