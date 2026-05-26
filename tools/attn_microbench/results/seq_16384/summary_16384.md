# Synthetic Attention Microbench Summary

> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.
> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.
> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).

| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |
|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|
| csa | 16384 | 1 | 1000 | 299.8 ± 10.9 | 189.3 ± 5.7 | 45.3 | 489.1 ± 12.3 | 309.227061457932 |
| hca | 16384 | 1 | 1000 | - | 146.5 ± 9.1 | 2.5 | - | 246.58016953617334 |
| swa | 16384 | 1 | 1000 | - | 144.0 ± 6.6 | - | - | 254.9059521406889 |
