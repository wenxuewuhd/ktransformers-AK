# Synthetic Attention Microbench Summary

> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.
> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.
> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).

| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |
|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|
| csa | 8192 | 1 | 1000 | 288.8 ± 11.7 | 168.4 ± 4.4 | 30.9 | 457.2 ± 12.5 | 277.4792816489935 |
| hca | 8192 | 1 | 1000 | - | 156.4 ± 9.6 | 18.9 | - | 273.72924610972404 |
| swa | 8192 | 1 | 1000 | - | 137.5 ± 10.5 | - | - | 235.196472145617 |
