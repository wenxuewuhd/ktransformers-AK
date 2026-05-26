# Synthetic Attention Microbench Summary

> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.
> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.
> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).

| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |
|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|
| csa | 1024 | 1 | 1000 | 273.7 ± 62.0 | 169.8 ± 3.9 | 29.7 | 443.6 ± 62.2 | 313.4067915380001 |
| hca | 1024 | 1 | 1000 | - | 152.1 ± 11.3 | 11.9 | - | 268.50854977965355 |
| swa | 1024 | 1 | 1000 | - | 140.2 ± 10.5 | - | - | 246.08726613223553 |
