# Synthetic Attention Microbench Summary

> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.
> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.
> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).

| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |
|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|
| csa | 32768 | 1 | 10 | 277.5 ± 16.2 | 158.8 ± 3.6 | 20.2 | 436.3 ± 16.6 | 270.40401473641396 |
| csa | 32768 | 1 | 10 | - | 176.1 ± 11.7 | 37.5 | - | 363.4059801697731 |
| hca | 32768 | 1 | 10 | - | 158.8 ± 17.1 | 20.2 | - | 343.51255744695663 |
| swa | 32768 | 1 | 10 | - | 138.6 ± 10.4 | - | - | 316.4181485772133 |
