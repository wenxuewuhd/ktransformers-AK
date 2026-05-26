# Synthetic Attention Microbench Summary

> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.
> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.
> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).

| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |
|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|
| csa | 4096 | 1 | 1000 | 261.3 ± 10.0 | 152.3 ± 3.5 | 9.5 | 413.6 ± 10.6 | 246.1914736777544 |
| hca | 4096 | 1 | 1000 | - | 144.6 ± 6.4 | 1.7 | - | 245.36670465022326 |
| swa | 4096 | 1 | 1000 | - | 142.9 ± 9.9 | - | - | 250.48568099737167 |
