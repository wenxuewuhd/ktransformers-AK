# Decode slowdown on the dynamic hot-32 path — root-cause data (2026-06-24)

Context: GPQA accuracy run reported decode ~12 tok/s on the depool **dynamic hot-32**
path (KT_PREFILL_STREAM_THRESHOLD=1 forces streaming on short prompts so the hot-32
resident is built), vs ~18 measured casually. Investigated whether GGUF-dedup caused it.

## dedup is NOT the cause (controlled A/B, warm + median)
| config | path | decode median tok/s |
|---|---|---|
| non-dedup (pinned pool) | static prefix-32 (short prompt, no stream) | 18.1 |
| dedup (GGUF reuse)      | static prefix-32                          | 17.9 |
| dedup                   | dynamic hot-32 (threshold=1)              | 12.0 |
| non-dedup               | dynamic hot-32 (threshold=1)              | 10.0 |
=> dedup == non-dedup on BOTH paths. The earlier "18" everyone recalled was the
   STATIC prefix-32 path (short prompts never triggered streaming at default thresh 512).

## The real difference: static vs dynamic path. And it is NOT hit-rate.
Decode resident hit-rate probe (KT_HITRATE_PROBE=1, eager decode):
- static prefix-32: ~14%  (random-level, 32/256=12.5%)
- dynamic hot-32  : ~43%  (per-40-step windows 20.7 -> 43.0 -> 42.7%)

=> dynamic hot-32 has **3x better** decode hit-rate (43% vs 14%) yet decodes **slower**
   (12 vs 18). So the slowdown is **OVERHEAD on the dynamic path, not hit-rate**.

Prime suspect: the inline `_apply_resident_layer_depool` write to the registered
`layer.w13_weight/.w2_weight` during the streaming prefill triggers a weight-region
flush that stalls the subsequent decode's NSA `.item()` syncs (see memory
npu-weight-region-write-stalls-nsa). Consistent with the per-batch decode sequence
having a 1.1-1.5 tok/s first batch then oscillating 9-20.

## Next: confirm overhead source + fix (not hit-rate; improving hit-rate won't help).

## Isolation (EXP2): streaming vs dynamic-resident
| config | decode median | per-batch shape |
|---|---|---|
| static (no stream), prefix-32        | 18.0 | tight |
| stream ON + DYNAMIC OFF, prefix-32   | 17.6 | tight (1.3 first batch, then 17.6-17.8) |
| stream ON + DYNAMIC ON, hot-32       | 12.0 | OSCILLATES 9-20 (1.5 first, then 9.3..20.6) |

=> Streaming prefill aftermath is harmless (17.6 ~= 18). The **dynamic-resident mechanism
   is the whole decode cost** (17.6 -> 12), even though hot-32's hit-rate is 3x better.
   It is NOT hit-rate and NOT streaming. The dynamic batches OSCILLATE (some 20 > 17.6,
   some 9-12) -> the hot-32 resident HELPS some batches but HURTS others, net -32%.

Open: WHY does putting the HOT experts on NPU decode slower than random prefix-32?
Counterintuitive (more NPU residency, less CPU-MoE, should be faster). Needs a decode
profile (NPU resident GEMM vs CPU-MoE vs dispatch) per step for hot-32 vs prefix-32.
