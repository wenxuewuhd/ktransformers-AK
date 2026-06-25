# Handover — depool GGUF-dedup + prefill speed + decode fix (2026-06-25)

Branch: parent `gguf-mxfp4-dedup`, submodule `third_party/sglang` @ `kt-sidestream-sharedstream`.
Working dir: `/workspace/code/kt-D-ggufdedup` (main clone, not a worktree).

## What this delivers (current best state, all committed)
DeepSeek-V4-Flash, single 910B3, depool path:
- **prefill ~30s** (1699 tok 35.6s; ~token-count-independent — the 256-expert convert dominates)
- **decode ~12-16 tok/s** (after the mask-remap fix; noisy, long-KV lower)
- **prefill->decode switch ~0** (hot-32 resident folded into the streaming prefill; no separate pass)
- **host DDR 202G used** (= ~147G kt-kernel CPU-MoE anon copy + ~55G overhead; GGUF 147G is in
  buff/cache, not "used"). **Saves ~137G vs non-dedup** (which builds a 137G pinned codes pool).
- arith 15x17=255 correct throughout.

## Launch
Static prefix-32 (resident = directly-loaded int8 experts 0-31, no per-prefill convert):
```
cd /workspace/code/kt-D-ggufdedup && KT_MXFP4_DEPOOL=1 KT_MXFP4_GGUF_DEDUP=1 KT_FORCE_SYNC_SUBMIT=0 \
KT_SIDE_STREAM=1 KT_PREFILL_STREAM=1 KT_PREFILL_STREAM_THRESHOLD=1 KT_DYNAMIC_RESIDENT=0 \
KT_NUM_GPU_EXPERTS=32 KT_CPUINFER=128 CHUNKED_PREFILL_SIZE=8192 \
KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
NPU_DEVICE_ID=6 PORT=8530 bash tools/p27_launch_ds4flash_npu.sh
```
Dynamic hot-32: same but `KT_DYNAMIC_RESIDENT=1`.
- `KT_PREFILL_STREAM_THRESHOLD=1` forces streaming on short prompts (default 512; GPQA<512 would
  else fall to hybrid CPU-MoE and not exercise the depool path).
- dedup/streaming is for FAST prefill (converts all 256 on NPU); the 32 resident experts are direct
  int8 regardless. Drop streaming (`KT_PREFILL_STREAM=0`) -> hybrid prefill ~137s, no dedup needed.

## Root-causes nailed this session (all by controlled A/B, see decode_dynamic_vs_static_hitrate.md)
1. **Long prefill 95s -> 30s**: the GGUF->pinned stage-copy ran SINGLE-THREADED (1.6 GB/s on K920);
   `_par_copy` over KT_MXFP4_COPY_THREADS=32 threads -> ~15-24 GB/s. (memory: dedup-prefill-slow-...)
2. **Dynamic decode 12 vs static 18**: NOT dedup, NOT hit-rate (hot-32 43% vs prefix 14%, 3x better),
   NOT streaming. The resident MASK buffers (gpu_experts_mask/logical_to_gpu_index) were not remapped
   to caching-allocator (only the weights were) -> rewriting them each prefill triggered the NSA
   weight-region stall. FIX = remap the masks too -> 12 -> 16.4. (memory: dynamic-resident-decode-slow-...)
3. **dynamic hot-32 has NO net decode benefit over static prefix-32** (~equal within noise, both
   short and long seq), despite 3x hit-rate: the CPU-MoE is overlapped (side stream), so reducing it
   (86%->57%) doesn't move the critical path. The fix removed a regression; it didn't add a win.
   => the dynamic-resident machinery isn't paying off for decode in this overlapped config.

## Commit inventory (this session, on top of side-stream/dedup-core base)
ESSENTIAL (the functional deliverable):
- sglang dc2ca9f2c GGUF dedup (137G saving) + 53704c843 reserve slot (full-context OOM fix)
- sglang c4bba1152 + 6097116f9 parallel stage-copy + 32 threads (prefill 95s->30s)
- sglang 62a011dd7 mask remap (decode 12->16.4)
- parent 8842015 half-block packing + 176d1e4 convert-into-slot
OPTIONAL (keep or drop):
- parent 91d4c84 blk kernel (raw-block convert, ~3s; default on; fallback _di still works if dropped)
- sglang a500fc4b3 hit-rate probe (KT_HITRATE_PROBE, gated, zero-cost off; useful for regression)
- parent 2f40c22 / a092193 / ed39ab6 decode investigation docs (root-cause record)
ADD-THEN-REVERTED CHURN (net-zero; squash for a clean PR):
- sglang 313e2abcb removed the KT_STREAM_TIMING debug + gated overlap WIP it had introduced
- parent 60ff6b7 + 67d5a36 added then reverted the KT_CONVERT_SYNC overlap fence + overlap tests
  (overlap shelved: primitive validated 1.6x but full-forward integration has an unresolved global
  slowdown; 3 hypotheses ruled out — see the reverted commits / memory if revisited)

## Long-context retrieval — root-caused to NSA selection, NOT our stream/MoE path
Needle-in-a-haystack A/B (tools/p27_curl_f2_prompts.sh prompts 5/6/7, same ~7k log, only the needle
POSITION changes): short log (dense) retrieves the key; ~7k log with needle in the MIDDLE fails;
~7k log with needle at the TAIL (inside NSA's recent/sliding dense window) retrieves it. Same length,
position-dependent => the long-context "wrong answer" is **NSA sparse-block SELECTION dropping the
middle block** (a shared attention-backend issue), not dedup, not streaming, not the base model.
(memory: longcontext-needle-fail-is-nsa-selection.) The hybrid reference path shows the same -> not
ours to fix in the prefill path.

The long-context (NSA-selection) failure is a shared attention-backend workstream, separate from and
not blocking the dedup/prefill/decode deliverable here.

## 32k+ prefill: HBM budget (not done; levers documented)
A single 32k chunk OOMs on the streaming convert (~0.9G short): model 42G + reserved slot 6.4G + KV
pool (~9.8G at mem-frac 0.85 / context 65536) leaves ~1G for activation, and 32k activation + convert
need ~2.6G -> OOM -> graceful hybrid fallback (slow CPU prefill, request still completes).
Levers (need measurement to confirm):
- `--context-length 34816` (just above the prompt) shrinks the KV pool to ~5G -> frees ~5G for
  activation. THE right lever for a known max length. (memory: dsv4-npu-expert-capacity / hbm-budget)
- `KT_NUM_GPU_EXPERTS=16` frees the resident pool (~3-6G).
- lowering `--mem-fraction-static` frees activation BUT shrinks the KV pool below the 32k prompt's
  ~33k-token need -> fails with "not enough tokens" instead. Trades one limit for another.
- Multi-chunk (chunked-prefill-size 8192) is NOT a fix: prompt > chunk-size hits the cross-chunk NSA
  crash (memory: chunked-prefill-compressor-crosschunk-bug; roadmap P2).
- p27 prompt 5/6/7 are sized ~7k = single chunk (< 8192) so they run clean; the comment shows how to
  scale to 32k (range~960 + chunked-prefill-size>=32768 + freed HBM).

## Known issues / not done
- decode is noisy/prompt-dependent; measure warm + median (a cold/peak single number misleads).
- dynamic-resident decode-neutral (see #3) — revisit only if CPU-MoE becomes the decode bottleneck.
- p27 prompt-5/6/7 (long-context discriminator) + the 8k case are UNCOMMITTED test-script edits.

## Diagnostics
- `KT_HITRATE_PROBE=1` (+ `--disable-cuda-graph` so decode runs eager) -> per-step resident hit-rate,
  bucketed prefill/decode; `KT_HITRATE_PATH=<file>` to persist.
- `KT_DYN_FORCE_PREFIX=1` (force resident = prefix 0..K-1; pre-existing, isolates experts-vs-write).
  (`KT_DEPOOL_RES_SKIP_WEIGHTS` was a temporary skip-weight-write probe, reverted after the A/B.)
- `KT_STREAM_TIMING` was removed (was a per-phase wall-clock; sync-distorted, see history).
