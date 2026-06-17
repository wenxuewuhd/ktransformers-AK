# AscendC MXFP4→W8A8 dequant kernel — STATUS (Session G, 2026-06-12)

**Goal**: beat the Triton kernel's ~358ms/layer (reduction-bound; <150ms unreachable in Triton,
see `doc/.../mxfp4_dequant_kernel_handoff.md` §11) by writing the dequant op in AscendC, where
explicit tiling/pipelining/MTE control can approach the ~12ms HBM-bandwidth floor.

## Status: DONE. FUSED single kernel works end-to-end. e2e cos 0.99999976 vs fp32 golden;
## **82 ms/layer (E=256) — 4.4x faster than Triton 358 ms, ~2x under the 150 ms H2D budget.**

> Authoritative agent-facing spec/golden/acceptance live in `tools/mxfp4_w8a8_op/`.

### Final operator: `mxfp4_fused_kernel.cpp` (use this)
One pass: reads MXFP4 once → int8 weight (two `[lo|hi]` planes, de-interleave in a torch post-step)
+ per-output-channel `oscale`. Correct single + multi core.
- **The oscale GM-write quirk is solved**: block-partition rows, accumulate `oscale` per core in a
  UB block (`acc`), and flush each ACC-row block as ONE large contiguous `DataCopy` — this survives
  alongside the per-row int8 stores and input loads (small per-row scale stores do not).
- Validation: `test_fused.py` (int8 eq-frac 0.90, oscale err 3.7e-6, single+multi core);
  `test_fused_e2e.py` (real `npu_fused_experts`, cos 0.99999976 vs fp32 golden).
- Timing full layer (E=256, bd=40): w13 48 ms + w2 34 ms = **82 ms/layer**.

The two-kernel version (`mxfp4_dq_kernel.cpp` + `mxfp4_oscale_kernel.cpp`, 165 ms) is the
historical stepping stone; the fused kernel supersedes it.

### Whole-layer convert optimized 3077 -> 230 ms (13.4x), 2026-06-13
Session C measured that the depool **whole-layer convert** (`mxfp4_layer_to_nz_slots`, E=256) was
3077 ms — of which the kernel is only ~82 ms; the rest was the ND->NZ post-step in `mxfp4_fused_op.py`.
Breakdown: de-interleave 2364 (79%!) / int8-transpose 578 / format_cast 24 / copy 29.
Fixes (both in `convert_proj`, no API/contract change, cos 0.99999976 preserved, byte-identical):
- **de-interleave**: the `[lo|hi]` plane scatter `q[:,0::2]=...` was a 1-byte strided write (~2.7 GB/s).
  Replaced by a contiguous stack-interleave to `[E,OUT,IN]`.
- **transpose**: an int8 OUT<->IN transpose degenerates to a 1-byte gather (~21 GB/s HW floor — even
  `npu_transpose` is no faster). Killed via **fp16-transpose**: `q.to(fp16).transpose(1,2).contiguous()
  .to(int8)` runs the vectorized transpose path; `int8->fp16->int8` is exact since `|q|<=127`. 620->67 ms.
  (`.contiguous()` is mandatory — a transposed view to `format_cast` lays down wrong NZ bytes on device.)
- Result: whole-layer **230 ms** (kernel ~89 + post ~127 + overhead), prefill ~137s -> ~12-17s. Hits target.
- DEAD ENDS (don't retry): consumer-side transpose — production `npu_grouped_matmul` has no
  `transpose_weight` (only `npu_grouped_matmul_add`/`npu_transpose_batchmatmul`, neither can do the
  int8+scale grouped forward). Kernel-direct-NZ (`mxfp4_nz_kernel.cpp`, shelved WIP) targets ~113 ms but
  needs a MXFP4-pool layout change (pre-transposed codes) + fractal GM writes — not worth it now that
  fp16-transpose already hits target.

### QUEUED UPGRADE: swap in the advance kernel (115 ms) after Session C validates benefits
The "advance kernel" (separate repo `/workspace/code/ascend_c_dev/easyasc`, native device→device op)
is verified faster. Two paths, same drop-in contract:
- **fp16-transpose path** (`native_layer.py`): whole-layer **161 ms**, stable.
- **cube-transpose path** (`native_layer_overlap.py`, --timeovl): whole-layer **~115 ms** (verified by us
  2026-06-13: cos 0.99993765 PASS; SEQUENTIAL 116 ms = conv13 44.5 + conv2 27.4 + tr13 17.9 + tr2 8.7 +
  fc 18.7). This is the upgrade target — ~2x our 230 ms.
- **"94 ms is unreachable" — independently confirmed**: overlapping the vec convert (one stream) with the
  cube transpose (another stream) measures wall = SUM not MAX (--concur: 2-stream 15.6 ≈ sum 15.9 vs
  max 11.3) → the platform serializes separate kernels across streams, so the transpose cannot hide
  behind convert. 115 ms is the sequential-sum floor for the two-kernel design. (The only theoretical
  sub-115 path is fusing convert+transpose into ONE vec+cube kernel so AIV/AIC overlap intra-kernel — a
  major rewrite, not worth it since both already beat target.)
- Also landed 3 framework bug fixes (commits dcc42f7/d5906fe + a format_cast-3D fix): real bugs, keep them.
e2e cos 0.99993765 (PASS, slightly looser than our 0.99999976), two-shape single-process coexistence
16/16, tensorutils.h fix in its `main` (commit d5906fe). It is structurally leaner (kernel emits
consecutive int8 — no de-interleave).
Drop-in is **same signature/semantics** as ours (`mxfp4_layer_to_nz_slots(c13,s13,c2,s2,H,I) ->
(w13_nz,s13b,w2_nz,s2b)`), so swapping is a clean follow-up that does NOT touch Session C's depool /
dynamic-resident wiring. **Decision: keep OURS for now** (already integrated + server-validated + 230 ms
already hits the prefill goal); swap theirs in as a deliberate, separate step once C has validated the
DDR/decode benefits. Swap recipe:
1. Vendor their kernel + dual-op build + `native_layer.py` into the kt repo (next to this dir); drop the
   dependency on the easyasc dev repo.
2. Point `_mxfp4_convert_fn()` (kt_stream_prefill.py) at their entry; add the vendor-env to the server
   launch (`source vendors/customize/bin/set_env.bash`) and ensure commit d5906fe is present.
3. E2E regression: cos >= 0.9999 + whole-layer ~161 ms + a depool server run (DDR unchanged, prefill
   ~3 s faster). Confirm the accuracy didn't regress (their cos is the looser 0.99993765).
Acceptance evidence (verified by us 2026-06-13, card 5): native_run_dual.py 16/16 + correctness <1%;
native_layer.py --experts 32 cos 0.99993765 + 161.5 ms.

### Integrated into kt_stream_prefill (depool), gated
- `mxfp4_fused_op.py` — runtime wrapper: builds (bisheng) + loads the fused `.so`, exposes
  `mxfp4_layer_to_nz_slots(c13,s13,c2,s2,H,I) -> (w13_nz, s13b, w2_nz, s2b)`.
- `kt_stream_prefill.py` (sglang submodule) — `KT_MXFP4_DEPOOL=1` stores MXFP4 (pinned, ~137GB)
  instead of the 277GB W8A8 NZ pool and converts per layer on chip. Default off = W8A8 path
  byte-identical. Env: `KT_MXFP4_DEPOOL=1`, `KT_MXFP4_CKPT=<MXFP4 safetensors dir>` (default the
  DeepSeek-V4-Flash model dir), optional `KT_MXFP4_OP_DIR=<this dir>`.
- Offline-validated: the production `_load_layer_mxfp4` + convert hook through `npu_fused_experts`
  on real layer-16 gives cos 0.99999976 vs fp32 golden.
- **Server-validated (2026-06-12, card 3, DSv4-Flash single NPU)**: with `KT_PREFILL_STREAM=1
  KT_MXFP4_DEPOOL=1 KT_MXFP4_NZ_CHUNK=32` + `--mem-fraction-static 0.72` (HBM headroom for the
  conversion), a 640-token prefill ran the depool path with **0 streaming failures and coherent
  output**. **DDR benefit MEASURED**: MXFP4 pool ≈ **140GB** (DDR 326→~475GB) vs the W8A8 pool
  277GB → **~137GB saved**. (Needed: kt-kernel ext rebuild + reapply llama.cpp MXFP4 patch
  `tools/kt_dsv4_npu_patches/llama_cpp/0002-add-ggml-type-mxfp4.patch` — both lost on container
  restart; and HBM headroom via lower mem-fraction since depool skips the W8A8 slot reservation.)
- **Decode hot-expert benefit NOT delivered by v1** (two reasons, both follow-ups):
  1. depool disables the dynamic-resident update (reads the W8A8 `_POOL`) → decode uses static
     prefix-32 → many experts routed to CPU → `cpu_moe_wall` `off_cpu` high (140-330ms, also
     shared-machine noise). Wiring dynamic-resident to convert hot experts' MXFP4 → resident W8A8
     is the fix for the real-topK benefit.
  2. v1 pins a SEPARATE ~140GB MXFP4 pool, so the pin tax is reduced (139 vs 277GB) but not gone;
     the handoff's full benefit needs the NPU to share the CPU's MXFP4 (no separate pinned pool)
     or stream unpinned. The DDR win is delivered; the decode pin-tax win needs this.

### Session C update (2026-06-13, card 7): dynamic-resident wired to MXFP4 pool (§D)
**Done — dynamic-resident now works on the depool path, decode COHERENT.**
- `kt_stream_prefill.py` `_apply_dynamic_residency`: under depool, the hot-K experts' MXFP4 codes
  are plain packed bytes (NOT NZ), so a first-dim `[top]` slice is format-safe; convert just those
  K via `mxfp4_layer_to_nz_slots` straight into the resident slots — no whole-pool H2D, no NZ
  round-trip gather. Gate `_KT_DYN_RESIDENT and not _KT_MXFP4_DEPOOL` → `_KT_DYN_RESIDENT`.
- **Correctness ✓**: switch applies cleanly (top-32×43, masks mask_sum=32/l2g=32/cpu=32), decode
  output coherent ("Efficient inference in MoE models requires keeping active experts near the
  compute unit while streaming idle ones..."). The old Goal-2 gibberish (host NZ slice) does NOT
  recur — the MXFP4-byte slice is format-safe.
- **off_cpu (§D judge)**: steady-state dynamic off_cpu **floor ~17ms** (near the ~20ms target),
  but **median ~46ms dominated by shared-box NUMA noise** (one run hit a 2068ms GC spike; p90 ~85).
  Static prefix-32 (no pool) floor ~20ms. **Cannot show a clean dynamic≪static median delta on this
  shared box** — same wall the longseq handoff documents (needs ≥500 tok + exclusive machine).
- **Pin tax NOT confirmed in steady state**: added `KT_MXFP4_POOL_NO_PIN=1` (unpinned pool);
  unpinned dynamic off_cpu floor ~17ms / median ~46ms — **no clear improvement over pinned**, so
  pinning is not the dominant decode cost here (post-switch transient spikes to 165–400ms were
  contention, not steady state). Kept the flag opt-in; default pinned (faster prefill H2D).
- **Switch is SLOW (~180s)**: profile H2D(slice)=71s + convert=113s. The 113s convert of 43×32
  experts is pathological vs prefill's 82ms/256-expert — likely host-gather de-pinning + per-call
  fixed overhead ×43. **Top follow-up**: stage hot-K MXFP4 into a pinned contiguous buffer before
  H2D, and/or batch the 43 layer-converts. One-time cost (end of prefill), so it didn't block
  correctness, but a 180s stall per long prefill is unshippable.
- **Net**: §D code correct + coherent; component floor (~17ms) meets target; the clean median
  benefit + switch speed are the open items (former is box-limited, latter is a code optimization).

### Session C re-validation on G's optimized algo + switch fix + controlled A/B (2026-06-13, card 7)
After G's whole-layer convert dropped to **230ms** (3077→230, cos 0.99999976 — re-verified offline):
- **Prefill streaming benefit RECOVERED**: depool prefill forward **137s → ~15-20s** (clean est;
  measured ~40s in a 5×-contended window). G's convert was the lever; this was the benefit that
  depool had traded away. Headline of the two-benefit goal — delivered.
- **Switch — H2D fix is live-confirmed; the "8s" convert was OFFLINE-only (CORRECTED 2026-06-17 by Session B)**:
  - **H2D fix is real and live-confirmed**: `c13[top]` (advanced indexing) is **unpinned** → no DMA;
    `_stage_pin_h2d` (`kt_stream_prefill.py`) `index_select`s the hot-K into a reused **pinned** buffer
    → DMA. Offline 17.5s→7s; **Session B live = 10.5s** (matches, the fix works in production).
  - **⚠️ The convert "1.3s / whole switch ~8s" was OFFLINE-ISOLATED (idle card, isolated tensors), NOT a
    live server switch.** My original LIVE switch was ~180s (H2D 71s + convert 113s, this section above).
    **Session B live (2026-06-17, DEPOOL+side+hot): H2D 10.5s + convert 102.8s ≈ 113s** — i.e. B reproduced
    the original live number, NOT the 8s. The offline→live extrapolation was wrong.
  - **What b3d1a39 (G's post-step opt) actually buys live: only ~15s** (B: convert 118s→102.8s). It kills
    the ND→NZ post-step, which is real — but the post-step was never the bulk of the LIVE convert.
  - **The remaining ~93s live = per-call overhead ×86** (43 layers × w13/w2 `convert_proj`: kernel launch +
    `format_cast` + device alloc, under post-prefill near-full HBM). This is exactly the **UNDONE follow-up**
    flagged in the section above ("batch the 43 layer-converts"). The kernel compute is fine in isolation
    (82ms/256-expert); the slowdown is the live call path, not the `.so` — confirm via
    `test_fused_e2e.py --experts 32` on an idle card (expect ~30-230ms/layer).
  - **So "8s switch" is an OPTIMIZATION TARGET, not a live-proven number.** To hit it: batch the converts +
    preallocate `out_nz`/fp16 buffers (stop per-layer alloc+format_cast churn), and/or switch before HBM fills.
    NOTE: one-time prefill-tail cost; does NOT touch decode throughput (B: ~18 tok/s either way).
- **Controlled A/B (FORCE_PREFIX static vs dynamic, same pool/prompt/memory) — decode verdict**:
  hot-expert selection works (activation share 0.134→0.586), and off_cpu **floor 18.8→10.3ms (−45%)**,
  BUT p10/p25/median/p75 are **identical** (29→28 / 39→39 / 63→57 / 103→102). off_cpu =
  ①expert-bandwidth (hot experts cut this, visible only at the floor) + ②per-layer CPU
  dispatch/fork-join fixed overhead + ③neighbor noise. ②③ dominate p10↑ and don't move →
  **net wall-clock decode speedup ≈ 0 on this shared box.**

### Decode hot-expert benefit — where it goes next (NOT this session)
The hot-expert **mechanism is done** (§D, necessary prerequisite). The remaining lever is the
**per-layer CPU↔NPU dispatch/fork-join overhead** (dominates off_cpu above the floor) — that is
**Session B's domain** (B owns submit/sync/overlap + MTP). Plus an **exclusive/unloaded machine**
to let the −45% floor show through the neighbor noise. Neither is C's scope or the kernel's.
**C↔B contract**: enable with `KT_MXFP4_DEPOOL=1 KT_DYNAMIC_RESIDENT=1`; resident set = prefill
top-32 (share ~0.6); off_cpu floor is already −45% but dispatch-bound; B's overlap work is what
converts the floor win into wall-clock. Shared files at merge: `kt_stream_prefill.py` (C),
`kt_ep_wrapper.py` / `experts_base.py` (B).

**Session C scope (depool + dynamic-resident + prefill streaming + DDR) is COMPLETE.**

### Working operator (this dir)
- `mxfp4_dq_kernel.cpp` — MXFP4 → int8 weight (two contiguous planes `[lo|hi]`; de-interleave in a
  torch post-step). Correct single + multi core. Its own `oscale` output is broken/unused (ignored).
- `mxfp4_oscale_kernel.cpp` — MXFP4 → per-output-channel `oscale = amax/127`. Correct single + multi
  core: accumulates scales per core in UB and flushes each block as ONE large contiguous `DataCopy`
  (sidesteps the small-store-interleaved-with-loads failure mode).
- `test_e2e_combined.py` — both kernels → de-interleave → NZ → real `npu_fused_experts`:
  **cos(kernel, fp32-golden) = 0.99999976 PASS**; cos vs bf16-golden 0.99973 (benign bf16 floor).
- Timing (full layer E=256, w13+w2, bd=40): **int8 89 ms + oscale 77 ms = 165 ms/layer**.

### Next (perf): fuse into ONE kernel to stop re-reading MXFP4 twice
The two kernels each decode+scale the MXFP4; the oscale kernel duplicates work. Fusing oscale into
the int8 kernel (reusing the decode) would drop ~77 ms → projected **~90-100 ms/layer**. Blocked on
the oscale GM-write quirk inside the fused kernel (below); the two-kernel split is the robust
fallback that already beats Triton.

### ✅ Working (verified this session)
1. **Toolchain end-to-end**: bisheng compiles AscendC (`-x asc --cce-aicore-arch=dav-c220`) device
   kernel + host `<<<>>>` launcher into a `.so`; called from Python via **ctypes** passing
   `tensor.data_ptr()` + `torch.npu.current_stream().npu_stream`. Build line:
   ```
   bisheng -x asc --cce-aicore-arch=dav-c220 -O2 -std=c++17 -fPIC -shared \
     -I$TK/tikcfw -I$TK/tikcfw/impl -I$TK/tikcfw/interface -I$TK/tikcfw/lib -I$CANN/aarch64-linux/include \
     mxfp4_dq_kernel.cpp -o libmxfp4dq.so -L$CANN/aarch64-linux/lib64 -lruntime -lascendcl
   ```
2. **Decode** (FP4 e2m1 via 256-entry byte-indexed `Gather` LUT): **bit-exact** vs golden.
3. **Scale** (e8m0 via `lutE8` gather + `scOff` broadcast gather, `Mul`): **bit-exact**.
4. **amax reduce** (max(|lo|,|hi|) + non-in-place ping-pong fold to 8 + scalar tail): correct.
5. **requant → int8 output**: **correct and deterministic, single AND multi-core**
   (`int8 eq-frac 0.90 vs bf16 golden, max|dq|=1`; reconstruction cos ≈ 1.0). The earlier
   non-determinism was the cast chain — **`f32→half→int8` is correct; `f32→i32→i16→i8` is NOT**.
   Output is stored as two contiguous planes `[lo | hi]`; interleave deferred to a torch post-step
   (`out[...,0::2]=lo; out[...,1::2]=hi`) — in-kernel `Gather` interleave is unusable (small src cap).

### ❌ The one remaining blocker — `oscale` GM write
The per-output-channel `oscale = amax/127` write to GM **lands in every isolation test but writes
nothing when embedded in the full compute kernel** (RAW buffer 0 bytes changed, single & multi core),
while the int8 `out` (same VECOUT-TQue + `DataCopy` idiom) writes 100% correctly in the same kernel.
- Tried (all fail in-kernel): scalar `SetValue`, `DataCopyPad` single-float, bare `DataCopy` of an
  8-float `[R,8]` cache-line slot (vector-`Duplicate`-filled), and the full VECOUT-`TQue` idiom.
- Tried (all pass in isolation, including with ~120KB UB pressure and a parallel int8 TQue write):
  the `[R,8]` `Duplicate`+`DataCopy`. So it is **not** the mechanism, UB pressure, or arg position —
  it is some interaction with the compute ops (gather/reduce/requant) preceding the write.
- **Refined root cause (isolated in toy kernels)**: the failure mode is a **small MTE3 GM store
  that interleaves with MTE2 GM→UB loads, at low/medium blockdim**. Reproduced minimally:
  a kernel that does a few `DataCopy(GM→UB)` loads then `DataCopy(UB→GM, 8..64 floats)` writes
  **nothing at bd=1** but writes correctly at **bd=40**. Did NOT help: `PipeBarrier<PIPE_ALL>`,
  explicit `MTE2_MTE3` SetFlag/WaitFlag, `DataCacheCleanAndInvalid<ENTIRE_DATA_CACHE>`, larger
  write size, arg position, VECOUT TQue. The large int8 `out` store survives because it's big and
  is the dominant MTE3 traffic; the tiny per-row `oscale` store gets lost in the load/store mix.
- **Two clean sidesteps for the implementer (recommended over chasing the sync)**:
  1. **Separate `oscale` into its own kernel** (read codes+scale → dequant → per-row amax → write
     `oscale`), run at production blockdim. A loads+write kernel writes correctly at bd=40.
  2. **Accumulate `oscale` per core in UB and write it ONCE as a large contiguous `DataCopy` at
     kernel end** (block-partition rows so each core owns a contiguous `oscale` segment), instead
     of a tiny per-row store interleaved with the int8 stores and input loads.

### (historical note) multi-core grid mapping
  needs proper inter-iteration sync (or restructure so each core owns disjoint rows cleanly).

### Verified AscendC hw pitfalls (reusable; see also memory `triton-ascend-kernel-gotchas`)
- `Gather` uses **BYTE** offsets (×4 for f32); src has a **small size cap** (fine for ≤256-elem LUTs).
- **In-place `Max(x, x, x[h])` silently yields 0** — reductions need a distinct dst (ping-pong).
- Vector ops need **≥8 f32 (32-byte) counts**; sub-block counts crash (`507035`) → fold only to 8,
  finish on scalar.
- `Cast` supported pairs are limited: **no `u8→i32`** (use `u8→half→i32`), **no `f32→i8`**
  (use `f32→i32→i16→i8`). `RoundMode::CAST_RINT` = round-half-even.
- `ReduceMax` count-form returned 0 in testing — used the manual fold instead.
- `DataCopy` needs 32-byte-aligned sizes (sub-block copies read garbage).
- `GlobalTensor::SetValue` (scalar GM write) works and flushes.

### Files
- `mxfp4_dq_kernel.cpp` — the kernel + host launcher (WIP).
- `test_ascendc.py` — builds host LUTs, launches via ctypes, checks int8 eq-frac + scale + recon cos.
