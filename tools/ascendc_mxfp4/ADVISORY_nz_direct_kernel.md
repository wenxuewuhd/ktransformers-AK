# Advisory → G: direct-NZ kernel (path 3) — suggestions, not directives

From the kernel-advisor session (2026-06-13). You own the kernel; these are suggestions +
questions + a delivered validation harness. Push back freely.

## Delivered for you to use
- **`nz_byte_check.py`** — byte-level validator for a kernel that writes int8 FRACTAL_NZ GM
  directly. `verify_against_format_cast(nz_kernel, nd_ref)` byte-compares your output to the
  proven `npu_format_cast(nd_ref,29)` AND cross-checks format_cast against the pure layout
  formula (defense in depth). Self-test passes. Use **byte-identity**, not just cos — cos can
  smear a tile-placement bug; a wrong tile shows up as a hard byte mismatch.
- **Layout formula independently re-verified** (0/2048 mismatch, fresh run, not just the memory):
  `p(in,out) = ((out//32)*ceil(IN/16) + (in//16))*512 + (in%16)*32 + (out%32)`, tile = 16 IN × 32 OUT.
  Safe to drive your GM writes off this.

## The design tension I'd nail down first (amax vs UB)
Your output write needs the **per-output-channel amax** (oscale) to quantize, but amax is a
reduction over **all IN** for that channel. Channel-vectorized (32 ch × IN fp) doesn't fit UB
(32×4096×2B = 256KB > 192KB), so you can't "decode → reduce → quantize → write" in one sweep the
way the current row-vectorized kernel does (it holds one full row = IN values).

Two ways out — which are you taking?
1. **2-pass**: pass A = amax only (you already produce correct per-channel oscale in
   `mxfp4_fused_kernel.cpp`; reuse it as-is), pass B = channel-vectorized, decode + quantize with
   the known scale + assemble 16×32 tiles + `DataCopy` 512B contiguous. Decode runs twice
   (~89ms ×2) but you delete transpose+format_cast (~600ms) — still a big net win, and pass B
   never needs to hold IN-wide fp data. **This is the one I'd suggest** — clean separation, reuses
   proven amax.
2. **Single-pass deferred**: tile IN, keep a running per-channel amax, but you can't write int8
   until amax is final → you'd buffer decoded values or re-walk. Messier; only worth it if the
   second decode actually shows up as a bottleneck (I doubt it will).

## Read-side stride (the mirror of the write-side scatter you already solved)
Codes are `[E, OUT, IN/2]` (output-major). 32 output channels at a fixed input position are 32
rows apart (stride IN/2). If you read them as a "column" you reintroduce a strided 1-byte read —
the same failure mode as single-channel NZ writes, just on input. Suggestion: **DataCopy 32 full
rows into UB, then take the 32-wide column in UB** (on-chip strided access is far cheaper than GM),
or transpose the 32×(IN/2) block once in UB. Worth confirming the read doesn't become the new floor.

## Validation ladder I'd use
1. `python3 nz_byte_check.py` (self-test, already green).
2. Your kernel on one real layer → `verify_against_format_cast(nz_kernel, nd_golden_int8)` →
   **byte-identical** (catches tile math).
3. `test_fused_e2e.py` → cos 0.99999976 unchanged (catches dequant/scale regressions).
4. Time `mxfp4_layer_to_nz_slots` E=256: target ≪ 788ms (FUSED fallback), aim ~200ms; warmup first.

## Questions
- Q1: 2-pass or single-pass for amax? (drives the whole UB budget)
- Q2: How are you feeding the 32-channel group — 32-row DataCopy + UB column, or strided GM read?
- Q3: UB budget after adding the 32×16 tile staging — does it still fit alongside the LUT/scale
  buffers, or does HALF_MAX tiling shrink?
- Q4: Output contract unchanged? If `mxfp4_layer_to_nz_slots` still returns FRACTAL_NZ `[E,IN,OUT]`
  + bf16 `[E,OUT]`, Session C's wiring needs zero change. If the kernel now emits NZ and you drop
  the wrapper's transpose/format_cast, confirm the returned tensor is still NZ-tagged (allocate via
  `format_cast(zeros,29)` then overwrite bytes — see `int8-fractal-nz-layout` memory — so it carries
  the right format tag for `npu_fused_experts`).

## Don't re-chase (already dead, per your own measurements)
Consumer-side transpose (no transpose flag in `npu_grouped_matmul`), transposed-view→format_cast
(fake 14ms / real 1034ms), hardware int8 transpose (~21GB/s floor, 620ms). Direct-NZ-in-kernel is
the only path that dodges the physical-transpose wall — agreed.
