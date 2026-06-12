# AscendC MXFP4→W8A8 dequant kernel — STATUS (Session G, 2026-06-12)

**Goal**: beat the Triton kernel's ~358ms/layer (reduction-bound; <150ms unreachable in Triton,
see `doc/.../mxfp4_dequant_kernel_handoff.md` §11) by writing the dequant op in AscendC, where
explicit tiling/pipelining/MTE control can approach the ~12ms HBM-bandwidth floor.

## Status: WORK IN PROGRESS — not yet end-to-end correct. Do NOT wire into production.

The **toolchain and 3 of 4 compute stages are verified correct**; the requant→int8 output path
has a stubborn non-determinism, and multi-core needs sync hardening.

### ✅ Proven (this session)
1. **Toolchain end-to-end**: bisheng compiles AscendC (`-x asc --cce-aicore-arch=dav-c220`) device
   kernel + host `<<<>>>` launcher into a `.so`; called from Python via **ctypes** passing
   `tensor.data_ptr()` + `torch.npu.current_stream().npu_stream`. A minimal `addone` kernel ran
   bit-exact. Build line:
   ```
   bisheng -x asc --cce-aicore-arch=dav-c220 -O2 -std=c++17 -fPIC -shared \
     -I$TK/tikcfw -I$TK/tikcfw/impl -I$TK/tikcfw/interface -I$TK/tikcfw/lib -I$CANN/aarch64-linux/include \
     mxfp4_dq_kernel.cpp -o libmxfp4dq.so -L$CANN/aarch64-linux/lib64 -lruntime -lascendcl
   ```
2. **Decode** (FP4 e2m1 via 256-entry byte-indexed `Gather` LUT): **bit-exact** vs `dequant_native`.
3. **Scale** (e8m0 via `lutE8` gather + `scOff` broadcast gather, `Mul`): **bit-exact**.
4. **amax reduce** (max(|lo|,|hi|) + non-in-place ping-pong fold to 8 + scalar tail): the stored
   per-channel `oscale = amax/127` matches the CPU reference to **1.86e-6** (single core).

### ❌ Remaining (the blockers)
- **requant→int8 output is non-deterministic** (changes run-to-run) even single-core, while the
  scale is deterministic+correct. So the bug is in the `Muls(inv)`→clamp→`Cast f32→i32→i16→i8`
  path or the reduce→requant transition. Suspected scalar↔vector (`inv`) sync or a WAR hazard in
  the two-plane output cast (shared `off`/`q16` temps reused for lo then hi without a barrier).
  Tried & did NOT fix: V_S/S_V `SetFlag`/`WaitFlag` (both `FetchEventID` and fixed `EVENT_ID`),
  `PipeBarrier<PIPE_ALL>`. Next: bisect with the working `scaledbg`-style harness — add reduce,
  then requant, then output, one at a time, dumping floats each step.
- **Interleave**: in-kernel `Gather` interleave fails — `Gather`'s src appears capped to a small
  window (~1KB; worked for the 256-elem decode LUT, fails reading the 16KB `comb`). Current code
  sidesteps it by storing two contiguous planes `[lo | hi]` and deferring the interleave to a cheap
  torch post-step (`out[...,0::2]=lo; out[...,1::2]=hi`). A fully in-kernel interleave needs a
  strided `DataCopyPad` or `Transpose`, not `Gather`.
- **Multi-core**: with `blockdim>1` the scale also goes flaky → the per-row reduce/scalar path
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
