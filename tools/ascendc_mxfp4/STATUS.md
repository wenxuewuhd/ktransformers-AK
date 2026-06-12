# AscendC MXFP4→W8A8 dequant kernel — STATUS (Session G, 2026-06-12)

**Goal**: beat the Triton kernel's ~358ms/layer (reduction-bound; <150ms unreachable in Triton,
see `doc/.../mxfp4_dequant_kernel_handoff.md` §11) by writing the dequant op in AscendC, where
explicit tiling/pipelining/MTE control can approach the ~12ms HBM-bandwidth floor.

## Status: WIP. The int8 weight output is CORRECT (single + multi core); only the per-channel
## `oscale` GM write is unresolved. Do NOT wire into production until oscale is fixed.

> Authoritative agent-facing spec/golden/acceptance live in `tools/mxfp4_w8a8_op/` — use those
> to (re)implement. This dir is the WIP reference kernel + hard-won findings.

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
- Likely a pipe/sync ordering issue specific to this op mix; the expert AscendC agent should solve
  with the right idiom (e.g. dedicated output TQue flushed after the int8 store, or a separate
  reduce-only pass emitting `oscale`).

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
