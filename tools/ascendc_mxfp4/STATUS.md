# AscendC MXFP4→W8A8 dequant kernel — STATUS (Session G, 2026-06-12)

**Goal**: beat the Triton kernel's ~358ms/layer (reduction-bound; <150ms unreachable in Triton,
see `doc/.../mxfp4_dequant_kernel_handoff.md` §11) by writing the dequant op in AscendC, where
explicit tiling/pipelining/MTE control can approach the ~12ms HBM-bandwidth floor.

## Status: WORKING end-to-end via TWO kernels (int8 + oscale). e2e cos 0.99999976 vs fp32 golden;
## 165 ms/layer (E=256), 2.2x faster than the Triton 358 ms. The oscale GM-write quirk in the
## single fused kernel was sidestepped, not solved (see below).

> Authoritative agent-facing spec/golden/acceptance live in `tools/mxfp4_w8a8_op/`.

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
