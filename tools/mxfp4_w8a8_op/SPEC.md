# Operator spec — MXFP4 → W8A8 fused dequant/requant (Ascend 910B3)

Self-contained spec for implementing one NPU operator. Everything needed is in **this directory**:

| file | role |
|---|---|
| `SPEC.md` | this spec — IO / math / performance / acceptance |
| `golden.py` | torch **golden** (accuracy reference; validated against the real network, eager+slow) |
| `acceptance.py` | end-to-end **acceptance** harness through the production `npu_fused_experts` |

SoC: `Ascend910B3`.

---

## 1. What the operator does

Convert one MoE layer's **MXFP4 (4-bit, GGUF-native) expert weights** to **W8A8**
(int8 weight + per-output-channel scale) on the NPU, fused, fast enough to hide under the H2D of
streaming (target ≲150 ms/layer; HBM-bandwidth floor ~12 ms). It replaces a 277 GB resident W8A8
pool with on-the-fly conversion.

---

## 2. Inputs (per MoE projection, per expert)

- **codes** `uint8 [OUT, IN/2]` — each byte holds 2 nibbles, **consecutive order**:
  byte `b` → IN position `2b` (low nibble `code & 0x0F`), `2b+1` (high nibble `code >> 4`).
- **scale** `uint8 [OUT, IN/32]` — **e8m0** (8-bit exponent). Every **32 IN elements** share one scale.
  Decode **exactly**: `scale_f32 = reinterpret_f32(uint32(e) << 23)`  (== `2^(e-127)`).

**FP4 e2m1 code table** (nibble 0..15 → value):
```
[ 0, 0.5, 1, 1.5, 2, 3, 4, 6,   0, -0.5, -1, -1.5, -2, -3, -4, -6 ]
```

**Dimensions** (DeepSeek-V4-Flash, one layer; H=4096, I=2048, E=256):

| projection | OUT | IN | IN/2 | IN/32 |
|---|---|---|---|---|
| **w13** = concat(gate w1, up w3) along OUT | 2*I = 4096 | H = 4096 | 2048 | 128 |
| **w2** = down | H = 4096 | I = 2048 | 1024 | 64 |

(w1/w3 are each `[E, I, H]`; host concatenates to w13 `[E, 2I, H]`. E = 256 experts per layer.)

---

## 3. Math (per expert, per output channel = per row)

1. **Dequant**: `w[in] = FP4[nibble(in)] * 2^(e8m0[in/32] - 127)` → `[IN]` (exact in bf16/fp32).
2. **Requant, per output channel** (per row, reduction over IN):
   `amax = max_in |w[in]|` (clamp ≥ 1e-8); `oscale = amax / 127`;
   `q_int8[in] = round(w[in] / oscale)` clamped to `[-127, 127]`.

The scale granularity is **per output channel** (one scale per row; reduction along IN only).

---

## 4. Outputs

- **q_int8** `int8 [OUT, IN]` — consecutive order (same layout as dequant).
- **oscale** `[OUT]` — per-output-channel scale `= amax/127` (bf16 for the downstream op).

### Downstream (the operator does NOT do this; just produce outputs that feed it)
`q_int8.transpose(1,2).contiguous()` → `torch_npu.npu_format_cast(·, 29)` (ND→FRACTAL_NZ) →
`npu_fused_experts(w=NZ int8 [E,IN,OUT], w_scale=oscale bf16 [E,OUT], ...)`.
The NZ format-cast is NPU-native and already fast — **out of scope** for this operator.

---

## 5. Expected performance / roofline

Per layer (E=256): read codes+scale ≈ **3.4 GB**, write int8 ≈ **6.4 GB** → HBM round-trip ≈ **9.8 GB**.
910B3 HBM ≈ 0.8–1.3 TB/s ⇒ **bandwidth floor ≈ 8–12 ms/layer**. This is a memory-bound,
elementwise + one per-row reduction kernel.

- **Target**: **≲ 150 ms/layer** (w13+w2 combined, warmup first). Order-of-magnitude is enough;
  ≲150 ms hides the conversion under the per-layer MXFP4 H2D (~150–180 ms).
- **Stretch**: approach the ~12 ms floor (large tiles + double-buffered MTE + efficient reduction).
- For reference, a straightforward eager PyTorch conversion is ~3.4 s/layer (pathological — the
  whole point is to beat it by ~100×).

---

## 6. Acceptance

1. **Decode bit-exact**: the dequant stage (before requant) matches `golden.dequant()` exactly
   (`max|err| == 0`; use the `e<<23` bit trick for e8m0, not `exp2`).
2. **End-to-end functional**: feed candidate output through native NZ + `npu_fused_experts` and
   compare to the golden path — `acceptance.py` reports
   **cosine(candidate, reference) ≥ 0.9999**. (Internal fp32 vs bf16 differences are fine; accuracy
   is judged at the GEMM output, not by bit-identical int8.)
3. **Performance**: full layer (w13+w2, E=256) conversion **≲ 150 ms** after warmup.

### How to validate
```bash
# golden self-test
python3 golden.py
# end-to-end acceptance (plug your kernel into candidate_proj_to_nz in acceptance.py)
ASCEND_RT_VISIBLE_DEVICES=<free card> python3 acceptance.py --experts 32
```
Out of the box `acceptance.py`'s candidate == golden, so cosine prints 1.0; replace the one marked
call site (`candidate_proj_to_nz`) with your kernel and re-run.

---

## 7. Golden (accuracy reference)

`golden.py :: mxfp4_to_w8a8_golden(codes_u8[OUT,IN/2], scale_u8[OUT,IN/32]) -> (int8[OUT,IN], bf16 scale[OUT])`
— the exact semantics above, validated end-to-end against the real network (eager, slow). This is
the source of truth for both the math and the acceptance comparison.
