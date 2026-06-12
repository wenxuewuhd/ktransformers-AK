#!/usr/bin/env python3
"""Fused MXFP4 -> int8 (+ per-output-channel scale) dequant/requant Triton kernel
for Ascend NPU. Single on-chip pass: no bf16 intermediate materialized.

Layout (native DSv4-Flash MXFP4, consecutive nibble order):
  codes [R, HALF=IN/2] uint8   byte b -> IN pos 2b(lo = b&0xF), 2b+1(hi = b>>4)
  scale [R, NB=IN/32]  uint8   e8m0 exponent; value = FP4[nibble] * 2^(e-127)
                               (per block-32 along IN; bytes b and b share block b//16
                                -> a [NB,16] tile shares one scale per row block)

Requant: per OUTPUT channel (= per row) amax over IN -> int8, chan_scale = amax/127.

The kernel decodes e2m1 arithmetically (no LUT gather), broadcasts the block scale
over a [NB,16] tile (no gather load), reduces row amax, requants, and interleaves
lo/hi back to contiguous [IN] int8 -- all on chip.
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

_round = tl.extra.ascend.libdevice.round


@triton.jit
def _decode_e2m1(n):
    """n: int tile of nibbles 0..15 -> fp32 e2m1 value (bit-exact to FP4 table).
    2^(exp-1) via integer shift (no transcendental exp2)."""
    sign = (n >> 3) & 1
    exp = (n >> 1) & 3
    mant = (n & 1).to(tl.float32)
    # 2^(exp-1) for exp in {1,2,3} via selects -- no transcendental, no variable shift
    base = tl.where(exp == 1, 1.0, tl.where(exp == 2, 2.0, 4.0))
    mag = tl.where(exp == 0, mant * 0.5, base * (1.0 + mant * 0.5))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def mxfp4_dequant_requant_kernel(
    codes_ptr, scale_ptr, out_ptr, oscale_ptr,
    R,
    HALF: tl.constexpr, NB: tl.constexpr, IN: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    blk = tl.arange(0, NB)[:, None]          # [NB,1] block index within row
    jj = tl.arange(0, 16)[None, :]           # [1,16] byte within block
    boff = blk * 16 + jj                      # [NB,16] byte offset within a row
    even = blk * 32 + jj * 2                  # [NB,16] int8 offset for lo nibbles
    odd = even + 1                            # [NB,16] int8 offset for hi nibbles
    nbrange = tl.arange(0, NB)               # [NB] scale offset within a row

    for k in tl.range(ROWS_PER_PROG):
        r = (pid * ROWS_PER_PROG + k).to(tl.int64)   # int64: r*IN overflows int32 at E=256
        if r < R:
            cbase = codes_ptr + r * HALF
            codes = tl.load(cbase + boff).to(tl.int32)  # [NB,16]
            lo = codes & 0xF
            hi = (codes >> 4) & 0xF
            v_lo = _decode_e2m1(lo)
            v_hi = _decode_e2m1(hi)
            e = tl.load(scale_ptr + r * NB + nbrange).to(tl.int32)     # [NB]
            bscale = ((e << 23).to(tl.float32, bitcast=True))[:, None]  # exact 2^(e-127)
            v_lo = v_lo * bscale
            v_hi = v_hi * bscale
            amax = tl.maximum(tl.max(tl.abs(v_lo)), tl.max(tl.abs(v_hi)))
            amax = tl.maximum(amax, 1e-8)
            chan_s = amax / 127.0
            inv = 127.0 / amax     # reciprocal once per row: q = round(v*inv), not v/chan_s
            q_lo = tl.minimum(tl.maximum(_round(v_lo * inv), -127.0), 127.0).to(tl.int8)
            q_hi = tl.minimum(tl.maximum(_round(v_hi * inv), -127.0), 127.0).to(tl.int8)
            obase = out_ptr + r * IN
            tl.store(obase + even, q_lo)   # consecutive nibble order:
            tl.store(obase + odd, q_hi)    # lo->IN pos 2b, hi->2b+1
            tl.store(oscale_ptr + r, chan_s)


@triton.jit
def mxfp4_dequant_only_kernel(
    codes_ptr, scale_ptr, out_ptr, R,
    HALF: tl.constexpr, NB: tl.constexpr, IN: tl.constexpr, ROWS_PER_PROG: tl.constexpr,
):
    """Dequant-only (fp32 out, no requant) -- to prove decode is bit-exact vs reference."""
    pid = tl.program_id(0)
    blk = tl.arange(0, NB)[:, None]
    jj = tl.arange(0, 16)[None, :]
    boff = blk * 16 + jj
    even = blk * 32 + jj * 2
    odd = even + 1
    nbrange = tl.arange(0, NB)
    for k in tl.range(ROWS_PER_PROG):
        r = (pid * ROWS_PER_PROG + k).to(tl.int64)
        if r < R:
            codes = tl.load(codes_ptr + r * HALF + boff).to(tl.int32)
            v_lo = _decode_e2m1(codes & 0xF)
            v_hi = _decode_e2m1((codes >> 4) & 0xF)
            e = tl.load(scale_ptr + r * NB + nbrange).to(tl.int32)
            bscale = ((e << 23).to(tl.float32, bitcast=True))[:, None]  # exact 2^(e-127)
            obase = out_ptr + r * IN
            tl.store(obase + even, v_lo * bscale)
            tl.store(obase + odd, v_hi * bscale)


def mxfp4_dequant_only(codes_u8, scale_u8, IN, rows_per_prog=1):
    lead = codes_u8.shape[:-1]
    HALF, NB = codes_u8.shape[-1], scale_u8.shape[-1]
    R = 1
    for d in lead:
        R *= d
    codes = codes_u8.reshape(R, HALF).contiguous()
    scale = scale_u8.reshape(R, NB).contiguous()
    out = torch.empty((R, IN), dtype=torch.float32, device=codes.device)
    mxfp4_dequant_only_kernel[(triton.cdiv(R, rows_per_prog),)](
        codes, scale, out, R, HALF=HALF, NB=NB, IN=IN, ROWS_PER_PROG=rows_per_prog,
        multibuffer=False,
    )
    return out.reshape(*lead, IN)


def mxfp4_dequant_requant(codes_u8, scale_u8, IN, rows_per_prog=8):
    """codes_u8 [..., HALF] uint8, scale_u8 [..., NB] uint8 (e8m0).
    Returns (int8 [..., IN], chan_scale fp32 [...]). Leading dims flattened to R rows."""
    assert codes_u8.dtype == torch.uint8 and scale_u8.dtype == torch.uint8
    lead = codes_u8.shape[:-1]
    HALF = codes_u8.shape[-1]
    NB = scale_u8.shape[-1]
    assert HALF * 2 == IN and NB * 32 == IN, (HALF, NB, IN)
    R = 1
    for d in lead:
        R *= d
    codes = codes_u8.reshape(R, HALF).contiguous()
    scale = scale_u8.reshape(R, NB).contiguous()
    out = torch.empty((R, IN), dtype=torch.int8, device=codes.device)
    oscale = torch.empty((R,), dtype=torch.float32, device=codes.device)
    grid = (triton.cdiv(R, rows_per_prog),)
    mxfp4_dequant_requant_kernel[grid](
        codes, scale, out, oscale, R,
        HALF=HALF, NB=NB, IN=IN, ROWS_PER_PROG=rows_per_prog,
        multibuffer=False,
    )
    return out.reshape(*lead, IN), oscale.reshape(*lead)


# Grid must stay < 65536 programs; rows-per-prog picked so cdiv(E*OUT, rpp) < 65536.
def _rpp_for(E, OUT):
    import math
    R = E * OUT
    return max(1, math.ceil(R / 65535))


def mxfp4_proj_to_slot_nz(codes_u8, scale_u8, IN):
    """One MoE projection: MXFP4 codes/scale [E,OUT,*] -> (w_nz, scale_bf16) ready for
    npu_fused_experts. w_nz is FRACTAL_NZ int8 [E, IN, OUT]; scale is bf16 per-out-channel
    [E, OUT]. Mirrors the reference mxfp4_to_w8a8_nz transpose+NZ exactly."""
    import torch_npu
    E, OUT = codes_u8.shape[0], codes_u8.shape[1]
    q, s = mxfp4_dequant_requant(codes_u8, scale_u8, IN, rows_per_prog=_rpp_for(E, OUT))
    w_nz = torch_npu.npu_format_cast(q.transpose(1, 2).contiguous(), 29)   # [E,IN,OUT] NZ
    return w_nz, s.to(torch.bfloat16)


def mxfp4_layer_to_slots(c13, s13, c2, s2, H, I):
    """Full layer depool conversion. Inputs are this layer's combined MXFP4 (host or device):
      c13/s13 : w13 = cat(w1,w3) codes [E,2I,H/2] + e8m0 scale [E,2I,H/32]
      c2/s2   : w2  codes [E,H,I/2] + e8m0 scale [E,H,I/32]
    Returns (w13_nz, w13_scale_bf16, w2_nz, w2_scale_bf16) -- the exact tensors the
    streaming slot + npu_fused_experts consume, replacing the resident W8A8 pool.
    Validated end-to-end (cos 0.99999976 vs the verified vectorized path) in
    tools/longseq_dbg/test_mxfp4_kernel_e2e.py."""
    w13_nz, s13b = mxfp4_proj_to_slot_nz(c13, s13, H)
    w2_nz, s2b = mxfp4_proj_to_slot_nz(c2, s2, I)
    return w13_nz, s13b, w2_nz, s2b
