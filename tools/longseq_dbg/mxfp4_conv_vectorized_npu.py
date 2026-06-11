#!/usr/bin/env python3
"""Phase-1 operator-level experiment: VECTORIZED MXFP4 -> W8A8 -> NZ on Ascend NPU.

Goal: implement mxfp4_to_w8a8_nz() as a fully-vectorized NPU op chain (NO python
per-expert / per-element loops; experts are CHUNKED only to bound HBM, but every
op inside a chunk is whole-tensor vectorized), then:

  (A) prove correctness == the verified CPU path
      dequant_native()  (tools/verify_mxfp4_layer.py)
      -> quant_per_outchannel_bf16()  (mxfp4_to_w8a8_accuracy.py)
      i.e. int8 equal-fraction == 1.0 and per-channel scale exactly equal.

  (B) time T_conv_vectorized at full layer scale (E=256, one MoE layer) with
      per-substep breakdown: unpack+dequant / requant / nz-format_cast.

  (C) verdict vs T_h2d_w8a8 (345 ms) and target ~150 ms.

Native MXFP4 storage (from real DSv4-Flash safetensors):
  w1/w3 : weight int8 [OUT=I,  IN/2=H/2]  scale e8m0 [OUT=I, H/32]
  w2    : weight int8 [OUT=H,  IN/2=I/2]  scale e8m0 [OUT=H, I/32]
  byte b holds nibble pair -> IN pos 2b (lo), 2b+1 (hi)        (consecutive order)
  value = FP4_TABLE[nibble] * 2^(e8m0 - 127)   (per block-32 along IN)

W8A8 requant (verified path): per OUTPUT channel (row, dim=IN) amax -> int8.

Card via ASCEND_RT_VISIBLE_DEVICES. Reads checkpoints directly, no server.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve()
_TOOLS = _HERE.parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import json  # noqa: E402
from safetensors import safe_open  # noqa: E402
from verify_mxfp4_layer import dequant_native  # noqa: E402  (the verified CPU reference)

DEV = "npu"
NZ = 29  # ACL_FORMAT_FRACTAL_NZ
H = 4096
I = 2048
E = 256

# FP4 codebook, same order as verify_mxfp4_layer.FP4_TABLE
FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ============================================================================
#  THE VECTORIZED CONVERSION  (no per-expert / per-element python loops)
# ============================================================================
def _fp4_lut(dtype):
    return torch.tensor(FP4_TABLE, dtype=dtype, device=DEV)


def dequant_chunk(codes_u8, scale_u8, OUT, IN, fp4):
    """codes_u8 [ec,OUT,IN/2] uint8 (nibble pairs, consecutive order),
       scale_u8 [ec,OUT,IN/32] uint8 (e8m0 exponent bytes)
       -> bf16 [ec,OUT,IN]. Fully vectorized."""
    ec = codes_u8.shape[0]
    nb = IN // 32
    # --- nibble unpack: bitwise-and / bit-shift, whole-tensor ---
    lo = (codes_u8 & 0x0F).long()        # [ec,OUT,IN/2] -> IN pos 0,2,4,...
    hi = (codes_u8 >> 4).long()          # [ec,OUT,IN/2] -> IN pos 1,3,5,...
    # --- FP4 LUT (gather/index) ---
    v_lo = fp4[lo]
    v_hi = fp4[hi]
    vals = torch.empty(ec, OUT, IN, dtype=fp4.dtype, device=DEV)
    vals[..., 0::2] = v_lo
    vals[..., 1::2] = v_hi
    # --- per-block-32 scale: e8m0 byte -> 2^(e-127) ; vectorized ---
    # exact 2^(e-127) via int bit trick (matches verify_mxfp4_layer.e8m0_to_fp32),
    # computed in fp32 then cast to dequant dtype.
    e = scale_u8.to(torch.int32)
    scale_f32 = ((e << 23).view(torch.float32)) if False else torch.ldexp(
        torch.ones_like(e, dtype=torch.float32), e - 127)
    scale = scale_f32.to(fp4.dtype)                       # [ec,OUT,nb]
    bf = (vals.view(ec, OUT, nb, 32) * scale.unsqueeze(-1)).view(ec, OUT, IN)
    return bf


def dequant_chunk_v2(codes_u8, scale_u8, OUT, IN, fp4):
    """Optimized dequant: contiguous stack-reshape interleave (no strided scatter),
       int32 gather indices, broadcast scale. Same math as dequant_chunk."""
    ec = codes_u8.shape[0]
    nb = IN // 32
    lo = (codes_u8 & 0x0F).int()
    hi = (codes_u8 >> 4).int()
    v_lo = fp4[lo]                       # [ec,OUT,IN/2]
    v_hi = fp4[hi]
    # contiguous interleave: stack on a new last dim then flatten -> 0,1,0,1,...
    vals = torch.stack([v_lo, v_hi], dim=-1).reshape(ec, OUT, nb, 32)
    e = scale_u8.to(torch.int32)
    scale = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127).to(fp4.dtype)
    bf = (vals * scale.unsqueeze(-1)).reshape(ec, OUT, IN)
    return bf


def dequant_chunk_v3(codes_u8, scale_u8, OUT, IN, fp4):
    """Fastest vectorized dequant. KEY: replace advanced-indexing gather `fp4[idx]`
    (pathologically slow on torch_npu, ~261ms/chunk) with `index_select` on a flat
    index tensor (~2ms/chunk), and use contiguous stack-reshape interleave (v2).
    Bit-identical to dequant_chunk."""
    ec = codes_u8.shape[0]
    nb = IN // 32
    lo = (codes_u8 & 0x0F).int().reshape(-1)
    hi = (codes_u8 >> 4).int().reshape(-1)
    shp = (ec, OUT, IN // 2)
    v_lo = fp4.index_select(0, lo).reshape(shp)
    v_hi = fp4.index_select(0, hi).reshape(shp)
    vals = torch.stack([v_lo, v_hi], dim=-1).reshape(ec, OUT, nb, 32)
    e = scale_u8.to(torch.int32)
    scale = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127).to(fp4.dtype)
    bf = (vals * scale.unsqueeze(-1)).reshape(ec, OUT, IN)
    return bf


def requant_chunk(bf):
    """bf [ec,OUT,IN] -> int8 [ec,OUT,IN], scale [ec,OUT] (per OUTPUT channel)."""
    amax = bf.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    s = amax / 127.0
    q = (bf / s).round().clamp(-127, 127).to(torch.int8)
    return q, s.squeeze(-1)


_DEQ = {"v1": dequant_chunk}  # v2 added after its def


def mxfp4_to_w8a8_nz(codes_u8, scale_u8, OUT, IN, fp4, dq_dtype, do_nz=True, deq=dequant_chunk):
    """Full chunked op chain -> (int8 NZ [ec,IN,OUT], scale [ec,OUT])."""
    bf = deq(codes_u8, scale_u8, OUT, IN, fp4.to(dq_dtype))
    q, s = requant_chunk(bf)
    if do_nz:
        qk = torch_npu.npu_format_cast(q.transpose(1, 2).contiguous(), NZ)  # [ec,IN,OUT]
    else:
        qk = q
    return qk, s.to(torch.bfloat16)


# ============================================================================
#  CORRECTNESS  (real data vs verified CPU path)
# ============================================================================
def _load_weight_map(md):
    return json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]


def _open_shard(md, wm, cache, key):
    sh = wm[key]
    if sh not in cache:
        cache[sh] = safe_open(md / sh, framework="pt")
    return cache[sh]


def _as_u8(t):
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return t.contiguous()


def quant_per_outchannel_cpu(w_f32):
    """CPU/torch reference requant (matches mxfp4_to_w8a8_accuracy.quant_per_outchannel_bf16
    but with bf16 cast on the master, to mirror that path exactly)."""
    w = torch.from_numpy(w_f32).to(torch.bfloat16)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.bfloat16)


def load_real_chunk(md, wm, cache, layer, experts, proj):
    """Load raw uint8 codes + uint8 e8m0 scale for a list of experts of one proj.
    Returns codes_u8 [ec,OUT,IN/2], scale_u8 [ec,OUT,nb], OUT, IN (host tensors)."""
    cs, ss = [], []
    for e in experts:
        wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
        sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
        h = _open_shard(md, wm, cache, wk)
        cs.append(_as_u8(h.get_tensor(wk)))
        ss.append(_as_u8(h.get_tensor(sk)))
    codes = torch.stack(cs)
    scl = torch.stack(ss)
    OUT = codes.shape[1]
    IN = codes.shape[2] * 2
    return codes, scl, OUT, IN


def correctness(md, layer, n_experts, dq_dtype, deq=dequant_chunk):
    wm = _load_weight_map(md)
    cache = {}
    experts = list(range(n_experts))
    fp4 = _fp4_lut(torch.float32)

    # ---- w13 = cat(w1,w3) along OUT ----
    print(f"[correctness] layer={layer} experts={n_experts} dq_dtype={dq_dtype}")
    for tag, projs in (("w13", ("w1", "w3")), ("w2", ("w2",))):
        # build the combined codes/scale on host
        chunks_c, chunks_s = [], []
        cpu_q_list, cpu_s_list = [], []
        for proj in projs:
            codes, scl, OUT, IN = load_real_chunk(md, wm, cache, layer, experts, proj)
            chunks_c.append(codes)
            chunks_s.append(scl)
            # CPU reference: dequant_native -> quant per out-channel, per expert
            for i, e in enumerate(experts):
                deq_ref = dequant_native(codes[i].numpy(), scl[i].numpy())  # [OUT,IN] f32
                q, s = quant_per_outchannel_cpu(deq_ref)
                cpu_q_list.append(q)
                cpu_s_list.append(s)
        if tag == "w13":
            # interleave so order is [e0_w1, e0_w3]? accuracy.py does cat([w1,w3],dim=0)
            # per expert -> OUT=2I. Build same combined layout for NPU + CPU.
            codes = torch.cat(chunks_c, dim=1)   # [ec, 2*OUT, IN/2]  (w1 rows then w3 rows)
            scl = torch.cat(chunks_s, dim=1)
            OUT = codes.shape[1]
            IN = codes.shape[2] * 2
            # CPU combined: for each expert cat(q_w1,q_w3) along OUT
            cpu_q = torch.stack([torch.cat([cpu_q_list[i], cpu_q_list[n_experts + i]], dim=0)
                                 for i in range(n_experts)])
            cpu_s = torch.stack([torch.cat([cpu_s_list[i], cpu_s_list[n_experts + i]], dim=0)
                                 for i in range(n_experts)])
        else:
            codes = chunks_c[0]
            scl = chunks_s[0]
            cpu_q = torch.stack(cpu_q_list)
            cpu_s = torch.stack(cpu_s_list)

        # ---- NPU vectorized (no NZ for elementwise compare) ----
        codes_d = codes.to(DEV)
        scl_d = scl.to(DEV)
        npu_q, npu_s = mxfp4_to_w8a8_nz(codes_d, scl_d, OUT, IN, fp4, dq_dtype, do_nz=False, deq=deq)
        npu_q = npu_q.cpu()
        npu_s = npu_s.cpu()

        eqfrac = (npu_q.int() == cpu_q.int()).float().mean().item()
        maxdiff = (npu_q.int() - cpu_q.int()).abs().max().item()
        s_abs = (npu_s.float() - cpu_s.float()).abs().max().item()
        print(f"  [{tag}] OUT={OUT} IN={IN}  int8 equal-frac={eqfrac:.6f}  "
              f"max|dq|={maxdiff}  scale max|err|={s_abs:.3e}")
    return


# ============================================================================
#  TIMING  (full layer E=256, substep breakdown)
# ============================================================================
def median_ms(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def timing(iters, chunk, dq_dtype, deq=dequant_chunk):
    fp4 = _fp4_lut(dq_dtype)

    def make(OUT, IN):
        codes = torch.randint(0, 256, (E, OUT, IN // 2), dtype=torch.uint8, device=DEV)
        scl = torch.randint(118, 136, (E, OUT, IN // 32), dtype=torch.uint8, device=DEV)
        return codes, scl

    c13, s13 = make(2 * I, H)     # w13 [E,2I,H]
    c2, s2 = make(H, I)           # w2  [E,H,I]

    # ---- substep closures (chunked, vectorized inside) ----
    def full():
        for a in range(0, E, chunk):
            b = min(a + chunk, E)
            mxfp4_to_w8a8_nz(c13[a:b], s13[a:b], 2 * I, H, fp4, dq_dtype, do_nz=True, deq=deq)
            mxfp4_to_w8a8_nz(c2[a:b], s2[a:b], H, I, fp4, dq_dtype, do_nz=True, deq=deq)

    def unpack_dequant():
        for a in range(0, E, chunk):
            b = min(a + chunk, E)
            deq(c13[a:b], s13[a:b], 2 * I, H, fp4)
            deq(c2[a:b], s2[a:b], H, I, fp4)

    # requant-only: one bf buffer per shape, replayed E/chunk times (per-chunk work
    # is identical, so total == full-layer requant without holding 13GB of masters).
    def requant_only_factory():
        bf13 = torch.randn(chunk, 2 * I, H, dtype=dq_dtype, device=DEV)
        bf2 = torch.randn(chunk, H, I, dtype=dq_dtype, device=DEV)

        def fn():
            for a in range(0, E, chunk):
                requant_chunk(bf13)
                requant_chunk(bf2)
        return fn, (bf13, bf2)

    def nz_only_factory():
        q13 = torch.randint(-127, 127, (chunk, 2 * I, H), dtype=torch.int8, device=DEV)
        q2 = torch.randint(-127, 127, (chunk, H, I), dtype=torch.int8, device=DEV)

        def fn():
            for a in range(0, E, chunk):
                torch_npu.npu_format_cast(q13.transpose(1, 2).contiguous(), NZ)
                torch_npu.npu_format_cast(q2.transpose(1, 2).contiguous(), NZ)
        return fn

    t_full = median_ms(full, iters)
    t_unpack = median_ms(unpack_dequant, iters)
    rq_fn, _hold = requant_only_factory()
    t_requant = median_ms(rq_fn, iters)
    del _hold
    torch.npu.empty_cache()
    t_nz = median_ms(nz_only_factory(), iters)

    del c13, s13, c2, s2
    torch.npu.empty_cache()
    return t_full, t_unpack, t_requant, t_nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--correct-experts", type=int, default=8,
                    help="experts to elementwise-verify vs CPU reference")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dq-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="dequant intermediate dtype; bf16 mirrors the verified path")
    ap.add_argument("--dequant-impl", choices=["v1", "v2", "v3"], default="v1",
                    help="v1=strided-scatter+advanced-index gather (slow); "
                         "v2=contiguous stack-reshape + advanced-index gather; "
                         "v3=stack-reshape + index_select LUT (fast)")
    ap.add_argument("--skip-correct", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    torch.npu.set_device(0)  # ASCEND_RT_VISIBLE_DEVICES remaps
    dq_dtype = torch.bfloat16 if args.dq_dtype == "bf16" else torch.float32
    deq = {"v1": dequant_chunk, "v2": dequant_chunk_v2,
           "v3": dequant_chunk_v3}[args.dequant_impl]
    print(f"[config] dequant_impl={args.dequant_impl}")

    if not args.skip_correct:
        correctness(args.model_dir, args.layer_idx, args.correct_experts, dq_dtype, deq=deq)

    if not args.skip_timing:
        print(f"\n[timing] full layer E={E} chunk={args.chunk} iters={args.iters} "
              f"dq_dtype={args.dq_dtype} impl={args.dequant_impl}")
        t_full, t_unpack, t_requant, t_nz = timing(args.iters, args.chunk, dq_dtype, deq=deq)
        T_GMM = 11.0
        T_H2D_W8A8 = 345.0
        print(f"\n=== T_conv_vectorized substep breakdown (median ms/layer, E=256) ===")
        print(f"  unpack+dequant : {t_unpack:8.2f} ms")
        print(f"  requant(int8)  : {t_requant:8.2f} ms")
        print(f"  nz format_cast : {t_nz:8.2f} ms")
        print(f"  -------------------------------")
        print(f"  T_conv (full)  : {t_full:8.2f} ms   (sum-of-parts="
              f"{t_unpack + t_requant + t_nz:.2f})")
        lhs = t_full + T_GMM
        print(f"\n=== VERDICT ===")
        print(f"  T_conv + T_gmm(11) = {lhs:.2f} ms")
        print(f"  vs T_h2d_w8a8=345 : {'PASS' if lhs < T_H2D_W8A8 else 'FAIL'} "
              f"(<345 means conv hides in saved-H2D budget)")
        print(f"  vs ~150 target    : {'PASS' if lhs < 150 else 'FAIL'} "
              f"(<150 means fits under MXFP4 H2D leg ~192ms)")


if __name__ == "__main__":
    main()
