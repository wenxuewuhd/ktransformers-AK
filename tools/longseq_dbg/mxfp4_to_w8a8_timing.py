#!/usr/bin/env python3
"""Experiment (2): TIMING — can the MXFP4->W8A8 conversion hide inside the saved H2D?

One MoE layer, DSv4-Flash: H=4096, I=2048, E=256, topk=6, prefill M=4096.

Four medians (warmup + N iters, torch.npu.synchronize at boundaries):

  T_h2d_w8a8  : H2D one layer W8A8  w13[E,H,2I]+w2[E,I,H] int8 (~6.4GB) pinned host -> device.
                <- CURRENT baseline tempo (full W8A8 pool streamed in).
  T_h2d_mxfp4 : H2D one layer MXFP4 raw blob (~3.2GB, half the bytes) pinned host -> device.
                <- new tempo: only the 4-bit blob crosses PCIe.
  T_conv      : MXFP4 -> bf16 dequant -> per-output-channel W8A8 requant -> NZ format_cast,
                done on NPU. UPPER BOUND (does the full dequant+requant+NZ on-device).
  T_gmm       : grouped_matmul one layer at prefill M=4096 (two gmm + swiglu), the real
                expert compute the conversion competes with for NPU cycles.

Feasibility: T_conv + T_gmm < T_h2d_w8a8  => conversion fits in the HALF of the H2D we
save by shipping MXFP4 instead of W8A8; new tempo = T_h2d_mxfp4 (~half) AND -277GB DDR.

Conversion and gmm both run on the NPU and cannot overlap each other; H2D (DMA) overlaps
with NPU compute, so the per-layer pipeline tempo = max(T_h2d_mxfp4, T_conv + T_gmm).
"""
import argparse
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"
NZ = 29
H = 4096
I = 2048
E = 256
TOPK = 6


def median_ms(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def bench_h2d(nbytes_blobs, iters):
    """nbytes_blobs: list of (shape, dtype). Pinned host -> device copy timing."""
    host = [torch.empty(shape, dtype=dt).pin_memory() for shape, dt in nbytes_blobs]
    dev = [torch.empty(shape, dtype=dt, device=DEV) for shape, dt in nbytes_blobs]

    def fn():
        for h, d in zip(host, dev):
            d.copy_(h, non_blocking=True)

    ms = median_ms(fn, iters)
    total_bytes = sum(t.numel() * t.element_size() for t in host)
    del host, dev
    torch.npu.empty_cache()
    return ms, total_bytes


def bench_conv(iters, chunk=32):
    """MXFP4 raw blob on device -> dequant -> per-out-channel int8 requant -> NZ.

    Models the on-NPU conversion cost as an UPPER BOUND. We synthesize the MXFP4
    blob as nibble codes + ue8m0 scales on device, then per expert-chunk:
      1. dequant: unpack nibbles via lookup -> bf16, * 2^(e-127) per block-32
      2. requant: per-output-channel amax -> int8
      3. NZ format_cast
    Done for w13 [E,2I,H] and w2 [E,H,I]. Experts are processed in chunks of
    `chunk` to bound the bf16 intermediate (the whole-layer bf16 master would be
    ~24GB); the summed NPU work — and thus time — is identical to one shot.
    """
    fp4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       dtype=torch.bfloat16, device=DEV)

    def make_blob(OUT, IN):
        codes = torch.randint(0, 256, (E, OUT, IN // 2), dtype=torch.uint8, device=DEV)
        exps = torch.randint(118, 136, (E, OUT, IN // 32), dtype=torch.uint8, device=DEV)
        return codes, exps

    def dequant_requant_nz_chunk(codes, exps, OUT, IN):
        ec = codes.shape[0]
        nb = IN // 32
        lo = (codes & 0x0F).int()
        hi = (codes >> 4).int()
        v_lo = fp4[lo]
        v_hi = fp4[hi]
        vals = torch.empty(ec, OUT, IN, dtype=torch.bfloat16, device=DEV)
        vals[..., 0::2] = v_lo
        vals[..., 1::2] = v_hi
        scale = torch.exp2(exps.to(torch.bfloat16) - 127.0)              # [ec,OUT,nb]
        bf = (vals.view(ec, OUT, nb, 32) * scale.unsqueeze(-1)).view(ec, OUT, IN)
        amax = bf.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        s = amax / 127.0
        q = (bf / s).round().clamp(-127, 127).to(torch.int8)
        return torch_npu.npu_format_cast(q.transpose(1, 2).contiguous(), NZ)

    c13, e13 = make_blob(2 * I, H)   # w13 [E,2I,H]
    c2, e2 = make_blob(H, I)          # w2  [E,H,I]

    def fn():
        for a in range(0, E, chunk):
            b = min(a + chunk, E)
            dequant_requant_nz_chunk(c13[a:b], e13[a:b], 2 * I, H)
            dequant_requant_nz_chunk(c2[a:b], e2[a:b], H, I)

    ms = median_ms(fn, iters)
    del c13, e13, c2, e2
    torch.npu.empty_cache()
    return ms


def bench_nz_only(iters, chunk=32):
    """Isolated cost of just the int8 ND->NZ format_cast for one layer's two weights.

    This is the one NPU-native op the conversion MUST do regardless of how dequant
    is implemented; a fused dequant+requant kernel would add to this floor."""
    out13 = torch.randint(-127, 127, (chunk, H, 2 * I), dtype=torch.int8, device=DEV)
    out2 = torch.randint(-127, 127, (chunk, I, H), dtype=torch.int8, device=DEV)

    def fn():
        for a in range(0, E, chunk):
            torch_npu.npu_format_cast(out13.contiguous(), NZ)
            torch_npu.npu_format_cast(out2.contiguous(), NZ)

    ms = median_ms(fn, iters)
    del out13, out2
    torch.npu.empty_cache()
    return ms


def bench_gmm(M, iters):
    N = M * TOPK
    hs = torch.randn(N, H, dtype=torch.bfloat16, device=DEV)
    w13 = torch.randn(E, H, 2 * I, dtype=torch.bfloat16, device=DEV) * 0.02
    w2 = torch.randn(E, I, H, dtype=torch.bfloat16, device=DEV) * 0.02
    base = N // E
    counts = torch.full((E,), base, dtype=torch.int64)
    counts[: (N - base * E)] += 1
    grp = counts.to(DEV)

    def fn():
        h = torch.ops.npu.npu_grouped_matmul(
            x=[hs], weight=[w13], bias=None, split_item=2, group_list_type=1,
            group_type=0, group_list=grp, output_dtype=torch.bfloat16)[0]
        h = torch.ops.npu.npu_swiglu(h)
        torch.ops.npu.npu_grouped_matmul(
            x=[h], weight=[w2], bias=None, split_item=2, group_list_type=1,
            group_type=0, group_list=grp, output_dtype=torch.bfloat16)[0]

    ms = median_ms(fn, iters)
    del hs, w13, w2
    torch.npu.empty_cache()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-m", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    torch.npu.set_device(0)

    # W8A8 layer = w13 int8 [E,H,2I] + w2 int8 [E,I,H]  (kernel NZ layout, same byte count)
    w8a8_blobs = [((E, H, 2 * I), torch.int8), ((E, I, H), torch.int8)]
    # MXFP4 raw blob = half the nibble bytes + small e8m0 scales
    mxfp4_blobs = [((E, 2 * I, H // 2), torch.uint8), ((E, 2 * I, H // 32), torch.uint8),
                   ((E, H, I // 2), torch.uint8), ((E, H, I // 32), torch.uint8)]

    print(f"# DSv4-Flash one MoE layer  H={H} I={I} E={E} topk={TOPK} prefill_M={args.prefill_m} iters={args.iters}")

    t_h2d_w8a8, b_w8a8 = bench_h2d(w8a8_blobs, args.iters)
    t_h2d_mxfp4, b_mxfp4 = bench_h2d(mxfp4_blobs, args.iters)
    t_conv = bench_conv(args.iters)
    t_nz = bench_nz_only(args.iters)
    t_gmm = bench_gmm(args.prefill_m, args.iters)

    # bandwidth floor for conversion: read ~3.2GB MXFP4 + write 6.4GB int8 on HBM.
    conv_bytes = b_mxfp4 + b_w8a8
    bw_floor_ms = conv_bytes / 1e9 / 800.0 * 1e3   # assume ~800 GB/s effective HBM

    print(f"\n=== TIMING (median ms/layer) ===")
    print(f"  T_h2d_w8a8  = {t_h2d_w8a8:9.2f} ms   ({b_w8a8/1e9:.2f} GB, {b_w8a8/1e9/(t_h2d_w8a8/1e3):.1f} GB/s)  <- current baseline")
    print(f"  T_h2d_mxfp4 = {t_h2d_mxfp4:9.2f} ms   ({b_mxfp4/1e9:.2f} GB, {b_mxfp4/1e9/(t_h2d_mxfp4/1e3):.1f} GB/s)  <- new tempo (half bytes)")
    print(f"  T_conv      = {t_conv:9.2f} ms   (MXFP4->bf16->W8A8(per-chan)->NZ, eager PyTorch UPPER BOUND)")
    print(f"  T_nz_only   = {t_nz:9.2f} ms   (just int8 ND->NZ format_cast, NPU-native floor of conv)")
    print(f"  T_gmm       = {t_gmm:9.2f} ms   (grouped_matmul prefill M={args.prefill_m})")
    print(f"  conv_bw_floor~{bw_floor_ms:8.2f} ms   (read {b_mxfp4/1e9:.1f}+write {b_w8a8/1e9:.1f} GB @ ~800 GB/s HBM)")

    rhs = t_h2d_w8a8
    print(f"\n=== FEASIBILITY:  T_conv + T_gmm  <  T_h2d_w8a8 ? ===")
    for label, conv in (("eager upper bound", t_conv), ("nz-floor + gmm only", t_nz),
                        ("bandwidth floor", bw_floor_ms)):
        lhs = conv + t_gmm
        verdict = "PASS" if lhs < rhs else "FAIL"
        print(f"  [{verdict}] {label:22s}: {conv:9.2f} + {t_gmm:.2f} = {lhs:9.2f} ms  vs  T_h2d_w8a8={rhs:.2f} ms"
              f"  -> tempo=max(T_h2d_mxfp4,lhs)={max(t_h2d_mxfp4, lhs):.1f}ms")
    print(f"\n  With T_h2d_mxfp4={t_h2d_mxfp4:.1f}ms: IF conv hides under the {rhs-t_h2d_mxfp4:.0f}ms saved-H2D budget,")
    print(f"  new tempo ~= T_h2d_mxfp4 ({t_h2d_mxfp4:.0f}ms) = ~{rhs/t_h2d_mxfp4:.2f}x vs baseline AND -277GB DDR.")
    print(f"  Budget for (conv+gmm) to stay copy-bound on the MXFP4 leg: <= T_h2d_mxfp4 = {t_h2d_mxfp4:.1f} ms.")


if __name__ == "__main__":
    main()
