#!/usr/bin/env python3
"""Stake 2+3 driver for the fused Triton MXFP4->int8 kernel.

Correctness: real DSv4-Flash layer, per expert/proj, compare kernel int8+scale to the
verified CPU reference (dequant_native -> quant_per_outchannel_cpu). Reports int8
equal-fraction, max int8 diff, scale max-abs error, and post-dequant cosine
(int8*scale vs CPU master) -- the acceptance metric (§10.1).

Timing: full E=256 layer (w13 [E,2I,H] + w2 [E,H,I]), warmup then median ms/layer,
verdict vs H2D budget.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # tools/
sys.path.insert(0, str(_HERE.parent))              # tools/longseq_dbg/

from safetensors import safe_open  # noqa: E402
from verify_mxfp4_layer import dequant_native  # noqa: E402
from mxfp4_conv_vectorized_npu import (  # noqa: E402
    _load_weight_map, _open_shard, _as_u8, quant_per_outchannel_cpu,
)
from mxfp4_dequant_triton import mxfp4_dequant_requant, mxfp4_dequant_only  # noqa: E402

DEV = "npu"
H, I, E = 4096, 2048, 256


def load_real_chunk(md, wm, cache, layer, experts, proj):
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


def correctness(md, layer, n_experts, rows_per_prog):
    wm = _load_weight_map(md)
    cache = {}
    experts = list(range(n_experts))
    print(f"[correctness] layer={layer} experts={n_experts} rows_per_prog={rows_per_prog}")
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.reshape(-1).float(), b.reshape(-1).float(), dim=0).item()

    for proj in ("w1", "w3", "w2"):
        codes, scl, OUT, IN = load_real_chunk(md, wm, cache, layer, experts, proj)
        # CPU reference + TRUE fp32 master per expert
        cpu_q, cpu_s, true_master = [], [], []
        for i in range(n_experts):
            deq = dequant_native(codes[i].numpy(), scl[i].numpy())   # [OUT,IN] f32 (TRUE)
            true_master.append(torch.from_numpy(deq))
            q, s = quant_per_outchannel_cpu(deq)
            cpu_q.append(q)
            cpu_s.append(s)
        cpu_q = torch.stack(cpu_q)            # [ec,OUT,IN] int8
        cpu_s = torch.stack(cpu_s).float()    # [ec,OUT]
        true_master = torch.stack(true_master)  # [ec,OUT,IN] f32

        codes_d, scl_d = codes.to(DEV), scl.to(DEV)

        # (1) pure dequant bit-exact vs dequant_native
        kdeq = mxfp4_dequant_only(codes_d, scl_d, IN).cpu()
        deq_bitexact = torch.equal(kdeq, true_master)
        deq_maxabs = (kdeq - true_master).abs().max().item()

        # (2) fused requant
        npu_q, npu_s = mxfp4_dequant_requant(codes_d, scl_d, IN, rows_per_prog=rows_per_prog)
        torch.npu.synchronize()
        npu_q, npu_s = npu_q.cpu(), npu_s.cpu().float()

        eqfrac = (npu_q.int() == cpu_q.int()).float().mean().item()
        maxdiff = (npu_q.int() - cpu_q.int()).abs().max().item()
        ker_w = npu_q.float() * npu_s.unsqueeze(-1)     # reconstructed weight (kernel)
        ref_w = cpu_q.float() * cpu_s.unsqueeze(-1)      # reconstructed weight (reference)
        # (3) GEMM-level cosine vs TRUE weight (the §10.1 functional metric)
        T = 64
        X = torch.randn(T, IN)
        true_out = X @ true_master.reshape(-1, IN).T
        ker_out = X @ ker_w.reshape(-1, IN).T
        ref_out = X @ ref_w.reshape(-1, IN).T
        print(f"  [{proj}] OUT={OUT} IN={IN}  dequant_bitexact={deq_bitexact} "
              f"(max|err|={deq_maxabs:.1e})  int8 eq-frac={eqfrac:.4f} max|dq|={maxdiff}")
        print(f"       GEMM cos: kernel-vs-true={cos(ker_out, true_out):.8f}  "
              f"ref-vs-true={cos(ref_out, true_out):.8f}  kernel-vs-ref={cos(ker_out, ref_out):.8f}")


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


def timing(iters, rows_per_prog):
    c13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=DEV)
    s13 = torch.randint(118, 136, (E, 2 * I, H // 32), dtype=torch.uint8, device=DEV)
    c2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=DEV)
    s2 = torch.randint(118, 136, (E, H, I // 32), dtype=torch.uint8, device=DEV)

    def run():
        mxfp4_dequant_requant(c13, s13, H, rows_per_prog=rows_per_prog)
        mxfp4_dequant_requant(c2, s2, I, rows_per_prog=rows_per_prog)

    t = median_ms(run, iters)
    print(f"\n[timing] full layer E={E} rows_per_prog={rows_per_prog} iters={iters}")
    print(f"  T_dequant_requant (w13+w2) : {t:8.2f} ms/layer")
    T_H2D_W8A8, T_H2D_MXFP4 = 345.0, 170.0
    print(f"  vs H2D W8A8 345ms : {'PASS' if t < T_H2D_W8A8 else 'FAIL'}")
    print(f"  vs ~150 target    : {'PASS' if t < 150 else 'FAIL'}")
    print(f"  vs MXFP4 H2D ~170 : {'hidden' if t < T_H2D_MXFP4 else 'EXPOSED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--correct-experts", type=int, default=4)
    ap.add_argument("--rows-per-prog", type=int, default=8)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-correct", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    torch.npu.set_device(0)
    if not args.skip_correct:
        correctness(args.model_dir, args.layer_idx, args.correct_experts, args.rows_per_prog)
    if not args.skip_timing:
        timing(args.iters, args.rows_per_prog)


if __name__ == "__main__":
    main()
