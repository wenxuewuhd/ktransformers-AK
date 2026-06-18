#!/usr/bin/env python3
"""Bench the REAL NPU MoE expert operator (npu_grouped_matmul) at prefill M.

Mirrors sglang unquant.py forward_npu: per-layer routed-expert FFN =
  npu_moe_init_routing_v2 -> npu_grouped_matmul(gmm1) -> npu_swiglu -> npu_grouped_matmul(gmm2)
DSv4-Flash shapes: H=4096, I=2048, E=256 routed, top_k=6, 43 MoE layers.

Goal: per-layer NPU MoE compute time vs per-layer H2D (int8 6.4GB ~271ms @23.6GiB/s).
If compute >= H2D -> streaming is copy-bound (premise holds); if compute << H2D ->
copy-bound with idle NPU (still fine for hiding load, but less compute headroom).
"""
import os, time, sys
import torch, torch_npu

torch.npu.set_device(0)
DEV = "npu"

H = 4096          # hidden_size
I = 2048          # moe_intermediate_size
E = 256           # n_routed_experts
TOPK = 6          # num_experts_per_tok
LAYERS = 43

def make_expert_tokens(N, E):
    # uniform-ish distribution of N token-expert rows across E experts (group_list_type=1: counts)
    base = N // E
    counts = torch.full((E,), base, dtype=torch.int64)
    counts[: (N - base * E)] += 1
    return counts.to(DEV)

def bench_one(M, dtype, iters=3):
    N = M * TOPK
    # gathered hidden states (post init_routing): [N, H]
    hs = torch.randn(N, H, dtype=dtype, device=DEV)
    w13 = torch.randn(E, H, 2 * I, dtype=dtype, device=DEV) * 0.02   # gate+up
    w2 = torch.randn(E, I, H, dtype=dtype, device=DEV) * 0.02        # down
    grp = make_expert_tokens(N, E)

    def run():
        h = torch.ops.npu.npu_grouped_matmul(
            x=[hs], weight=[w13], bias=None,
            split_item=2, group_list_type=1, group_type=0,
            group_list=grp, output_dtype=dtype)[0]
        h = torch.ops.npu.npu_swiglu(h)
        h = torch.ops.npu.npu_grouped_matmul(
            x=[h], weight=[w2], bias=None,
            split_item=2, group_list_type=1, group_type=0,
            group_list=grp, output_dtype=dtype)[0]
        return h

    run(); torch.npu.synchronize()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    # FLOPs per layer: 2 gmm, each N * (in*out) * 2
    flop = (N * H * 2 * I + N * I * H) * 2
    tflops = flop / (ms / 1e3) / 1e12
    wbytes_bf16 = E * (H * 2 * I + I * H) * 2 / 1e9
    wbytes_int8 = E * (H * 2 * I + I * H) * 1 / 1e9
    del hs, w13, w2
    torch.npu.empty_cache()
    return ms, tflops, wbytes_bf16, wbytes_int8

if __name__ == "__main__":
    dt = torch.bfloat16
    print(f"# DSv4-Flash NPU MoE grouped_matmul bench  H={H} I={I} E={E} topk={TOPK}  dtype={dt}")
    print(f"# weight bytes/layer: bf16={E*(H*2*I+I*H)*2/1e9:.1f}GB  int8={E*(H*2*I+I*H)/1e9:.1f}GB")
    print(f"# H2D @23.6GiB/s: bf16={E*(H*2*I+I*H)*2/1e9/23.6*1e3:.0f}ms  int8={E*(H*2*I+I*H)/1e9/23.6*1e3:.0f}ms")
    print(f"{'M':>7} {'ms/layer':>10} {'TFLOPS':>8} {'43L_total_s':>12}")
    for M in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        try:
            ms, tf, _, _ = bench_one(M, dt)
            print(f"{M:>7} {ms:>10.2f} {tf:>8.1f} {ms*LAYERS/1e3:>12.2f}", flush=True)
        except Exception as e:
            print(f"{M:>7}  ERROR: {repr(e)[:120]}", flush=True)
            break
