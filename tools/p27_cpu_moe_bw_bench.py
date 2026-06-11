#!/usr/bin/env python3
"""CPU MoE decode bandwidth / thread-sweep micro-benchmark.

Isolates a single DSv4-Flash MoE layer and times the **decode** path
(``forward_one``, qlen=1) through the handoff-blessed *synchronous* compute path
(``copy_inputs_to_cpu_buffers`` -> ``run_pinned_forward_sync`` ->
``copy_forward_output_to_device``). No NPU stream callback, so compute is
guaranteed to run (unlike PATH_A which silently no-ops in isolation).

Why this exists: graph decode is ~70% CPU MoE, and CPU MoE is memory-bound but
only reaches ~13% of DDR peak (~95 GB/s @ 96 threads). This tool reproduces the
bandwidth-vs-threads curve and locates the >=128-thread thrash cliff, with a
hard correctness gate (output norm > 0 + cross-thread-count consistency).

cpu_infer thread config is a process-global singleton, so sweep = one process
per ``--cpuinfer`` value (driver loops over it).

Usage (single point)::

  PYTHONPATH=<worktree>/kt-kernel:<main>/third_party/sglang/python \\
  KT_DECODE_TIMING=1 ASCEND_RT_VISIBLE_DEVICES=6 \\
  python3.11 tools/p27_cpu_moe_bw_bench.py \\
    --gguf /workspace/models/cache/dsv4_layer3.gguf \\
    --layer-idx 3 --cpuinfer 96 --threadpool 8 --iters 200 --warmup 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401  (registers host pin_memory allocator)
except ImportError:
    pass


# Q8_0 = 1.0625 bytes/element, MXFP4 = 0.53125; one expert = gate[I,H]+up[I,H]+down[H,I]
def _bytes_per_expert(hidden: int, inter: int, bytes_per_elem: float) -> float:
    elems = 2 * inter * hidden + hidden * inter  # gate + up + down
    return elems * bytes_per_elem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, default=3)
    ap.add_argument("--cpuinfer", type=int, default=96)
    ap.add_argument("--threadpool", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--k", type=int, default=6, help="experts per token (all on CPU)")
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1, help="tokens (decode=1)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--npu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--bytes-per-elem", type=float, default=1.0625,
                    help="weight bytes/element for bandwidth calc: Q8_0=1.0625, MXFP4=0.53125")
    args = ap.parse_args()

    gguf = args.gguf.expanduser().resolve()
    if not gguf.is_file():
        print(f"ERROR: gguf not found: {gguf}", file=sys.stderr)
        return 2

    if not torch.npu.is_available():
        print("ERROR: torch.npu not available (need a free Ascend card)", file=sys.stderr)
        return 2
    device = torch.device("npu", args.npu_id)

    from kt_kernel import KTMoEWrapper

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)  # all experts on CPU

    t_load0 = time.perf_counter()
    wrapper = KTMoEWrapper(
        layer_idx=args.layer_idx,
        num_experts=args.num_experts,
        num_experts_per_tok=args.k,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.cpuinfer,
        threadpool_count=args.threadpool,
        weight_path=str(gguf),
        chunked_prefill_size=8,
        method="LLAMAFILE",
        numa_nodes=None,
    )
    wrapper.load_weights()
    load_s = time.perf_counter() - t_load0
    print(f"[load] cpuinfer={args.cpuinfer} threadpool={args.threadpool} load={load_s:.1f}s", flush=True)

    H = args.hidden_size
    B = args.batch
    torch.manual_seed(args.seed)
    hidden = (torch.randn(B, H, dtype=torch.bfloat16) * 0.1).to(device)

    def make_routing(it: int):
        g = torch.Generator().manual_seed(args.seed * 100003 + it)
        ids = torch.empty(B, args.k, dtype=torch.long)
        for b in range(B):
            ids[b] = torch.randperm(args.num_experts, generator=g)[: args.k]
        w = torch.softmax(torch.randn(B, args.k, generator=g), dim=-1).float()
        return ids.to(device), w.to(device)

    def one_forward(it: int) -> tuple[float, float]:
        ids, w = make_routing(it)
        wrapper.copy_inputs_to_cpu_buffers(hidden, ids, w)
        # The D2H input copies are non_blocking on the NPU stream; the real graph
        # path orders them before the host callback. Sync here so the synchronous
        # run_pinned_forward_sync reads landed inputs (else intermittent zeros).
        torch.npu.synchronize(device)
        t0 = time.perf_counter()
        wrapper.run_pinned_forward_sync(hidden, 0)
        dt = time.perf_counter() - t0
        out = wrapper.copy_forward_output_to_device(hidden)
        torch.npu.synchronize(device)
        return dt, float(out.detach().float().norm().item())

    # ---- warmup ----
    norms = []
    for it in range(args.warmup):
        _, n = one_forward(it)
        norms.append(n)
    if max(norms) <= 0.0:
        print(
            "FAIL: output norm == 0 across all warmup iters -> CPU MoE never "
            "computed (phantom path). Aborting; timing would be meaningless.",
            file=sys.stderr,
        )
        return 1

    # ---- timed ----
    times_ms = []
    out_sig = None
    for it in range(args.warmup, args.warmup + args.iters):
        dt, n = one_forward(it)
        times_ms.append(dt * 1e3)
        if n <= 0.0:
            print(f"FAIL: zero output at iter {it}", file=sys.stderr)
            return 1
    # capture a deterministic output signature at a fixed iter for cross-run consistency
    ids, w = make_routing(10_000)
    wrapper.copy_inputs_to_cpu_buffers(hidden, ids, w)
    torch.npu.synchronize(device)
    wrapper.run_pinned_forward_sync(hidden, 0)
    sig_out = wrapper.copy_forward_output_to_device(hidden)
    torch.npu.synchronize(device)
    out_sig = [round(float(x), 5) for x in sig_out.detach().float().flatten()[:8].tolist()]

    times_ms.sort()
    med = statistics.median(times_ms)
    p10 = times_ms[len(times_ms) // 10]
    mn = times_ms[0]
    bpe = _bytes_per_expert(H, args.moe_intermediate_size, args.bytes_per_elem)
    bytes_moved = B * args.k * bpe  # worst case: all k experts on CPU
    bw_med = bytes_moved / (med / 1e3) / 1e9  # GB/s
    bw_min = bytes_moved / (mn / 1e3) / 1e9

    threads_per_numa = args.cpuinfer // args.threadpool
    print(
        f"[result] cpuinfer={args.cpuinfer} per_numa={threads_per_numa} "
        f"median={med:.3f}ms min={mn:.3f}ms p10={p10:.3f}ms "
        f"bw_median={bw_med:.1f}GB/s bw_min={bw_min:.1f}GB/s "
        f"bytes/tok={bytes_moved/1e6:.1f}MB sig={out_sig}",
        flush=True,
    )

    if args.json_out:
        rec = {
            "cpuinfer": args.cpuinfer,
            "threadpool": args.threadpool,
            "per_numa": threads_per_numa,
            "load_s": round(load_s, 1),
            "median_ms": round(med, 4),
            "min_ms": round(mn, 4),
            "p10_ms": round(p10, 4),
            "bw_median_gbs": round(bw_med, 2),
            "bw_min_gbs": round(bw_min, 2),
            "bytes_per_tok_mb": round(bytes_moved / 1e6, 1),
            "out_sig": out_sig,
            "iters": args.iters,
        }
        with open(args.json_out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
