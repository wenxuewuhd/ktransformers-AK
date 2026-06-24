#!/usr/bin/env python3
"""Feasibility test for overlapping the raw-block H2D with the on-device convert.

Today the streaming prefill does, per layer, H2D(raw blocks) -> convert, sequentially on one
stream (stage ~7s + convert ~13s over 43 layers). The H2D uses the SDMA copy engine and the
convert uses AICore -> they can run concurrently on separate streams. With device-side
double-buffering, H2D(L+1) overlaps convert(L) -> wall-clock ~= max(sum_H2D, sum_convert)
instead of the sum. This test proves: (1) the overlapped pipeline is byte-identical to the
sequential one, and (2) it is actually faster.

Cost of the overlap = 2x device raw-block buffers resident (the OOM-risk-at-long-context
tradeoff). Run: NPU_DEVICE_ID=<free> python3 tools/ascendc_mxfp4/test_overlap_feasibility.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "third_party" / "llama.cpp" / "gguf-py"))
import torch_npu  # noqa: E402,F401
from gguf import GGUFReader  # noqa: E402
import mxfp4_fused_op as M  # noqa: E402

CK = "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"
NLAYERS = 8          # enough to amortise the pipeline fill/drain
START = 10


def _layer_blocks(L):
    r = GGUFReader(f"/workspace/models/cache/dsv4_layer{L}_mxfp4.gguf")

    def blk(name):
        t = next(t for t in r.tensors if t.name == f"blk.{L}.{name}.weight")
        return np.asarray(t.data)

    gate, up, down = blk("ffn_gate_exps"), blk("ffn_up_exps"), blk("ffn_down_exps")
    blk13 = np.concatenate([gate, up], axis=1)         # [E, 2I, nbH*17]
    return blk13, np.ascontiguousarray(down)           # blk2 = [E, H, nbI*17]


def main():
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    cfg = json.load(open(os.path.join(CK, "config.json")))
    H, I = cfg["hidden_size"], cfg["moe_intermediate_size"]

    # host (pinned) raw blocks for NLAYERS layers
    print(f"loading {NLAYERS} layers' raw blocks (pinned)...", flush=True)
    pin13, pin2 = [], []
    for k in range(NLAYERS):
        b13, b2 = _layer_blocks(START + k)
        t13 = torch.empty(b13.shape, dtype=torch.uint8, pin_memory=True)
        t13.copy_(torch.from_numpy(b13))
        t2 = torch.empty(b2.shape, dtype=torch.uint8, pin_memory=True)
        t2.copy_(torch.from_numpy(b2))
        pin13.append(t13)
        pin2.append(t2)
    sh13, sh2 = pin13[0].shape, pin2[0].shape
    gb = (pin13[0].nbytes + pin2[0].nbytes) / 1e9
    print(f"  blk13={tuple(sh13)} blk2={tuple(sh2)}  {gb:.2f} GB/layer", flush=True)

    def make_slots():
        # let the first convert allocate; reuse thereafter
        return [None, None]

    # ---- sequential baseline (one stream) ----
    def run_sequential(capture):
        s13, s2 = make_slots()
        d13 = torch.empty(sh13, dtype=torch.uint8, device=dev)
        d2 = torch.empty(sh2, dtype=torch.uint8, device=dev)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for L in range(NLAYERS):
            d13.copy_(pin13[L], non_blocking=True)
            d2.copy_(pin2[L], non_blocking=True)
            w13, _, w2, _ = M.mxfp4_layer_to_nz_slots_blk(d13, d2, H, I, out_w13=s13, out_w2=s2)
            s13, s2 = w13, w2
            if capture is not None:
                capture.append((w13.cpu(), w2.cpu()))
        torch.npu.synchronize()
        return time.perf_counter() - t0

    # ---- overlapped pipeline (copy stream + device ping-pong + events) ----
    def run_overlap(capture):
        s13, s2 = make_slots()
        buf13 = [torch.empty(sh13, dtype=torch.uint8, device=dev) for _ in range(2)]
        buf2 = [torch.empty(sh2, dtype=torch.uint8, device=dev) for _ in range(2)]
        cpy = torch.npu.Stream()
        main = torch.npu.current_stream()
        ev_h2d = [torch.npu.Event() for _ in range(2)]
        ev_conv = [torch.npu.Event() for _ in range(2)]
        torch.npu.synchronize()
        t0 = time.perf_counter()
        # prologue: H2D layer 0 into buf[0]
        with torch.npu.stream(cpy):
            buf13[0].copy_(pin13[0], non_blocking=True)
            buf2[0].copy_(pin2[0], non_blocking=True)
            ev_h2d[0].record(cpy)
        for L in range(NLAYERS):
            par = L % 2
            nxt = 1 - par
            if L + 1 < NLAYERS:
                # don't overwrite buf[nxt] until the convert that last read it has finished
                if L >= 1:
                    cpy.wait_event(ev_conv[nxt])
                with torch.npu.stream(cpy):
                    buf13[nxt].copy_(pin13[L + 1], non_blocking=True)
                    buf2[nxt].copy_(pin2[L + 1], non_blocking=True)
                    ev_h2d[nxt].record(cpy)
            main.wait_event(ev_h2d[par])
            w13, _, w2, _ = M.mxfp4_layer_to_nz_slots_blk(
                buf13[par], buf2[par], H, I, out_w13=s13, out_w2=s2)
            s13, s2 = w13, w2
            ev_conv[par].record(main)
            if capture is not None:
                capture.append((w13.cpu(), w2.cpu()))
        torch.npu.synchronize()
        return time.perf_counter() - t0

    # warm (allocations, kernel load)
    run_sequential(None)
    run_overlap(None)

    # correctness: capture both, compare byte-equal
    seq_cap, ovl_cap = [], []
    run_sequential(seq_cap)
    run_overlap(ovl_cap)
    ok = True
    for L in range(NLAYERS):
        e13 = torch.equal(seq_cap[L][0], ovl_cap[L][0])
        e2 = torch.equal(seq_cap[L][1], ovl_cap[L][1])
        ok = ok and e13 and e2
        if not (e13 and e2):
            print(f"  MISMATCH layer {L}: w13={e13} w2={e2}")
    print(f"correctness: overlapped == sequential byte-equal: {ok}")

    # timing (median of a few)
    seq = min(run_sequential(None) for _ in range(3))
    ovl = min(run_overlap(None) for _ in range(3))
    print(f"sequential: {seq * 1000:.0f} ms ({NLAYERS} layers, {seq / NLAYERS * 1000:.0f} ms/layer)")
    print(f"overlapped: {ovl * 1000:.0f} ms ({NLAYERS} layers, {ovl / NLAYERS * 1000:.0f} ms/layer)")
    print(f"speedup: {seq / ovl:.2f}x   (extrapolated 43-layer prefill stage+convert: "
          f"{seq / NLAYERS * 43:.1f}s -> {ovl / NLAYERS * 43:.1f}s)")
    print("RESULT:", "PASS" if ok and ovl < seq else "FAIL")
    sys.exit(0 if (ok and ovl < seq) else 1)


if __name__ == "__main__":
    main()
