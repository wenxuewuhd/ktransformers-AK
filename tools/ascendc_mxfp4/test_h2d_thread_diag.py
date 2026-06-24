#!/usr/bin/env python3
"""Diagnostic #1: is the production-overlap 4.5x slowdown caused by issuing the H2D from a WORKER
thread (cross-thread torch.npu) vs the MAIN thread?

Same H2D||convert pipeline (device ping-pong + copy stream + events, KT_CONVERT_SYNC=stream), but the
per-layer H2D enqueue is issued either from the main thread or from a 1-worker ThreadPoolExecutor.
If the worker variant is much slower, cross-thread torch.npu dispatch is the culprit -> the fix is to
issue H2D on the main thread and keep the worker CPU-only.

Run: NPU_DEVICE_ID=<free> python3 tools/ascendc_mxfp4/test_h2d_thread_diag.py
"""
import concurrent.futures
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

M._CONVERT_SYNC = "stream"     # production-overlap fence mode
CK = "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"
NLAYERS = 8
START = 10


def _layer_blocks(L):
    r = GGUFReader(f"/workspace/models/cache/dsv4_layer{L}_mxfp4.gguf")

    def blk(name):
        t = next(t for t in r.tensors if t.name == f"blk.{L}.{name}.weight")
        return np.asarray(t.data)

    return (np.concatenate([blk("ffn_gate_exps"), blk("ffn_up_exps")], axis=1),
            np.ascontiguousarray(blk("ffn_down_exps")))


def main():
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    cfg = json.load(open(os.path.join(CK, "config.json")))
    H, I = cfg["hidden_size"], cfg["moe_intermediate_size"]

    pin13, pin2 = [], []
    for k in range(NLAYERS):
        b13, b2 = _layer_blocks(START + k)
        for src, lst in ((b13, pin13), (b2, pin2)):
            t = torch.empty(src.shape, dtype=torch.uint8, pin_memory=True)
            t.copy_(torch.from_numpy(src))
            lst.append(t)
    sh13, sh2 = pin13[0].shape, pin2[0].shape

    cpy = torch.npu.Stream()
    main_s = torch.npu.current_stream()
    ev_h2d = [torch.npu.Event() for _ in range(2)]
    ev_conv = [torch.npu.Event() for _ in range(2)]
    buf13 = [torch.empty(sh13, dtype=torch.uint8, device=dev) for _ in range(2)]
    buf2 = [torch.empty(sh2, dtype=torch.uint8, device=dev) for _ in range(2)]
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def h2d(layer, par, gate_conv):
        torch.npu.set_device(dev)
        if gate_conv:
            cpy.wait_event(ev_conv[par])
        with torch.npu.stream(cpy):
            buf13[par].copy_(pin13[layer], non_blocking=True)
            buf2[par].copy_(pin2[layer], non_blocking=True)
        ev_h2d[par].record(cpy)

    def run(mode):     # mode: "seq" | "main" | "worker"
        s13 = s2 = None
        torch.npu.synchronize()
        t0 = time.perf_counter()
        if mode == "seq":
            d13 = torch.empty(sh13, dtype=torch.uint8, device=dev)
            d2 = torch.empty(sh2, dtype=torch.uint8, device=dev)
            for L in range(NLAYERS):
                d13.copy_(pin13[L], non_blocking=True)
                d2.copy_(pin2[L], non_blocking=True)
                w13, _, w2, _ = M.mxfp4_layer_to_nz_slots_blk(d13, d2, H, I, out_w13=s13, out_w2=s2)
                s13, s2 = w13, w2
        else:
            h2d(0, 0, False)                          # prologue
            for L in range(NLAYERS):
                par = L % 2
                if L + 1 < NLAYERS:
                    if mode == "main":
                        h2d(L + 1, 1 - par, L >= 1)
                    else:
                        ex.submit(h2d, L + 1, 1 - par, L >= 1).result()
                main_s.wait_event(ev_h2d[par])
                w13, _, w2, _ = M.mxfp4_layer_to_nz_slots_blk(
                    buf13[par], buf2[par], H, I, out_w13=s13, out_w2=s2)
                s13, s2 = w13, w2
                ev_conv[par].record(main_s)
        torch.npu.synchronize()
        return time.perf_counter() - t0

    for m in ("seq", "main", "worker"):
        run(m)                                        # warm
    res = {m: min(run(m) for _ in range(3)) for m in ("seq", "main", "worker")}
    print(f"sequential       : {res['seq']*1000:6.0f} ms  ({res['seq']/NLAYERS*1000:.0f} ms/layer)")
    print(f"overlap H2D@main : {res['main']*1000:6.0f} ms  "
          f"({res['seq']/res['main']:.2f}x vs seq)")
    print(f"overlap H2D@work : {res['worker']*1000:6.0f} ms  "
          f"({res['seq']/res['worker']:.2f}x vs seq)")
    print()
    print(f"worker/main slowdown: {res['worker']/res['main']:.2f}x  "
          f"-> cross-thread H2D is the slowdown cause: {res['worker'] > res['main']*1.3}")
    print("CONCLUSION:", "issue H2D on MAIN thread, worker stays CPU-only"
          if res['main'] < res['seq'] and res['worker'] > res['main'] * 1.3
          else "cross-thread not the (only) cause -- profile further")


if __name__ == "__main__":
    main()
