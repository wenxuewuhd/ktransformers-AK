#!/usr/bin/env python3
"""Diagnostic #3: is the production-overlap slowdown the worker's 32-thread CPU copy STARVING the
main thread (which must stay active to dispatch convert/attention ops)? The trace showed a GLOBAL
slowdown (cat 94s, attention 12x) -> not a device issue, a host-scheduling one.

Measures convert wall-time on the main thread while a background worker runs an N-thread parallel
pinned->pinned memcpy (mimics _fill_stage). If convert slows sharply as N rises, the 32-thread copy
is starving the main thread -> fix = fewer copy threads in the overlap path (the copy only has to
finish within the convert window, so it doesn't need 32).

Run: NPU_DEVICE_ID=<free> python3 tools/ascendc_mxfp4/test_cpu_contention.py
"""
import json
import os
import sys
import threading
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
L = 16


def main():
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    cfg = json.load(open(os.path.join(CK, "config.json")))
    H, I = cfg["hidden_size"], cfg["moe_intermediate_size"]
    torch.set_num_threads(1)                            # match server (OMP pinned low)

    r = GGUFReader(f"/workspace/models/cache/dsv4_layer{L}_mxfp4.gguf")
    blk13 = torch.from_numpy(np.concatenate([
        np.asarray(next(t for t in r.tensors if t.name == f"blk.{L}.ffn_gate_exps.weight").data),
        np.asarray(next(t for t in r.tensors if t.name == f"blk.{L}.ffn_up_exps.weight").data)],
        axis=1)).to(dev)
    slot = torch.empty((blk13.shape[0], H, blk13.shape[1]), dtype=torch.int8, device=dev)

    # dummy pinned buffers for the background copy load (~3.4G, like a layer's raw blocks)
    src = torch.empty(int(3.4e9), dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(int(3.4e9), dtype=torch.uint8, pin_memory=True)
    E = src.numel()

    stop = threading.Event()

    def load(nthreads):
        """Background: keep doing an N-thread parallel memcpy, like the prefetch worker per layer."""
        while not stop.is_set():
            ths = [threading.Thread(target=lambda lo, hi: dst[lo:hi].copy_(src[lo:hi]),
                                    args=(E * k // nthreads, E * (k + 1) // nthreads))
                   for k in range(nthreads)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()

    def convert_ms():
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            M.convert_proj_blk(blk13, H, out_nz=slot)
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / 5 * 1000

    convert_ms()                                        # warm
    base = convert_ms()
    print(f"convert (no background copy): {base:.0f} ms/layer")
    print(f"{'bg_copy_threads':>16} {'convert_ms':>11} {'slowdown':>9}")
    print(f"{0:16d} {base:11.0f} {'1.00x':>9}")
    for n in (4, 8, 16, 32):
        stop.clear()
        bg = threading.Thread(target=load, args=(n,), daemon=True)
        bg.start()
        time.sleep(0.5)
        ms = convert_ms()
        stop.set()
        bg.join(timeout=5)
        print(f"{n:16d} {ms:11.0f} {ms/base:8.2f}x")
    print()
    print("If convert_ms blows up as bg threads rise, the 32-thread worker copy starves the main")
    print("thread -> use fewer copy threads in the overlap (copy is hidden behind convert anyway).")


if __name__ == "__main__":
    main()
