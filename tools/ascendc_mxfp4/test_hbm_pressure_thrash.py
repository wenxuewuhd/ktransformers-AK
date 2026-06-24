#!/usr/bin/env python3
"""Diagnostic #2: confirm the production-overlap 4.5x slowdown is HBM-pressure caching-allocator
thrashing on the convert's per-chunk cat/empty (not the overlap itself).

Runs the real convert (convert_proj_blk: per-chunk out/q/osc empty + cat) at decreasing free-HBM
headroom (dummy allocations pin the rest). If convert time stays flat until headroom drops below a
threshold then spikes, that IS the allocator thrashing -> the fix is to keep the overlap's extra
device footprint small (chunk-level ping-pong ~0.86G) so headroom never crosses the cliff.

Run: NPU_DEVICE_ID=<free> python3 tools/ascendc_mxfp4/test_hbm_pressure_thrash.py
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
L = 16


def free_gb(dev):
    free, total = torch.npu.mem_get_info(dev)
    return free / 1e9, total / 1e9


def main():
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    cfg = json.load(open(os.path.join(CK, "config.json")))
    H, I = cfg["hidden_size"], cfg["moe_intermediate_size"]

    r = GGUFReader(f"/workspace/models/cache/dsv4_layer{L}_mxfp4.gguf")
    blk13_np = np.concatenate([
        np.asarray(next(t for t in r.tensors if t.name == f"blk.{L}.ffn_gate_exps.weight").data),
        np.asarray(next(t for t in r.tensors if t.name == f"blk.{L}.ffn_up_exps.weight").data)], axis=1)
    blk13 = torch.from_numpy(blk13_np).to(dev)         # 2.28G input
    slot = torch.empty((blk13.shape[0], H, blk13.shape[1]), dtype=torch.int8, device=dev)  # NZ-ish 4.3G

    def convert_time():
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            M.convert_proj_blk(blk13, H, out_nz=slot)
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / 3 * 1000

    convert_time()                                     # warm
    f0, tot = free_gb(dev)
    print(f"card total={tot:.0f}G, free after inputs+slot={f0:.1f}G")
    print(f"{'free_headroom_G':>16} {'convert_ms':>11} {'note':>10}")
    pin = []
    base = None
    # eat HBM down toward the cliff; leave room for convert's own transients (~1-2G)
    for target_free in (f0, 16, 10, 7, 5, 4, 3.2, 2.6, 2.2):
        if target_free < f0:
            cur, _ = free_gb(dev)
            need = (cur - target_free)
            if need > 0.3:
                pin.append(torch.empty(int(need * 1e9), dtype=torch.uint8, device=dev))
        cur, _ = free_gb(dev)
        try:
            ms = convert_time()
        except RuntimeError as e:
            print(f"{cur:16.1f} {'OOM':>11}   {str(e)[:30]}")
            break
        if base is None:
            base = ms
        note = "THRASH" if ms > base * 1.8 else "ok"
        print(f"{cur:16.1f} {ms:11.0f} {note:>10}")
    print()
    print("If convert_ms is flat at high headroom and spikes (THRASH) below some GB, that cliff is the")
    print("allocator thrash. Layer-level overlap (+6.8G) pushes server headroom past it; chunk-level")
    print("(+0.86G, actually LESS than the current 3.4G full-layer H2D) stays clear -> the fix.")


if __name__ == "__main__":
    main()
