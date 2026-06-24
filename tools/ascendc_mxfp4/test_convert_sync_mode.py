#!/usr/bin/env python3
"""Verify a STREAM-scoped fence still fixes the E=256 convert race (so the H2D||convert overlap
can drop the device-wide synchronize that serialises the copy stream).

With TASK_QUEUE_ENABLE=1 the raw ctypes kernel and the torch post-step race; convert_proj_blk
fences per chunk. This test runs E=256 convert under KT_CONVERT_SYNC = full / stream / none, each
repeated, and checks: (a) determinism (all repeats byte-equal), (b) stream == full (the known-good
golden). 'none' is expected to be non-deterministic (proves the race is live here).

Run: NPU_DEVICE_ID=<free> python3 tools/ascendc_mxfp4/test_convert_sync_mode.py
"""
import json
import os
import sys
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
REPEATS = 6


def main():
    assert M._TQ_SYNC, "run with TASK_QUEUE_ENABLE=1 (default) so the fence actually fires"
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    cfg = json.load(open(os.path.join(CK, "config.json")))
    H, I = cfg["hidden_size"], cfg["moe_intermediate_size"]

    r = GGUFReader(f"/workspace/models/cache/dsv4_layer{L}_mxfp4.gguf")

    def blk(name):
        t = next(t for t in r.tensors if t.name == f"blk.{L}.{name}.weight")
        return np.asarray(t.data)

    blk13_np = np.concatenate([blk("ffn_gate_exps"), blk("ffn_up_exps")], axis=1)
    blk13 = torch.from_numpy(blk13_np).to(dev)
    E = blk13.shape[0]
    print(f"E={E} blk13={tuple(blk13.shape)}  (chunk={M._NZ_CHUNK} -> {E // M._NZ_CHUNK} chunks)")

    def run(mode):
        M._CONVERT_SYNC = mode
        w13, s13 = M.convert_proj_blk(blk13, H)
        torch.npu.synchronize()
        # cheap byte-sensitive checksums (w13 is ~4.3G int8 -> avoid full-size int32/index tensors).
        # the race is a gross corruption (cos ~0.1), so a global sum + a strided positional subset
        # reliably distinguish good vs racy output.
        full = int(w13.sum(dtype=torch.int64).item())
        sub = w13[:, ::7, ::13].to(torch.int32)
        strided = int((sub * torch.arange(1, sub.shape[-1] + 1, device=dev, dtype=torch.int32)
                       ).sum(dtype=torch.int64).item())
        return (full, strided, float(s13.float().sum().item()))

    golden = None
    results = {}
    for mode in ("full", "stream", "none"):
        sigs = [run(mode) for _ in range(REPEATS)]
        det = all(s == sigs[0] for s in sigs)
        results[mode] = (det, sigs[0])
        if mode == "full":
            golden = sigs[0]
        match = (sigs[0] == golden)
        print(f"  {mode:6s}: deterministic={det}  ==full_golden={match}  sig={sigs[0]}")

    stream_det, stream_sig = results["stream"]
    none_det, _ = results["none"]
    stream_ok = stream_det and stream_sig == golden
    print()
    print(f"stream-scoped fence fixes the race: {stream_ok} "
          f"(deterministic={stream_det}, ==golden={stream_sig == golden})")
    print(f"race is live here (none non-deterministic): {not none_det}")
    # PASS = stream is a safe drop-in for full. (We don't fail on 'none' being deterministic — that
    # only means the race didn't happen to manifest this run; stream being correct is what matters.)
    print("RESULT:", "PASS" if stream_ok else "FAIL")
    sys.exit(0 if stream_ok else 1)


if __name__ == "__main__":
    main()
