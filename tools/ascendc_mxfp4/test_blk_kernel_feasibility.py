#!/usr/bin/env python3
"""Feasibility test for Option B: mxfp4_fused_blk reads raw GGUF block_mxfp4 and de-interleaves in
UB via Gather, vs the current mxfp4_fused (which needs host/device de-interleaved codes+scale).

Verifies: (1) byte-identical kernel output (out planes + oscale); (2) speed (the blk kernel should
fold the de-interleave in ~free, eliminating the ~1.3s/layer GM strided de-interleave).
Run: NPU_DEVICE_ID=<free> python3 tools/ascendc_mxfp4/test_blk_kernel_feasibility.py
"""
import ctypes
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
import torch_npu  # noqa: E402
from gguf import GGUFReader  # noqa: E402
import mxfp4_fused_op as M  # noqa: E402

_ACC = 512
CK = "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"
L = 16


def main():
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    cfg = json.load(open(os.path.join(CK, "config.json")))
    H, I = cfg["hidden_size"], cfg["moe_intermediate_size"]
    IN, OUT = H, 2 * I               # w13 projection
    HALF, NB = IN // 2, IN // 32
    nb = NB
    nb17 = nb * 17

    # raw blocks for w13 = cat(gate, up) along OUT
    r = GGUFReader(f"/workspace/models/cache/dsv4_layer{L}_mxfp4.gguf")

    def blk(name):
        t = next(t for t in r.tensors if t.name == f"blk.{L}.{name}.weight")
        return np.array(t.data)            # [E, N, nb*17]

    g, u = blk("ffn_gate_exps"), blk("ffn_up_exps")
    E = g.shape[0]
    EC = 32                                  # one chunk
    blocks = np.concatenate([g[:EC], u[:EC]], axis=1)          # [EC, 2I, nb*17]
    blocks_dev = torch.from_numpy(blocks).reshape(EC * OUT, nb17).contiguous().to(dev)

    # de-interleaved codes/scale (what the current kernel consumes)
    b4 = blocks.reshape(EC, OUT, nb, 17)
    codes = np.ascontiguousarray(b4[..., 1:17]).reshape(EC * OUT, HALF)
    scale = np.ascontiguousarray(b4[..., 0]).reshape(EC * OUT, NB)
    cd = torch.from_numpy(codes).contiguous().to(dev)
    sd = torch.from_numpy(scale).contiguous().to(dev)
    Rc = EC * OUT

    # consts (shared)
    lutLo, lutHi, lutE8, scOff = M._consts(HALF, NB, dev)
    # de-interleave offsets (byte offsets into the half-cast block buffer)
    j = np.arange(HALF, dtype=np.int64)
    codeOff = (((j // 16) * 17 + 1 + (j % 16)) * 2).astype(np.uint32)
    b = np.arange(NB, dtype=np.int64)
    scaleOff = ((b * 17) * 2).astype(np.uint32)
    codeOff_d = torch.from_numpy(codeOff).to(dev)
    scaleOff_d = torch.from_numpy(scaleOff).to(dev)

    lib = M.get_lib()
    lib.launch_mxfp4_fused.argtypes = (
        [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 4)
    lib.launch_mxfp4_fused_blk.restype = None
    lib.launch_mxfp4_fused_blk.argtypes = (
        [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 4)
    st = torch.npu.current_stream().npu_stream
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    Rp = (Rc + _ACC - 1) // _ACC * _ACC
    bd = 40

    def run_ref():
        out = torch.empty((Rc, IN), dtype=torch.int8, device=dev)
        osc = torch.empty((Rp,), dtype=torch.float32, device=dev)
        lib.launch_mxfp4_fused(ctypes.c_void_p(st), bd, P(cd), P(sd), P(out), P(osc),
                               P(lutLo), P(lutHi), P(lutE8), P(scOff), Rc, HALF, NB, IN)
        return out, osc

    def run_blk():
        out = torch.empty((Rc, IN), dtype=torch.int8, device=dev)
        osc = torch.empty((Rp,), dtype=torch.float32, device=dev)
        lib.launch_mxfp4_fused_blk(ctypes.c_void_p(st), bd, P(blocks_dev), P(out), P(osc),
                                   P(lutLo), P(lutHi), P(lutE8), P(scOff),
                                   P(codeOff_d), P(scaleOff_d), Rc, HALF, NB, IN)
        return out, osc

    o_ref, s_ref = run_ref(); torch.npu.synchronize()
    o_blk, s_blk = run_blk(); torch.npu.synchronize()
    out_eq = bool(torch.equal(o_ref, o_blk))
    osc_eq = bool(torch.equal(s_ref[:Rc], s_blk[:Rc]))
    print(f"  out byte_equal={out_eq}  oscale byte_equal={osc_eq}  (Rc={Rc} HALF={HALF})")

    # timing (per chunk -> x8 chunks/proj x2 proj x43 layers for a full prefill estimate)
    for name, fn in (("ref(kernel only)", run_ref), ("blk(kernel+deint)", run_blk)):
        fn(); torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {name}: {ms:.1f} ms/chunk")
    print("RESULT:", "PASS" if out_eq and osc_eq else "FAIL")
    sys.exit(0 if (out_eq and osc_eq) else 1)


if __name__ == "__main__":
    main()
