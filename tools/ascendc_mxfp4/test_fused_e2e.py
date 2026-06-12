#!/usr/bin/env python3
"""Fused kernel END-TO-END: MXFP4 -> (one kernel) -> int8+oscale -> NZ -> npu_fused_experts,
compared to the fp32 golden. Plus full-layer timing.
Run: ASCEND_RT_VISIBLE_DEVICES=<free> python3 test_fused_e2e.py --experts 32
"""
import ctypes, sys, time, statistics
from pathlib import Path
import numpy as np
import torch, torch_npu

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "mxfp4_w8a8_op"))
sys.path.insert(0, str(_HERE.parent))
from acceptance import npu_fused_experts, load_layer_mxfp4  # noqa: E402
from test_e2e_combined import golden_fp32_proj_to_nz  # noqa: E402

DEV = "npu"; NZ = 29; H, I = 4096, 2048; ACC = 512
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
_lib = ctypes.CDLL(str(_HERE.parent / "libmxfp4fused.so"))
_lib.launch_mxfp4_fused.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 4


def consts(HALF, NB):
    b = np.arange(256, dtype=np.int64)
    lo = FP4[b & 0xF].astype(np.float32); hi = FP4[(b >> 4) & 0xF].astype(np.float32)
    le = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(HALF, dtype=np.int64); so = ((j >> 4) * 4).astype(np.int32)
    t = lambda a: torch.from_numpy(a).to(DEV)
    return t(lo), t(hi), t(le), t(so)


def fused_to_nz(cdv, sdv, IN, bd=40):
    st = torch.npu.current_stream().npu_stream
    E, OUT, HALF = cdv.shape; NB = sdv.shape[2]; R = E * OUT
    ll, lh, le, so = consts(HALF, NB)
    cd = cdv.reshape(R, HALF).contiguous(); sd = sdv.reshape(R, NB).contiguous()
    out = torch.empty((R, IN), dtype=torch.int8, device=DEV)
    Rp = (R + ACC - 1) // ACC * ACC; osc = torch.empty((Rp,), dtype=torch.float32, device=DEV)
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    _lib.launch_mxfp4_fused(ctypes.c_void_p(st), bd, P(cd), P(sd), P(out), P(osc),
                            P(ll), P(lh), P(le), P(so), R, HALF, NB, IN)
    torch.npu.synchronize()
    HALFp = IN // 2; q = torch.empty((R, IN), dtype=torch.int8, device=DEV)
    q[:, 0::2] = out[:, :HALFp]; q[:, 1::2] = out[:, HALFp:]
    q = q.reshape(E, OUT, IN)
    q_nz = torch_npu.npu_format_cast(q.transpose(1, 2).contiguous(), NZ)
    return q_nz, osc[:R].reshape(E, OUT).to(torch.bfloat16)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=64); ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--blockdim", type=int, default=40)
    args = ap.parse_args()
    torch.npu.set_device(0); torch.manual_seed(0)
    md = Path(args.model_dir); experts = list(range(args.experts)); E = args.experts
    c13, s13, c2, s2 = load_layer_mxfp4(md, args.layer, experts)
    c13, s13, c2, s2 = (x.to(DEV) for x in (c13, s13, c2, s2))

    def conv(fn):
        return (*fn(c13, s13, H), *fn(c2, s2, I))

    ref = conv(golden_fp32_proj_to_nz)
    cand = conv(lambda a, b, IN: fused_to_nz(a, b, IN, args.blockdim))
    M, tk = args.tokens, args.top_k
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    tw, tid = torch.topk(torch.softmax(torch.randn(M, E, device=DEV), -1), tk, -1); tid = tid.to(torch.int32)
    fe = lambda s: npu_fused_experts(x.clone(), s[0], s[1], s[2], s[3], tw.to(x.dtype), tid, tk)
    oc, o3 = fe(cand), fe(ref)
    torch.npu.synchronize()
    cf = lambda u, v: float((u.reshape(-1).float() @ v.reshape(-1).float()) /
                            (u.reshape(-1).float().norm() * v.reshape(-1).float().norm() + 1e-9))
    cos = cf(oc, o3)
    print(f"[fused-e2e] E={E} M={M} bd={args.blockdim}")
    print(f"  cos(fused-kernel, fp32-golden) = {cos:.8f}  {'PASS' if cos >= 0.9999 else 'FAIL'} (>=0.9999)")


if __name__ == "__main__":
    main()
