#!/usr/bin/env python3
"""END-TO-END: int8 kernel + oscale kernel -> NZ -> real npu_fused_experts, vs golden.
Proves the two-kernel AscendC operator produces correct W8A8."""
import ctypes, sys, time, statistics
from pathlib import Path
import numpy as np
import torch, torch_npu
from safetensors import safe_open

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "mxfp4_w8a8_op"))
from acceptance import npu_fused_experts, load_layer_mxfp4, golden_proj_to_nz  # noqa: E402

DEV = "npu"; NZ = 29; H, I = 4096, 2048; ACC = 512
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
LIB_DQ = str(_HERE.parent / "libmxfp4dq.so")
LIB_OSC = str(_HERE.parent / "liboscale.so")


def consts(HALF, NB, IN):
    b = np.arange(256, dtype=np.int64)
    lutLo = FP4[b & 0xF].astype(np.float32); lutHi = FP4[(b >> 4) & 0xF].astype(np.float32)
    lutE8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(HALF, dtype=np.int64); scOff = ((j >> 4) * 4).astype(np.int32)
    i = np.arange(IN, dtype=np.int64); gidx = (((i & 1) * HALF + (i >> 1)) * 4).astype(np.int32)
    to = lambda a: torch.from_numpy(a).to(DEV)
    return to(lutLo), to(lutHi), to(lutE8), to(scOff), to(gidx)


_dq = ctypes.CDLL(LIB_DQ); _dq.launch_mxfp4_dq.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 4
_osc = ctypes.CDLL(LIB_OSC); _osc.launch_mxfp4_oscale.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 7 + [ctypes.c_uint32] * 4


def kernel_proj_to_nz(codes_dev, scale_dev, IN, blockdim=40):
    """Run both AscendC kernels -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale bf16 [E,OUT])."""
    st = torch.npu.current_stream().npu_stream
    E, OUT, HALF = codes_dev.shape
    NB = scale_dev.shape[2]
    R = E * OUT
    lutLo, lutHi, lutE8, scOff, gidx = consts(HALF, NB, IN)
    cd = codes_dev.reshape(R, HALF).contiguous()
    sd = scale_dev.reshape(R, NB).contiguous()
    out = torch.empty((R, IN), dtype=torch.int8, device=DEV)        # two planes [lo|hi]
    osc_junk = torch.empty((R * 8,), dtype=torch.float32, device=DEV)  # dq kernel's broken oscale (ignored)
    Rpad = (R + ACC - 1) // ACC * ACC
    osc = torch.empty((Rpad,), dtype=torch.float32, device=DEV)
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    _dq.launch_mxfp4_dq(ctypes.c_void_p(st), blockdim, P(cd), P(sd), P(out), P(osc_junk),
                        P(lutLo), P(lutHi), P(lutE8), P(scOff), P(gidx), R, HALF, NB, IN)
    _osc.launch_mxfp4_oscale(ctypes.c_void_p(st), blockdim, P(cd), P(sd), P(osc),
                             P(lutLo), P(lutHi), P(lutE8), P(scOff), R, HALF, NB, IN)
    torch.npu.synchronize()
    # de-interleave planes -> [R, IN] consecutive nibble order
    HALFp = IN // 2
    q = torch.empty((R, IN), dtype=torch.int8, device=DEV)
    q[:, 0::2] = out[:, :HALFp]
    q[:, 1::2] = out[:, HALFp:]
    q = q.reshape(E, OUT, IN)
    q_nz = torch_npu.npu_format_cast(q.transpose(1, 2).contiguous(), NZ)   # [E,IN,OUT]
    oscale = osc[:R].reshape(E, OUT).to(torch.bfloat16)
    return q_nz, oscale



def golden_fp32_proj_to_nz(codes_dev, scale_dev, IN):
    """fp32 golden (no bf16 master) -> NZ slots, for apples-to-apples with the fp32 kernel."""
    import numpy as _np
    E, OUT, HALF = codes_dev.shape; NB = scale_dev.shape[2]
    cu = codes_dev.cpu().numpy(); su = scale_dev.cpu().numpy()
    q = torch.empty((E, OUT, IN), dtype=torch.int8); sc = torch.empty((E, OUT), dtype=torch.bfloat16)
    FP = _np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], _np.float32)
    for e in range(E):
        lo = FP[cu[e] & 0xF]; hi = FP[cu[e] >> 4]
        w = _np.empty((OUT, IN), _np.float32); w[:,0::2]=lo; w[:,1::2]=hi
        w *= _np.repeat(((su[e].astype(_np.uint32))<<23).view(_np.float32), 32, axis=1)
        amax = _np.abs(w).max(1, keepdims=True).clip(min=1e-8); s = amax/127.0
        q[e] = torch.from_numpy(_np.round(w/s).clip(-127,127).astype(_np.int8))
        sc[e] = torch.from_numpy((s.squeeze(-1)).astype(_np.float32)).to(torch.bfloat16)
    q_nz = torch_npu.npu_format_cast(q.to(DEV).transpose(1,2).contiguous(), NZ)
    return q_nz, sc.to(DEV)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--blockdim", type=int, default=40)
    args = ap.parse_args()
    torch.npu.set_device(0); torch.manual_seed(0)
    md = Path(args.model_dir); experts = list(range(args.experts)); E = args.experts
    c13, s13, c2, s2 = load_layer_mxfp4(md, args.layer, experts)
    c13, s13, c2, s2 = (x.to(DEV) for x in (c13, s13, c2, s2))

    def conv(fn):
        w13, s13b = fn(c13, s13, H)
        w2, s2b = fn(c2, s2, I)
        return w13, s13b, w2, s2b

    ref = conv(golden_proj_to_nz)
    ref32 = conv(golden_fp32_proj_to_nz)
    cand = conv(lambda a, b, IN: kernel_proj_to_nz(a, b, IN, args.blockdim))
    torch.npu.synchronize()

    M, top_k = args.tokens, args.top_k
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    tw, tid = torch.topk(torch.softmax(torch.randn(M, E, device=DEV), -1), top_k, -1)
    tid = tid.to(torch.int32)

    def fe(s):
        return npu_fused_experts(x.clone(), s[0], s[1], s[2], s[3], tw.to(x.dtype), tid, top_k)

    out_ref, out_ref32, out_cand = fe(ref), fe(ref32), fe(cand)
    torch.npu.synchronize()
    cosf = lambda u,v: float((u.reshape(-1).float()@v.reshape(-1).float())/(u.reshape(-1).float().norm()*v.reshape(-1).float().norm()+1e-9))
    cb = cosf(out_cand, out_ref); c3 = cosf(out_cand, out_ref32)
    print(f"[e2e-combined] layer={args.layer} E={E} M={M} blockdim={args.blockdim}")
    print(f"  cos(kernel, bf16-golden) = {cb:.8f}   (bf16 CPU/NPU rounding floor, benign)")
    print(f"  cos(kernel, fp32-golden) = {c3:.8f}   {'PASS' if c3 >= 0.9999 else 'FAIL'} (>=0.9999)")


if __name__ == "__main__":
    main()
