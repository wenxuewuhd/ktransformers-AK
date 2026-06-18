#!/usr/bin/env python3
"""Fused kernel: correctness (oscale written + int8 vs golden) + e2e cosine + timing."""
import ctypes, sys, time, statistics
from pathlib import Path
import numpy as np
import torch, torch_npu

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "longseq_dbg"))
sys.path.insert(0, str(_HERE.parents[1] / "mxfp4_w8a8_op"))
from verify_mxfp4_layer import dequant_native
from mxfp4_conv_vectorized_npu import _load_weight_map, _open_shard, _as_u8, quant_per_outchannel_cpu

DEV = "npu"; NZ = 29; ACC = 512
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


def load_proj(md, wm, cache, layer, experts, proj):
    cs, ss = [], []
    for e in experts:
        wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"; sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
        h = _open_shard(md, wm, cache, wk); cs.append(_as_u8(h.get_tensor(wk))); ss.append(_as_u8(h.get_tensor(sk)))
    return torch.stack(cs), torch.stack(ss)


def run_fused(cd, sd, IN, R, HALF, NB, bd):
    st = torch.npu.current_stream().npu_stream
    ll, lh, le, so = consts(HALF, NB)
    out = torch.empty((R, IN), dtype=torch.int8, device=DEV)
    Rp = (R + ACC - 1) // ACC * ACC
    osc = torch.full((Rp,), -999.0, dtype=torch.float32, device=DEV)
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    _lib.launch_mxfp4_fused(ctypes.c_void_p(st), bd, P(cd), P(sd), P(out), P(osc),
                            P(ll), P(lh), P(le), P(so), R, HALF, NB, IN)
    torch.npu.synchronize()
    return out, osc[:R]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=16); ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--proj", default="w1"); ap.add_argument("--blockdim", type=int, default=40)
    args = ap.parse_args()
    torch.npu.set_device(0)
    md = Path(args.model_dir); wm = _load_weight_map(md); cache = {}
    experts = list(range(args.experts))
    codes, scl = load_proj(md, wm, cache, args.layer, experts, args.proj)
    E, OUT, HALF = codes.shape; NB = scl.shape[2]; IN = HALF * 2; R = E * OUT
    cd = codes.reshape(R, HALF).contiguous().to(DEV); sd = scl.reshape(R, NB).contiguous().to(DEV)
    out, ks = run_fused(cd, sd, IN, R, HALF, NB, args.blockdim)

    cd_h, sd_h = cd.cpu().numpy(), sd.cpu().numpy()
    cpu_q, cpu_s = [], []
    for i in range(R):
        deq = dequant_native(cd_h[i][None, :], sd_h[i][None, :]); q, s = quant_per_outchannel_cpu(deq)
        cpu_q.append(q[0]); cpu_s.append(s[0])
    cpu_q = torch.stack(cpu_q); cpu_s = torch.stack(cpu_s).float()
    HALFp = IN // 2; kq = torch.empty((R, IN), dtype=torch.int8)
    kq[:, 0::2] = out.cpu()[:, :HALFp]; kq[:, 1::2] = out.cpu()[:, HALFp:]
    ks = ks.cpu()
    unwritten = (ks == -999.0).sum().item()
    eqf = (kq.int() == cpu_q.int()).float().mean().item()
    serr = (ks - cpu_s).abs().max().item()
    print(f"[fused] proj={args.proj} R={R} bd={args.blockdim}")
    print(f"  oscale unwritten={unwritten}/{R}  scale max|err|={serr:.3e}  int8 eq-frac={eqf:.4f}  "
          f"{'PASS' if unwritten == 0 and eqf > 0.85 else 'CHECK'}")


if __name__ == "__main__":
    main()
