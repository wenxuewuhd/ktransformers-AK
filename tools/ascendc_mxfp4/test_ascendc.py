#!/usr/bin/env python3
"""Correctness + timing for the AscendC MXFP4->int8 kernel vs the verified CPU reference."""
import ctypes, sys, time, statistics
from pathlib import Path
import numpy as np
import torch, torch_npu

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))                 # tools/
sys.path.insert(0, str(_HERE.parents[1] / "longseq_dbg"))
from safetensors import safe_open
from verify_mxfp4_layer import dequant_native
from mxfp4_conv_vectorized_npu import _load_weight_map, _open_shard, _as_u8, quant_per_outchannel_cpu

DEV = "npu"
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
LIB = str(_HERE.parent / "libmxfp4dq.so")


def build_consts(HALF, NB, IN):
    b = np.arange(256, dtype=np.int64)
    lutLo = FP4[b & 0xF].astype(np.float32)
    lutHi = FP4[(b >> 4) & 0xF].astype(np.float32)
    lutE8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)   # 2^(b-127)
    j = np.arange(HALF, dtype=np.int64)
    scOff = ((j >> 4) * 4).astype(np.int32)                # (j/16)*4 bytes into scF f32
    i = np.arange(IN, dtype=np.int64)
    gidx = (((i & 1) * HALF + (i >> 1)) * 4).astype(np.int32)   # comb f32 byte offset (*4)
    to = lambda a: torch.from_numpy(a).to(DEV)
    return to(lutLo), to(lutHi), to(lutE8), to(scOff), to(gidx)


def load_proj(md, wm, cache, layer, experts, proj):
    cs, ss = [], []
    for e in experts:
        wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
        sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
        h = _open_shard(md, wm, cache, wk)
        cs.append(_as_u8(h.get_tensor(wk)))
        ss.append(_as_u8(h.get_tensor(sk)))
    return torch.stack(cs), torch.stack(ss)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--proj", default="w1")
    ap.add_argument("--blockdim", type=int, default=40)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    torch.npu.set_device(0)

    lib = ctypes.CDLL(LIB)
    lib.launch_mxfp4_dq.restype = None
    lib.launch_mxfp4_dq.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 4
    stream = torch.npu.current_stream().npu_stream

    md = Path(args.model_dir)
    wm = _load_weight_map(md)
    cache = {}
    experts = list(range(args.experts))
    codes, scl = load_proj(md, wm, cache, args.layer, experts, args.proj)  # [E,OUT,HALF],[E,OUT,NB]
    E, OUT, HALF = codes.shape
    NB = scl.shape[2]
    IN = HALF * 2
    R = E * OUT
    lutLo, lutHi, lutE8, scOff, gidx = build_consts(HALF, NB, IN)

    cd = codes.reshape(R, HALF).contiguous().to(DEV)
    sd = scl.reshape(R, NB).contiguous().to(DEV)
    out = torch.empty((R, IN), dtype=torch.int8, device=DEV)
    osc = torch.empty((R,), dtype=torch.float32, device=DEV)

    def launch():
        lib.launch_mxfp4_dq(
            ctypes.c_void_p(stream), args.blockdim,
            ctypes.c_void_p(cd.data_ptr()), ctypes.c_void_p(sd.data_ptr()),
            ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(osc.data_ptr()),
            ctypes.c_void_p(lutLo.data_ptr()), ctypes.c_void_p(lutHi.data_ptr()),
            ctypes.c_void_p(lutE8.data_ptr()), ctypes.c_void_p(scOff.data_ptr()),
            ctypes.c_void_p(gidx.data_ptr()), R, HALF, NB, IN)

    launch()
    torch.npu.synchronize()

    # reference
    cd_h = cd.cpu().numpy()
    sd_h = sd.cpu().numpy()
    cpu_q, cpu_s = [], []
    for i in range(R):
        deq = dequant_native(cd_h[i][None, :], sd_h[i][None, :])   # [1,IN] f32
        q, s = quant_per_outchannel_cpu(deq)
        cpu_q.append(q[0]); cpu_s.append(s[0])
    cpu_q = torch.stack(cpu_q); cpu_s = torch.stack(cpu_s).float()
    # kernel stores [lo_plane | hi_plane]; interleave to consecutive-nibble order (torch post-step)
    out_planes = out.cpu()
    HALFp = IN // 2
    kq = torch.empty((R, IN), dtype=torch.int8)
    kq[:, 0::2] = out_planes[:, :HALFp]
    kq[:, 1::2] = out_planes[:, HALFp:]
    ks = osc.cpu()
    eqf = (kq.int() == cpu_q.int()).float().mean().item()
    mx = (kq.int() - cpu_q.int()).abs().max().item()
    serr = (ks - cpu_s).abs().max().item()
    print(f"[ascendc] proj={args.proj} R={R} HALF={HALF} NB={NB} IN={IN} blockdim={args.blockdim}")
    print(f"  int8 eq-frac={eqf:.4f}  max|dq|={mx}  scale|err|={serr:.2e}")
    print("  ks[:4] ", [round(v,5) for v in ks[:4].tolist()])
    print("  cpus[:4]", [round(v,5) for v in cpu_s[:4].tolist()])
    # reconstruction cosine vs true weight
    import numpy as np
    rec = (kq.float() * ks.unsqueeze(-1)).numpy()
    true = np.stack([dequant_native(cd_h[i][None,:], sd_h[i][None,:])[0] for i in range(R)])
    a=rec.reshape(-1); b=true.reshape(-1)
    cosr = float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    print(f"  reconstruction cos(kq*ks, true_weight) = {cosr:.6f}")
    print("  kq[0,:16] ", kq[0,:16].tolist())
    print("  cpuq[0,:16]", cpu_q[0,:16].tolist())

    if args.timing:
        for _ in range(3):
            launch()
        torch.npu.synchronize()
        ts = []
        for _ in range(10):
            torch.npu.synchronize(); t = time.perf_counter(); launch(); torch.npu.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        print(f"  timing: {statistics.median(ts):.2f} ms for {R} rows (proj {args.proj})")


if __name__ == "__main__":
    main()
