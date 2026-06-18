#!/usr/bin/env python3
"""Test the standalone oscale kernel vs the golden per-output-channel scale."""
import ctypes, sys, time, statistics
from pathlib import Path
import numpy as np
import torch, torch_npu

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "longseq_dbg"))
from safetensors import safe_open
from verify_mxfp4_layer import dequant_native
from mxfp4_conv_vectorized_npu import _load_weight_map, _open_shard, _as_u8, quant_per_outchannel_cpu

DEV = "npu"
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
LIB = str(_HERE.parent / "liboscale.so")
ACC = 512


def build_consts(HALF, NB):
    b = np.arange(256, dtype=np.int64)
    lutLo = FP4[b & 0xF].astype(np.float32)
    lutHi = FP4[(b >> 4) & 0xF].astype(np.float32)
    lutE8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(HALF, dtype=np.int64)
    scOff = ((j >> 4) * 4).astype(np.int32)
    to = lambda a: torch.from_numpy(a).to(DEV)
    return to(lutLo), to(lutHi), to(lutE8), to(scOff)


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
    lib.launch_mxfp4_oscale.restype = None
    lib.launch_mxfp4_oscale.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 7 + [ctypes.c_uint32] * 4
    stream = torch.npu.current_stream().npu_stream

    md = Path(args.model_dir)
    wm = _load_weight_map(md)
    cache = {}
    experts = list(range(args.experts))
    codes, scl = load_proj(md, wm, cache, args.layer, experts, args.proj)
    E, OUT, HALF = codes.shape
    NB = scl.shape[2]
    IN = HALF * 2
    R = E * OUT
    lutLo, lutHi, lutE8, scOff = build_consts(HALF, NB)

    cd = codes.reshape(R, HALF).contiguous().to(DEV)
    sd = scl.reshape(R, NB).contiguous().to(DEV)
    Rpad = (R + ACC - 1) // ACC * ACC
    osc = torch.full((Rpad,), -999.0, dtype=torch.float32, device=DEV)

    def launch():
        lib.launch_mxfp4_oscale(
            ctypes.c_void_p(stream), args.blockdim,
            ctypes.c_void_p(cd.data_ptr()), ctypes.c_void_p(sd.data_ptr()),
            ctypes.c_void_p(osc.data_ptr()),
            ctypes.c_void_p(lutLo.data_ptr()), ctypes.c_void_p(lutHi.data_ptr()),
            ctypes.c_void_p(lutE8.data_ptr()), ctypes.c_void_p(scOff.data_ptr()),
            R, HALF, NB, IN)

    launch()
    torch.npu.synchronize()

    # golden per-row scale
    cd_h, sd_h = cd.cpu().numpy(), sd.cpu().numpy()
    cpu_s = []
    for i in range(R):
        deq = dequant_native(cd_h[i][None, :], sd_h[i][None, :])
        _, s = quant_per_outchannel_cpu(deq)
        cpu_s.append(s[0])
    cpu_s = torch.stack(cpu_s).float()
    ks = osc.cpu()[:R]
    unwritten = (ks == -999.0).sum().item()
    serr = (ks - cpu_s).abs().max().item()
    rel = ((ks - cpu_s).abs() / cpu_s.clamp(min=1e-8)).max().item()
    print(f"[oscale] proj={args.proj} R={R} HALF={HALF} NB={NB} blockdim={args.blockdim}")
    print(f"  unwritten={unwritten}/{R}  scale max|err|={serr:.3e}  max rel-err={rel:.3e}  "
          f"{'PASS' if unwritten == 0 and rel < 0.02 else 'CHECK'}")
    print(f"  ks[:4]={[round(v,6) for v in ks[:4].tolist()]}  cpus[:4]={[round(v,6) for v in cpu_s[:4].tolist()]}")

    if args.timing:
        for _ in range(3):
            launch()
        torch.npu.synchronize()
        ts = []
        for _ in range(10):
            torch.npu.synchronize(); t = time.perf_counter(); launch(); torch.npu.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        print(f"  timing: {statistics.median(ts):.2f} ms for {R} rows")


if __name__ == "__main__":
    main()
