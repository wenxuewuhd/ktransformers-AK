"""Runtime wrapper for the fused AscendC MXFP4->W8A8 kernel, for kt_stream_prefill depool.

Builds libmxfp4fused.so on first use (bisheng), loads it via ctypes, and exposes:

  mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=40)
      -> (w13_nz, s13b, w2_nz, s2b)   # exactly the slot tensors npu_fused_experts consumes
                                       # w*_nz: FRACTAL_NZ int8 [E,IN,OUT];  s*b: bf16 [E,OUT]

Inputs are this layer's combined MXFP4 (device uint8):
  c13/s13: w13 = cat(w1,w3) codes [E,2I,H/2] + e8m0 scale [E,2I,H/32]
  c2/s2  : w2  codes [E,H,I/2] + e8m0 scale [E,H,I/32]

Reads MXFP4 once; one kernel pass per projection. Validated end-to-end (cos 0.99999976 vs fp32
golden through npu_fused_experts). See SPEC in tools/mxfp4_w8a8_op/.
"""
import ctypes
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "mxfp4_fused_kernel.cpp"
_SO = _HERE / "libmxfp4fused.so"
_NZ = 29
_ACC = 512
_FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

_lib = None
_lock = threading.Lock()
_consts_cache = {}


def _cann_home():
    return os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/cann-8.5.0")


def _build():
    cann = _cann_home()
    tk = f"{cann}/aarch64-linux/tikcpp"
    inc = [f"{tk}/tikcfw", f"{tk}/tikcfw/impl", f"{tk}/tikcfw/interface", f"{tk}/tikcfw/lib",
           f"{cann}/aarch64-linux/include"]
    cmd = ["bisheng", "-x", "asc", "--cce-aicore-arch=dav-c220", "-O2", "-std=c++17", "-fPIC",
           "-shared", *[f"-I{p}" for p in inc], str(_SRC), "-o", str(_SO),
           f"-L{cann}/aarch64-linux/lib64", "-lruntime", "-lascendcl"]
    subprocess.run(cmd, check=True, capture_output=True)


def get_lib():
    """Build (if needed) and load the fused kernel .so. Thread-safe, idempotent."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            if not _SO.exists() or _SO.stat().st_mtime < _SRC.stat().st_mtime:
                _build()
            lib = ctypes.CDLL(str(_SO))
            lib.launch_mxfp4_fused.restype = None
            lib.launch_mxfp4_fused.argtypes = (
                [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 4)
            _lib = lib
    return _lib


def _consts(HALF, NB, dev):
    key = (HALF, NB, str(dev))
    if key in _consts_cache:
        return _consts_cache[key]
    b = np.arange(256, dtype=np.int64)
    lutLo = _FP4[b & 0xF].astype(np.float32)
    lutHi = _FP4[(b >> 4) & 0xF].astype(np.float32)
    lutE8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(HALF, dtype=np.int64)
    scOff = ((j >> 4) * 4).astype(np.int32)
    out = tuple(torch.from_numpy(a).to(dev) for a in (lutLo, lutHi, lutE8, scOff))
    _consts_cache[key] = out
    return out


_NZ_CHUNK = int(os.environ.get("KT_MXFP4_NZ_CHUNK", "32"))  # experts/chunk -> bounds HBM transient


def convert_proj(codes_dev, scale_dev, IN, blockdim=40):
    """One projection: MXFP4 codes/scale [E,OUT,*] -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale bf16 [E,OUT]).

    Chunked over experts so the transient (int8 planes + de-interleave + NZ cast) stays small —
    only the final [E,IN,OUT] NZ output is full-size (HBM-bounded like the W8A8 slot)."""
    import torch_npu
    lib = get_lib()
    dev = codes_dev.device
    E, OUT, HALF = codes_dev.shape
    NB = scale_dev.shape[2]
    HALFp = IN // 2
    lutLo, lutHi, lutE8, scOff = _consts(HALF, NB, dev)
    st = torch.npu.current_stream().npu_stream
    P = lambda t: ctypes.c_void_p(t.data_ptr())

    out_nz = None
    oscale = torch.empty((E, OUT), dtype=torch.bfloat16, device=dev)
    for c in range(0, E, _NZ_CHUNK):
        ce = min(c + _NZ_CHUNK, E)
        Ec = ce - c
        Rc = Ec * OUT
        cd = codes_dev[c:ce].reshape(Rc, HALF).contiguous()
        sd = scale_dev[c:ce].reshape(Rc, NB).contiguous()
        out = torch.empty((Rc, IN), dtype=torch.int8, device=dev)   # two planes [lo|hi]
        Rp = (Rc + _ACC - 1) // _ACC * _ACC
        osc = torch.empty((Rp,), dtype=torch.float32, device=dev)
        lib.launch_mxfp4_fused(ctypes.c_void_p(st), blockdim, P(cd), P(sd), P(out), P(osc),
                               P(lutLo), P(lutHi), P(lutE8), P(scOff), Rc, HALF, NB, IN)
        q = torch.empty((Rc, IN), dtype=torch.int8, device=dev)
        q[:, 0::2] = out[:, :HALFp]
        q[:, 1::2] = out[:, HALFp:]
        nz = torch_npu.npu_format_cast(q.reshape(Ec, OUT, IN).transpose(1, 2).contiguous(), _NZ)
        if out_nz is None:
            out_nz = torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        del out, q, nz, osc, cd, sd
    return out_nz, oscale


def mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=40):
    """Full layer depool conversion -> (w13_nz, s13b, w2_nz, s2b), the exact tensors the streaming
    slot + npu_fused_experts consume (replacing the resident W8A8 pool)."""
    w13_nz, s13b = convert_proj(c13, s13, H, blockdim)
    w2_nz, s2b = convert_proj(c2, s2, I, blockdim)
    return w13_nz, s13b, w2_nz, s2b
