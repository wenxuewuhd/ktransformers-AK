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


def convert_proj(codes_dev, scale_dev, IN, blockdim=40, packing="consecutive"):
    """One projection: MXFP4 codes/scale [E,OUT,*] -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale bf16 [E,OUT]).

    Chunked over experts so the transient (int8 planes + de-interleave + NZ cast) stays small —
    only the final [E,IN,OUT] NZ output is full-size (HBM-bounded like the W8A8 slot).

    packing: nibble layout of the code bytes — how the kernel's lo/hi planes map back to K-positions.
      "consecutive" (native safetensors): byte j -> Kpos 2j (lo), 2j+1 (hi)  -> interleave.
      "halfblock"   (GGUF block_mxfp4):   byte j -> Kpos g*32+jl (lo), +16 (hi) within its 32-group
                                           -> per-group [lo0..15 | hi0..15] concat.
    Both decode the SAME K-ordered weights bit-for-bit; the kernel and scale->block mapping (scOff)
    are packing-agnostic, so only this post-step rearrange differs (no .so change)."""
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
        # The kernel is launched via a raw ctypes <<<stream>>> call that torch does not stream-order
        # against the post-step ops reading `out`/`osc` below; without this sync the next chunk's
        # caching-allocator reuse of `out` can race a still-running kernel (intermittent garbage,
        # cos~0.1 — fires cold-start). One sync/chunk; negligible on the streaming path (chunk kernels
        # were already serial). Applies to both packings.
        torch.npu.synchronize()
        # De-interleave the [lo|hi] planes (contiguous stack) then transpose OUT<->IN. The old depool
        # hot spot was (a) a strided 1-byte de-interleave scatter (~2.4s/layer) and (b) an int8
        # transpose that degenerates to a 1-byte gather (~0.6s, ~20GB/s). (a) is gone via the
        # contiguous stack; (b) is killed by transposing in fp16 (vectorized) and round-tripping
        # int8->fp16->int8 — exact because |q|<=127. Net post-step ~3s -> ~0.13s. The .contiguous()
        # is mandatory: feeding a transposed view to format_cast lays down WRONG NZ bytes on device
        # (looks fine via .cpu() which de-formats, but grouped_matmul reads garbage).
        lo, hi = out[:, :HALFp], out[:, HALFp:]
        if packing == "halfblock":
            nb = HALFp // 16
            q = torch.cat([lo.reshape(Rc, nb, 16), hi.reshape(Rc, nb, 16)], dim=2).reshape(Ec, OUT, IN)
        else:
            q = torch.stack([lo, hi], dim=2).reshape(Ec, OUT, IN)  # consecutive interleave [E,OUT,IN]
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)           # [E,IN,OUT]
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        # Second sync: let the osc read (oscale assignment above) finish before the next chunk's
        # `osc = torch.empty(...)` reuses that buffer and the next kernel (raw ctypes, unordered)
        # overwrites it. Without it oscale races (w13 above is already drained by its post-step).
        torch.npu.synchronize()
        del out, q, nd, nz, osc, cd, sd
    return out_nz, oscale


def mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=40, packing="consecutive"):
    """Full layer depool conversion -> (w13_nz, s13b, w2_nz, s2b), the exact tensors the streaming
    slot + npu_fused_experts consume (replacing the resident W8A8 pool). packing: see convert_proj
    ("consecutive" for native safetensors codes, "halfblock" for GGUF block_mxfp4 codes)."""
    w13_nz, s13b = convert_proj(c13, s13, H, blockdim, packing)
    w2_nz, s2b = convert_proj(c2, s2, I, blockdim, packing)
    return w13_nz, s13b, w2_nz, s2b
