#!/usr/bin/env python3
"""NPU parity gate for the GGUF dedup path (verification standard #1: 现转结果逐bit不变).

Builds one full layer's (E=256) W8A8-NZ slots two ways and asserts the produced NZ weight
tensors + bf16 oscales are BYTE-IDENTICAL on device:

  A (today):  native safetensors consecutive codes -> convert(packing="consecutive")
  B (dedup):  GGUF block_mxfp4  halfblock   codes -> convert(packing="halfblock")

Why a local synchronized convert instead of mxfp4_fused_op.convert_proj directly: that function
launches the AscendC kernel via a raw ctypes `<<<stream>>>` call and reuses the chunk `out`
buffer across iterations; without a per-chunk sync the kernel of chunk c+1 can race the post-step
read of chunk c (a PRE-EXISTING latent race, orthogonal to this dedup change — production tolerates
it timing-wise). We add a per-chunk synchronize here purely so the A-vs-B byte compare is
deterministic; it does not change the per-chunk math. Equivalence is independently confirmed
bit-exact through npu_fused_experts at single-chunk E=32 (see git history / commit message).

Requires one free NPU (NPU_DEVICE_ID).
"""
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "third_party" / "llama.cpp" / "gguf-py"))

import torch_npu  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from safetensors import safe_open  # noqa: E402
import mxfp4_fused_op as M  # noqa: E402

_MXFP4_CKPT = os.environ.get("KT_MXFP4_CKPT", "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
_GGUF_TMPL = os.environ.get("KT_GGUF_TEMPLATE",
                            "/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf")
_NZ = 29
_DEV = None


def _u8(t):
    return (t if t.dtype == torch.uint8 else t.view(torch.uint8)).contiguous()


def _native(idx, layer, E):
    cache = {}

    def _open(k):
        sh = idx[k]
        if sh not in cache:
            cache[sh] = safe_open(os.path.join(_MXFP4_CKPT, sh), framework="pt")
        return cache[sh]

    def stack(proj):
        cs, ss = [], []
        for e in range(E):
            wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
            sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
            cs.append(_u8(_open(wk).get_tensor(wk)))
            ss.append(_u8(_open(sk).get_tensor(sk)))
        return torch.stack(cs), torch.stack(ss)

    c1, s1 = stack("w1")
    c3, s3 = stack("w3")
    c2, s2 = stack("w2")
    return torch.cat([c1, c3], 1), torch.cat([s1, s3], 1), c2, s2


def _gguf(layer):
    r = GGUFReader(_GGUF_TMPL.format(layer_idx=layer))

    def proj(name):
        t = next(t for t in r.tensors if t.name == f"blk.{layer}.{name}.weight")
        E, N, b17 = t.data.shape
        nb = b17 // 17
        blk = torch.from_numpy(np.array(t.data)).reshape(E, N, nb, 17)
        return blk[..., 1:17].reshape(E, N, nb * 16).contiguous(), blk[..., 0].contiguous()

    cg, sg = proj("ffn_gate_exps")
    cu, su = proj("ffn_up_exps")
    cd, sd = proj("ffn_down_exps")
    return torch.cat([cg, cu], 1), torch.cat([sg, su], 1), cd, sd


def _convert_proj_synced(codes, scale, IN, packing, bd=40):
    """convert_proj with a per-chunk sync (deterministic A/B compare). Mirrors the production
    post-step exactly (incl. the packing branch); only adds synchronize()s."""
    E, OUT, HALF = codes.shape
    NB = scale.shape[2]
    HALFp = IN // 2
    ll, lh, le, so = M._consts(HALF, NB, codes.device)
    stp = torch.npu.current_stream().npu_stream
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    lib = M.get_lib()
    out_nz = None
    oscale = torch.empty((E, OUT), dtype=torch.bfloat16, device=_DEV)
    for c in range(0, E, 32):
        ce = min(c + 32, E)
        Ec = ce - c
        Rc = Ec * OUT
        cd = codes[c:ce].reshape(Rc, HALF).contiguous()
        sd = scale[c:ce].reshape(Rc, NB).contiguous()
        out = torch.empty((Rc, IN), dtype=torch.int8, device=_DEV)
        Rp = (Rc + 511) // 512 * 512
        osc = torch.empty((Rp,), dtype=torch.float32, device=_DEV)
        lib.launch_mxfp4_fused(ctypes.c_void_p(stp), bd, P(cd), P(sd), P(out), P(osc),
                               P(ll), P(lh), P(le), P(so), Rc, HALF, NB, IN)
        torch.npu.synchronize()
        lo, hi = out[:, :HALFp], out[:, HALFp:]
        if packing == "halfblock":
            nb = HALFp // 16
            q = torch.cat([lo.reshape(Rc, nb, 16), hi.reshape(Rc, nb, 16)], 2).reshape(Ec, OUT, IN)
        else:
            q = torch.stack([lo, hi], 2).reshape(Ec, OUT, IN)
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=_DEV)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        torch.npu.synchronize()
        del out, q, nd, nz, osc, cd, sd
    return out_nz, oscale


def main():
    global _DEV
    layer = int(os.environ.get("TEST_LAYER", "16"))
    _DEV = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(_DEV)
    cfg = json.load(open(os.path.join(_MXFP4_CKPT, "config.json")))
    E, H, I = (int(cfg[k]) for k in ("n_routed_experts", "hidden_size", "moe_intermediate_size"))
    idx = json.load(open(os.path.join(_MXFP4_CKPT, "model.safetensors.index.json")))["weight_map"]

    nc13, ns13, nc2, ns2 = (t.to(_DEV) for t in _native(idx, layer, E))
    gc13, gs13, gc2, gs2 = (t.to(_DEV) for t in _gguf(layer))

    ok = True
    for name, (nc, ns, gc, gs, IN) in {
        "w13": (nc13, ns13, gc13, gs13, H),
        "w2": (nc2, ns2, gc2, gs2, I),
    }.items():
        a_w, a_s = _convert_proj_synced(nc, ns, IN, "consecutive")
        b_w, b_s = _convert_proj_synced(gc, gs, IN, "halfblock")
        torch.npu.synchronize()
        w_eq = bool(torch.equal(a_w, b_w))
        s_eq = bool(torch.equal(a_s, b_s))
        ok = ok and w_eq and s_eq
        print(f"  {name}: nz{tuple(a_w.shape)} byte_eq={w_eq}  oscale byte_eq={s_eq}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
