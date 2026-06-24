#!/usr/bin/env python3
"""Offline byte-equivalence gate for the GGUF dedup path (no NPU needed).

Proves that the per-layer MXFP4 codes+scale the depool streaming path feeds the AscendC
kernel can be read from the *already-loaded CPU GGUF* (block_mxfp4, half-block packed)
instead of a *separate pinned safetensors codes pool* (consecutive packed), byte-for-byte.

For a sample layer, for each projection (w1/w3/w2):
  native  = safetensors consecutive codes/scale  (what _load_layer_mxfp4 produces today)
  gguf    = block_mxfp4 [E,N,nb*17] -> de-interleave -> (halfblock codes, e8m0 scale)
asserts:
  1. scale bytes identical                     (convert script copies e8m0 verbatim)
  2. repack_halfblock_to_consecutive(gguf_codes) == native_codes   (lossless repack)

If this passes, feeding (gguf halfblock codes, scale) to the kernel with the half-block
post-step is bit-identical to feeding (native consecutive codes, scale) today.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "third_party" / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader  # noqa: E402
from safetensors import safe_open  # noqa: E402

_MXFP4_CKPT = os.environ.get("KT_MXFP4_CKPT", "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
_GGUF_TMPL = os.environ.get("KT_GGUF_TEMPLATE",
                            "/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf")


def _as_u8(t):
    return (t if t.dtype == torch.uint8 else t.view(torch.uint8)).contiguous().numpy()


def _repack_halfblock_to_consecutive(w_u8: np.ndarray) -> np.ndarray:
    """[N, K/2] GGUF (byte j -> Kpos j, j+16 within 32-group) -> native consecutive
    (byte i -> Kpos 2i, 2i+1). Inverse of convert_mxfp4_layer_to_gguf._repack_*."""
    N, kh = w_u8.shape
    nb = kh // 16
    w = w_u8.reshape(N, nb, 16)
    nib = np.empty((N, nb, 32), dtype=np.uint8)
    nib[..., 0:16] = w & 0x0F          # GGUF low nibble -> Kpos 0..15
    nib[..., 16:32] = (w >> 4) & 0x0F  # GGUF high nibble -> Kpos 16..31
    out = (nib[..., 0::2] | (nib[..., 1::2] << 4)).astype(np.uint8)  # consecutive
    return out.reshape(N, kh)


def _native_proj(model_dir, idx, layer, proj, E):
    """safetensors consecutive codes [E,N,K/2] + scale [E,N,K/32] for one projection."""
    cs, ss = [], []
    cache = {}
    for e in range(E):
        wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
        sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
        for k in (wk, sk):
            sh = idx[k]
            if sh not in cache:
                cache[sh] = safe_open(os.path.join(model_dir, sh), framework="pt")
        cs.append(_as_u8(cache[idx[wk]].get_tensor(wk)))
        ss.append(_as_u8(cache[idx[sk]].get_tensor(sk)))
    return np.stack(cs), np.stack(ss)


def _gguf_proj(reader, layer, proj):
    """block_mxfp4 [E,N,nb*17] -> de-interleave -> (halfblock codes [E,N,nb*16], scale [E,N,nb])."""
    name = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}[proj]
    t = next(t for t in reader.tensors if t.name == f"blk.{layer}.{name}.weight")
    E, N, b17 = t.data.shape
    nb = b17 // 17
    blocks = np.asarray(t.data).reshape(E, N, nb, 17)
    scale = blocks[..., 0].copy()                       # [E,N,nb]
    codes = blocks[..., 1:17].reshape(E, N, nb * 16).copy()  # [E,N,nb*16] halfblock
    return codes, scale


def main():
    layer = int(os.environ.get("TEST_LAYER", "16"))
    idx = json.load(open(os.path.join(_MXFP4_CKPT, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(_MXFP4_CKPT, "config.json")))
    E = int(cfg["n_routed_experts"])
    reader = GGUFReader(_GGUF_TMPL.format(layer_idx=layer))

    ok = True
    for proj in ("w1", "w3", "w2"):
        nat_c, nat_s = _native_proj(_MXFP4_CKPT, idx, layer, proj, E)
        gg_c, gg_s = _gguf_proj(reader, layer, proj)
        # per-expert repack (works on [N,K/2]); vectorize over E by flattening the expert axis
        Ec, N, kh = nat_c.shape
        rep = _repack_halfblock_to_consecutive(gg_c.reshape(Ec * N, kh)).reshape(Ec, N, kh)
        s_eq = np.array_equal(nat_s, gg_s)
        c_eq = np.array_equal(nat_c, rep)
        ok = ok and s_eq and c_eq
        print(f"  layer{layer} {proj}: codes{tuple(nat_c.shape)} scale{tuple(nat_s.shape)} "
              f"| scale_eq={s_eq} codes_eq(repacked)={c_eq}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
