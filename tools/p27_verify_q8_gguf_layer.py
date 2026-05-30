#!/usr/bin/env python3
"""Verify per-layer Q8_0 GGUF against W8A8 dequant (layout + scale sanity).

Usage::

  PYBIN=python3.11 PYTHONPATH=.../gguf-py \\
    $PYBIN tools/p27_verify_q8_gguf_layer.py \\
      --w8a8 /workspace/models/DeepSeek-V4-Flash-W8A8 \\
      --gguf /workspace/models/cache/dsv4_layer3_q8_0.gguf \\
      --layer-idx 3 --expert-idx 38
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_REPO = Path(__file__).resolve().parents[1]
_GGUF_PY = _REPO / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(_GGUF_PY))
import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402


def _dequant_q8_0_array(arr: np.ndarray) -> np.ndarray:
    """arr: uint8 blob with shape (..., n_blocks, 34) or packed rows."""
    from gguf.quants import quant_shape_from_byte_shape

    if arr.dtype != np.uint8:
        arr = arr.view(np.uint8)
    # If last dim is 34, treat as block_q8_0 layout
    if arr.shape[-1] == 34:
        n_blocks = arr.shape[-1]
        _ = n_blocks
        flat = arr.reshape(-1, 34)
        out = np.empty((flat.shape[0], 32), dtype=np.float32)
        for i in range(flat.shape[0]):
            block = flat[i]
            d = np.float16(block[0:2].tobytes().view(np.float16)[0])
            qs = block[2:].view(np.int8)
            out[i] = qs.astype(np.float32) * float(d)
        return out.reshape(*arr.shape[:-1], 32 * arr.shape[-2] if arr.ndim > 1 else 32)
    byte_shape = arr.shape
    float_shape = quant_shape_from_byte_shape(byte_shape, GGMLQuantizationType.Q8_0)
    out = np.empty(float_shape, dtype=np.float32)
    # ggml dequantize via block iterator
    block_size = 32
    type_size = 34
    flat_in = arr.reshape(-1, type_size)
    flat_out = out.reshape(-1, block_size)
    for i in range(flat_in.shape[0]):
        b = flat_in[i]
        d = float(np.frombuffer(b[:2], dtype=np.float16)[0])
        flat_out[i] = b[2:].view(np.int8).astype(np.float32) * d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w8a8", type=Path, required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--expert-idx", type=int, default=38)
    ap.add_argument("--proj", type=str, default="w1", choices=("w1", "w3", "w2"))
    args = ap.parse_args()

    idx = json.loads((args.w8a8 / "model.safetensors.index.json").read_text())["weight_map"]
    layer = args.layer_idx
    e = args.expert_idx
    wk = f"layers.{layer}.ffn.experts.{e}.{args.proj}.weight"
    sk = wk.replace(".weight", ".weight_scale")
    with safe_open(args.w8a8 / idx[wk], framework="pt") as f:
        w_ref = (f.get_tensor(wk).float() * f.get_tensor(sk).float()).numpy()

    reader = gguf.GGUFReader(str(args.gguf))
    name_map = {"w1": "gate", "w3": "up", "w2": "down"}
    tname = f"blk.{layer}.ffn_{name_map[args.proj]}_exps.weight"
    tensor = next(t for t in reader.tensors if t.name == tname)
    raw = np.array(tensor.data, dtype=np.uint8)
    # Convert stacks (E, intermediate, hidden) then Q8_0 on hidden; stored as
    # E * intermediate rows × (hidden/32*34) bytes/row. GGUF metadata dims are
    # ggml-order (hidden, intermediate, E) = (4096, 2048, 256).
    hidden, inter, n_expert = int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])
    row_bytes = (hidden // 32) * 34
    packed = raw.reshape(n_expert, inter, row_bytes)
    if args.proj in ("w1", "w3"):
        expert_blob = packed[e]  # (intermediate, row_bytes)
        w_gguf = _dequant_q8_0_array(expert_blob)
    else:
        # down: (hidden, intermediate) with intermediate inner
        down_row_bytes = (inter // 32) * 34
        packed_down = raw.reshape(n_expert, hidden, down_row_bytes)
        expert_blob = packed_down[e]
        w_gguf = _dequant_q8_0_array(expert_blob)
    if w_gguf.shape != w_ref.shape:
        print(f"  shape mismatch gguf={w_gguf.shape} ref={w_ref.shape}")
        return 1

    finite = bool(np.isfinite(w_gguf).all())
    cos = float(
        np.dot(w_gguf.ravel(), w_ref.ravel())
        / (np.linalg.norm(w_gguf) * np.linalg.norm(w_ref) + 1e-12)
    )
    max_err = float(np.abs(w_gguf - w_ref).max())
    print(f"[verify] tensor={tname} expert={e} proj={args.proj}")
    print(f"  gguf file mtime: {args.gguf.stat().st_mtime}")
    print(f"  shape ref={w_ref.shape} gguf_dequant={w_gguf.shape}")
    print(f"  finite(gguf_dequant)={finite}  cosine={cos:.6f}  max_abs_err={max_err:.4e}")
    if not finite:
        print("  FAIL: dequant contains NaN/Inf — re-run convert_w8a8_to_gguf_q8_0.py for this layer")
        return 1
    if cos < 0.95:
        print("  WARN: cosine < 0.95 — likely old layout (permute) or quant mismatch; re-convert GGUF")
        return 1
    print("  PASS: GGUF weights look consistent with W8A8 dequant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
