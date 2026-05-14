#!/usr/bin/env python3
"""Print safetensors header dtypes for DeepSeek-V4-Flash sample keys (header only, no tensor load)."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


def main() -> None:
    model_path = Path(os.environ.get("MODEL_PATH", "/workspace/models/DeepSeek-V4-Flash-W8A8"))
    if not model_path.is_dir():
        print(f"MODEL_PATH not a directory: {model_path}", file=sys.stderr)
        sys.exit(1)

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Missing {index_path}", file=sys.stderr)
        sys.exit(1)

    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    # Raw HF / assembled keys use ``layers.N.attn.*`` (see ``_canonicalize_checkpoint_weight_name``).
    keys = [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wq_b.weight",
        "layers.0.attn.wo_a.weight",
        "layers.0.attn.wo_b.weight",
        "layers.0.ffn.experts.0.w1.weight",
    ]

    print(f"model_path={model_path.resolve()}")
    print("safetensors __header__ dtype + shape (layer 0 samples):\n")

    for k in keys:
        shard = weight_map.get(k)
        if not shard:
            print(f"MISSING in index.json:\t{k}")
            continue
        hdr = read_safetensors_header(model_path / shard)
        meta = hdr.get(k, {})
        dt = meta.get("dtype", "?")
        sh = meta.get("shape", "?")
        print(f"{dt}\t{sh}\t{k}")
        print(f"  (shard {shard})")


if __name__ == "__main__":
    main()
