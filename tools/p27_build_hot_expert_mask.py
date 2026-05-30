#!/usr/bin/env python3
"""Build activation_freq .pt for --kt-expert-placement-strategy frequency.

Input: JSON lines with topk expert ids per layer (from msprof / request logs),
       or a zero tensor for prefix-equivalent smoke tests.

Output: torch.save({"activation_freq": Tensor[num_layers, num_experts]})

Example:
  python3 tools/p27_build_hot_expert_mask.py \\
    --num-layers 43 --num-experts 256 \\
    --out /tmp/activation_freq.pt \\
    --prefix 32

  python3 tools/p27_build_hot_expert_mask.py \\
    --num-layers 43 --num-experts 256 \\
    --out /tmp/activation_freq.pt \\
    --counts-json tools/npu_results_dbg/topk_counts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def _parse_counts_json(path: Path, num_layers: int, num_experts: int) -> torch.Tensor:
    raw = json.loads(path.read_text())
    freq = torch.zeros(num_layers, num_experts, dtype=torch.float32)
    if isinstance(raw, dict) and "layers" in raw:
        layers = raw["layers"]
    elif isinstance(raw, list):
        layers = raw
    else:
        raise ValueError(f"Unsupported counts JSON layout in {path}")

    for layer_idx, layer_counts in enumerate(layers):
        if layer_idx >= num_layers:
            break
        if isinstance(layer_counts, dict):
            for expert_id, count in layer_counts.items():
                eid = int(expert_id)
                if 0 <= eid < num_experts:
                    freq[layer_idx, eid] = float(count)
        elif isinstance(layer_counts, list):
            for expert_id, count in enumerate(layer_counts):
                if expert_id < num_experts:
                    freq[layer_idx, expert_id] = float(count)
    return freq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-layers", type=int, default=43)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--prefix",
        type=int,
        default=None,
        help="Synthetic freq: boost experts 0..prefix-1 on every MoE layer (smoke test).",
    )
    p.add_argument(
        "--counts-json",
        type=Path,
        default=None,
        help="JSON with per-layer expert hit counts.",
    )
    args = p.parse_args()

    if args.counts_json is not None:
        freq = _parse_counts_json(args.counts_json, args.num_layers, args.num_experts)
    elif args.prefix is not None:
        freq = torch.zeros(args.num_layers, args.num_experts, dtype=torch.float32)
        k = min(max(args.prefix, 0), args.num_experts)
        if k > 0:
            freq[:, :k] = 1.0
    else:
        raise SystemExit("Provide --counts-json or --prefix")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"activation_freq": freq}, args.out)
    print(f"Wrote {args.out} shape={tuple(freq.shape)} sum={float(freq.sum()):.1f}")


if __name__ == "__main__":
    main()
