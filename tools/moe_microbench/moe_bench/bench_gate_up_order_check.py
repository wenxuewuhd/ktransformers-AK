from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from moe_bench.config import load_config


def _load_expert_w1_w3(ckpt: Path, layer: int, expert: int = 0):
    import json as js
    from safetensors import safe_open

    idx = js.load(open(ckpt / "model.safetensors.index.json"))
    w1_key = f"layers.{layer}.ffn.experts.{expert}.w1.weight"
    w3_key = f"layers.{layer}.ffn.experts.{expert}.w3.weight"
    shard = ckpt / idx["weight_map"][w1_key]
    with safe_open(shard, framework="pt", device="cpu") as f:
        w1 = f.get_tensor(w1_key).float()
        w3 = f.get_tensor(w3_key).float()
    return w1, w3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify gate/up chunk order against ckpt w1/w3")
    parser.add_argument("--ckpt", type=str, default="/workspace/models/DeepSeek-V4-Flash-W8A8")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config()
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    ckpt = Path(args.ckpt)
    w1, w3 = _load_expert_w1_w3(ckpt, args.layer, args.expert)
    assert w1.shape == (I, H) and w3.shape == (I, H), (w1.shape, w3.shape)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    x = torch.randn(1, H, generator=gen)

    ref_gate = F.silu(x @ w1.T) * (x @ w3.T)
    ref_up_first = F.silu(x @ w3.T) * (x @ w1.T)

    W_gate_first = torch.cat([w1, w3], dim=0)
    W_up_first = torch.cat([w3, w1], dim=0)
    out_gate_first = F.silu(x @ W_gate_first[:I].T) * (x @ W_gate_first[I:].T)
    out_up_first = F.silu(x @ W_up_first[:I].T) * (x @ W_up_first[I:].T)

    err_gate = (out_gate_first - ref_gate).abs().mean().item()
    err_up = (out_up_first - ref_gate).abs().mean().item()
    err_ref_swap = (ref_up_first - ref_gate).abs().mean().item()

    conclusion = "gate-first" if err_gate <= err_up else "up-first"
    payload = {
        "layer": args.layer,
        "expert": args.expert,
        "w1_shape": list(w1.shape),
        "w3_shape": list(w3.shape),
        "err_gate_first_fused": err_gate,
        "err_up_first_fused": err_up,
        "err_ref_swap_separate": err_ref_swap,
        "conclusion": conclusion,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
