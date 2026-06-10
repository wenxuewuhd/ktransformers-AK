#!/usr/bin/env python3
"""Analyze the prefill expert-activation histogram dumped by kt_ep_wrapper.

Reports, for the decode-resident-pool decision (handoff §3.5.C):
  - per-layer skew: what share of activations the top-K experts capture
  - how many experts are cold (never hit) — bounds how much residency can help
  - the [num_layers x K] top-K expert-id table = the proposed decode resident set

Usage: python analyze_expert_hist.py [hist.pt] [K]
"""
import sys, torch

path = sys.argv[1] if len(sys.argv) > 1 else "tools/longseq_dbg/expert_hist.pt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 32

d = torch.load(path)
counts = d["counts"].float()          # [L, E]
L, E = counts.shape
tokens = d["tokens"]
total = counts.sum()
print(f"# {path}: layers={L} experts={E} total_hits={int(total)} "
      f"tokens/layer~{int(tokens.float().mean())}")

srt = counts.sort(dim=1, descending=True)
topk_ids = srt.indices[:, :K]         # [L, K] proposed resident set
topk_cnt = srt.values[:, :K]

share_per_layer = topk_cnt.sum(1) / counts.sum(1).clamp(min=1)
cold_per_layer = (counts == 0).sum(1)
# global hit count for the *fixed-prefix* 0..K-1 set, for contrast vs dynamic top-K
prefix_share = counts[:, :K].sum() / total

print(f"# top{K}/layer activation share: mean={share_per_layer.mean():.3f} "
      f"min={share_per_layer.min():.3f} max={share_per_layer.max():.3f}")
print(f"# cold experts/layer (0 hits): mean={cold_per_layer.float().mean():.1f} "
      f"max={int(cold_per_layer.max())}  (of {E})")
print(f"# fixed-prefix[0:{K}] share={prefix_share:.3f}  vs dynamic top{K}={share_per_layer.mean():.3f} "
      f"-> dynamic gain={share_per_layer.mean()-prefix_share:+.3f}")
print(f"# even split baseline (K/E)={K/E:.3f} -> skew ratio={share_per_layer.mean()/(K/E):.2f}x")

# Save the proposed resident table for the decode loader (sub-goal 2 step 4).
out = path.replace(".pt", f"_top{K}.pt")
torch.save({"resident_expert_ids": topk_ids, "K": K, "num_experts": E}, out)
print(f"# proposed decode-resident top{K} table -> {out}  (shape {tuple(topk_ids.shape)})")
print("# sample layer0 resident ids:", topk_ids[0].tolist()[:16], "...")

# Machine-readable summary sidecar (metadata + conclusions) for durable record.
import json
summary = {
    "source_pt": path,
    "num_layers": int(L), "num_experts": int(E),
    "total_hits": int(total), "tokens_per_layer_mean": int(tokens.float().mean()),
    "K": K,
    "dynamic_topK_share": {"mean": round(share_per_layer.mean().item(), 4),
                            "min": round(share_per_layer.min().item(), 4),
                            "max": round(share_per_layer.max().item(), 4)},
    "static_prefix_share": round(prefix_share.item(), 4),
    "dynamic_gain": round((share_per_layer.mean() - prefix_share).item(), 4),
    "cold_experts_per_layer_mean": round(cold_per_layer.float().mean().item(), 2),
    "skew_ratio_vs_even": round((share_per_layer.mean() / (K / E)).item(), 2),
}
jout = path.replace(".pt", f"_summary_top{K}.json")
json.dump(summary, open(jout, "w"), indent=2)
print(f"# summary json -> {jout}")
