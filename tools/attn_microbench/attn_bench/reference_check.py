"""Small-case numeric checks (B4 layout + indexer topk sanity)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import math

import torch

from attn_bench.bench_common import prepare_npu
from attn_bench.config import apply_overrides, load_config, repo_root
from attn_bench.init_npu import setup_pythonpath
from attn_bench.metadata import build_metadata
from attn_bench.ops_runner import run_csa_attn, run_csa_indexer
from attn_bench.page_table import compute_page_spec
from attn_bench.synthetic import assert_shapes, build_synthetic, build_topk_random


def _brute_indexer_topk(t, cfg, *, c4_cols: int) -> torch.Tensor:
    """CPU reference: dequant int8 key, relu dot with query*weights, top-k (causal)."""
    q = t.li_query[0].float().cpu()  # [H, D]
    w = t.li_weights[0].float().cpu()  # [H]
    scale = t.li_key_scale.squeeze(-2).float().cpu()  # [P, page_size, 1]
    key = t.li_key.float().cpu() * scale  # [P, page_size, 1, D]
    key_flat = key.reshape(-1, cfg.index_head_dim)[:c4_cols]  # [c4_cols, D]
    scores = (q.unsqueeze(1) * key_flat.unsqueeze(0)).relu()  # [H, c4, D]
    scores = (scores * w.view(-1, 1, 1)).sum(dim=(0, 2))  # [c4]
    # causal: position 0 may attend to indices [0 .. c4_cols-1] all valid for decode
    k = min(cfg.index_topk, c4_cols)
    topk = scores.topk(k).indices.to(torch.int32)
    if k < cfg.index_topk:
        pad = torch.full((cfg.index_topk - k,), -1, dtype=torch.int32)
        topk = torch.cat([topk, pad], dim=0)
    return topk.unsqueeze(0)


def _build_cmp_kv_variant(t, cfg, *, entries_per_page: int):
    """B4: entries_per_page=128 (current) vs 32 (lcm domain)."""
    spec = t.page_spec
    num_pages = max(1, math.ceil(spec.c4_cols / entries_per_page))
    dev = t.q.device
    dtype = t.cmp_kv_c4.dtype
    cmp_kv = torch.randn(
        (num_pages, entries_per_page, cfg.num_heads_kv, cfg.head_dim),
        dtype=dtype,
        device=dev,
    )
    pos = torch.arange(spec.c4_cols, device=dev, dtype=torch.int32)
    page_table = ((pos // entries_per_page) % num_pages).unsqueeze(0)
    return cmp_kv, page_table, num_pages, entries_per_page


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Small-case reference / B4 layout check")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    setup_pythonpath(repo_root())
    cfg = apply_overrides(load_config(), seq_len=args.seq_len, seed=args.seed, repeat=1, warmup=1)
    t = build_synthetic(cfg, swa_no_sink=False)
    assert_shapes(cfg, t)
    prepare_npu(argparse.Namespace(dry_run=False))
    meta = build_metadata(cfg, t)

    report: dict = {"seq_len": cfg.seq_len, "c4_cols": t.page_spec.c4_cols, "checks": {}}

    # Indexer topk overlap (approx — NPU quant path differs from fp32 brute)
    npu_topk = run_csa_indexer(t, meta, cfg).cpu()
    ref_topk = _brute_indexer_topk(t, cfg, c4_cols=t.page_spec.c4_cols)
    overlap = (npu_topk == ref_topk).float().mean().item()
    report["checks"]["indexer_topk_exact_match_ratio"] = overlap
    report["checks"]["indexer_topk_note"] = (
        "NPU uses int8+quant; low match expected; sudden 0.0 flags layout bug"
    )

    # B4: attn with entries_per_page 128 vs 32 (same c4_cols logical span)
    gen = torch.Generator(device=t.q.device)
    gen.manual_seed(cfg.seed)
    topk = build_topk_random(cfg, t.page_spec.c4_cols, t.q.device, gen)

    outs = {}
    for label, epp in (("page128", 128), ("page32", 32)):
        cmp_kv, c4_pt, n_pages, epp_used = _build_cmp_kv_variant(t, cfg, entries_per_page=epp)
        t2 = replace(t, cmp_kv_c4=cmp_kv, c4_page_table=c4_pt)
        try:
            out = run_csa_attn(t2, meta, cfg, topk)
            outs[label] = {
                "cmp_kv_shape": list(cmp_kv.shape),
                "c4_page_table_shape": list(c4_pt.shape),
                "out_abs_max": float(out.abs().max()),
                "out_has_nan": bool(torch.isnan(out).any()),
            }
        except Exception as exc:
            outs[label] = {"error": str(exc)}

    report["checks"]["b4_layout_attn"] = outs
    report["checks"]["b4_note"] = (
        "page128=current microbench; page32=lcm(4,128) domain entries/page. "
        "Both run => kernel accepts both; compare server buffer shape for ground truth."
    )

    text = json.dumps(report, indent=2)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
