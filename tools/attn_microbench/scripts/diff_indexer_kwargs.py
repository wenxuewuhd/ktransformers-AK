#!/usr/bin/env python3
"""Diff production indexer dump vs microbench kwargs capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MICRO_ROOT = Path(__file__).resolve().parents[1]


def _encode(v):
    import torch

    if isinstance(v, torch.Tensor):
        out = {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "device": str(v.device),
        }
        if v.numel() == 1:
            out["value"] = v.item()
        elif v.numel() <= 64:
            out["sample"] = v.flatten().tolist()
        else:
            out["sample"] = v.flatten()[:8].tolist()
        return out
    return {"value": str(v)}


def capture_microbench_dump(out_path: Path, seq_len: int = 32768) -> None:
    from attn_bench.bench_common import prepare_npu
    from attn_bench.config import apply_overrides, load_config
    from attn_bench.init_npu import setup_pythonpath
    from attn_bench.metadata import build_metadata
    from attn_bench.ops_runner import torch_npu_dynamic_quant
    from attn_bench.synthetic import build_synthetic

    setup_pythonpath(MICRO_ROOT.parents[1])
    prepare_npu(argparse.Namespace(dry_run=False))
    cfg = apply_overrides(load_config(), seq_len=seq_len, warmup=1, repeat=1)
    t = build_synthetic(cfg, swa_no_sink=False)
    meta = build_metadata(cfg, t)
    q, q_scale = torch_npu_dynamic_quant(t.li_query)
    kwargs = {
        "query": q,
        "key": t.li_key,
        "key_dequant_scale": t.li_key_scale.squeeze(cfg.invariants["li_key_scale_squeeze_dim"]),
        "actual_seq_lengths_query": t.actual_seq_lengths_q,
        "actual_seq_lengths_key": t.seqused_kv,
        "block_table": t.c4_page_table,
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "weights": t.li_weights.to(__import__("torch").float16),
        "query_dequant_scale": q_scale.to(__import__("torch").float16),
        "cmp_ratio": 4,
        "query_quant_mode": 0,
        "key_quant_mode": 0,
        "sparse_mode": 3,
        "sparse_count": cfg.index_topk,
        "metadata": meta.li_quant,
    }
    payload = {
        "source": "microbench_ops_runner",
        "seq_len": seq_len,
        "kwargs": {k: _encode(v) for k, v in kwargs.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scalar(v: dict) -> str:
    if "value" in v:
        return str(v["value"])
    if "sample" in v and len(v.get("sample", [])) == 1:
        return str(v["sample"][0])
    return ""


def _row(name: str, prod: dict | None, micro: dict | None) -> str:
    if prod is None and micro is None:
        return f"| `{name}` | — | — | — | ⚠ missing |"
    p = prod or {}
    m = micro or {}
    ps = str(p.get("shape", p.get("value", "—")))
    ms = str(m.get("shape", m.get("value", "—")))
    pd = str(p.get("dtype", _scalar(p) or "—"))
    md = str(m.get("dtype", _scalar(m) or "—"))
    match = (
        ps == ms
        and pd == md
        and (_scalar(p) == _scalar(m) or not _scalar(p))
    )
    flag = "✅" if match else "❌"
    return f"| `{name}` | {ps} / {pd} / {_scalar(p) or '—'} | {ms} / {md} / {_scalar(m) or '—'} | {flag} |"


def write_diff(prod_path: Path, micro_path: Path, out_md: Path) -> int:
    prod = json.loads(prod_path.read_text(encoding="utf-8"))
    micro = json.loads(micro_path.read_text(encoding="utf-8"))
    pk = prod.get("kwargs", {})
    mk = micro.get("kwargs", {})
    keys = sorted(set(pk) | set(mk))
    lines = [
        "# P1.4 indexer kwargs field diff",
        "",
        f"- production: `{prod_path}`",
        f"- microbench: `{micro_path}`",
        "",
        "| field | production shape/dtype/value | microbench shape/dtype/value | match |",
        "|-------|------------------------------|------------------------------|-------|",
    ]
    mism = 0
    for k in keys:
        line = _row(k, pk.get(k), mk.get(k))
        lines.append(line)
        if "❌" in line:
            mism += 1
    lines.extend(["", f"**Mismatches: {mism}**", ""])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[diff] wrote {out_md} mismatches={mism}")
    return mism


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prod",
        default=str(MICRO_ROOT / "results" / "production_indexer_dump.json"),
    )
    p.add_argument(
        "--micro-out",
        default=str(MICRO_ROOT / "results" / "microbench_indexer_dump.json"),
    )
    p.add_argument(
        "--out",
        default=str(MICRO_ROOT / "results" / "p1_field_diff.md"),
    )
    p.add_argument("--seq-len", type=int, default=32768)
    args = p.parse_args()

    prod_path = Path(args.prod)
    if not prod_path.is_file():
        print(f"[diff][ERROR] missing production dump: {prod_path}")
        return 2

    micro_path = Path(args.micro_out)
    capture_microbench_dump(micro_path, args.seq_len)
    mism = write_diff(prod_path, micro_path, Path(args.out))
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
