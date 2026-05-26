from __future__ import annotations

import argparse
import json
import sys

import torch

from attn_bench.bench_common import (
    add_common_args,
    load_cfg_from_args,
    prepare_npu,
    run_sanity,
    write_payload,
)
from attn_bench.config import repo_root
from attn_bench.init_npu import setup_pythonpath
from attn_bench.metadata import build_metadata
from attn_bench.msprof_runner import msprof_kwargs_from_cfg, parse_op_summary, run_with_msprof
from attn_bench.ops_runner import run_csa_attn, run_csa_indexer
from attn_bench.synthetic import assert_shapes, build_synthetic, build_topk_random
from attn_bench.timing import bench_op


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CSA (c4+indexer) synthetic microbench")
    add_common_args(p)
    p.add_argument(
        "--skip-indexer",
        action="store_true",
        help="Use random topk; only benchmark attn segment",
    )
    args = p.parse_args(argv)

    setup_pythonpath(repo_root())
    cfg = load_cfg_from_args(args)
    # CSA/HCA layers use attention sinks (zeros).
    t = build_synthetic(cfg, swa_no_sink=False)
    assert_shapes(cfg, t)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "kind": "csa",
                    "seq_len": cfg.seq_len,
                    "batch_size": cfg.batch_size,
                    "c4_page_table": list(t.c4_page_table.shape),
                    "c4_page_max": int(t.c4_page_table.max()),
                    "cmp_kv_c4": list(t.cmp_kv_c4.shape),
                    "li_key": list(t.li_key.shape),
                    "li_key_dtype": str(t.li_key.dtype),
                },
                indent=2,
            )
        )
        return 0

    prepare_npu(args)
    meta = build_metadata(cfg, t)

    if args.msprof:
        kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
        trace_idx = run_with_msprof(
            lambda: run_csa_indexer(t, meta, cfg),
            name=f"csa_indexer_seq{cfg.seq_len}",
            **kw,
        )
        indexer_hw = parse_op_summary(trace_idx, "QuantLightningIndexer")
        topk_static = run_csa_indexer(t, meta, cfg)
        trace_attn = run_with_msprof(
            lambda: run_csa_attn(t, meta, cfg, topk_static),
            name=f"csa_attn_seq{cfg.seq_len}",
            **kw,
        )
        attn_hw = parse_op_summary(trace_attn, "SparseAttnSharedkv")
        payload = {
            "kind": "csa",
            "mode": "msprof_hardware_only",
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "indexer_hw": indexer_hw,
            "attn_hw": attn_hw,
            "trace_dirs": [str(trace_idx), str(trace_attn)],
        }
        write_payload(payload, args.out)
        return 0

    if args.skip_indexer:
        gen = torch.Generator(device=t.q.device)
        gen.manual_seed(cfg.seed)
        topk = build_topk_random(cfg, t.page_spec.c4_cols, t.q.device, gen)
        indexer_result = None
    else:
        if args.sanity:
            run_sanity("csa_indexer", lambda: run_csa_indexer(t, meta, cfg))
        indexer_result = bench_op(
            lambda: run_csa_indexer(t, meta, cfg), cfg.warmup, cfg.repeat
        )
        topk = run_csa_indexer(t, meta, cfg)

    if args.sanity:
        run_sanity("csa_attn", lambda: run_csa_attn(t, meta, cfg, topk))

    attn_result = bench_op(
        lambda: run_csa_attn(t, meta, cfg, topk), cfg.warmup, cfg.repeat
    )

    payload = {
        "kind": "csa",
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "skip_indexer": args.skip_indexer,
        "attn_us": attn_result.to_dict(),
    }
    if indexer_result is not None:
        payload["indexer_us"] = indexer_result.to_dict()
        payload["isolated_device_sum_us"] = {
            "device_mean_us": indexer_result.device_mean_us + attn_result.device_mean_us,
            "device_std_us": (
                (indexer_result.device_std_us**2 + attn_result.device_std_us**2) ** 0.5
            ),
            "host_mean_us": indexer_result.host_mean_us + attn_result.host_mean_us,
            "mean_us": indexer_result.mean_us + attn_result.mean_us,
            "note": "NOT end-to-end; sum of independently timed indexer + attn",
        }

    write_payload(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
