from __future__ import annotations

import argparse
import json
import sys

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
from attn_bench.ops_runner import run_swa_attn
from attn_bench.synthetic import assert_shapes, build_synthetic
from attn_bench.timing import bench_op


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SWA synthetic attention microbench")
    add_common_args(p)
    p.add_argument(
        "--with-sink",
        action="store_true",
        help="Use zero sinks (default: -inf = no sink, pure SWA)",
    )
    args = p.parse_args(argv)

    setup_pythonpath(repo_root())
    cfg = load_cfg_from_args(args)
    t = build_synthetic(cfg, swa_no_sink=not args.with_sink)
    assert_shapes(cfg, t)

    if args.dry_run:
        spec = t.page_spec
        print(
            json.dumps(
                {
                    "kind": "swa",
                    "seq_len": cfg.seq_len,
                    "batch_size": cfg.batch_size,
                    "shapes": {
                        "q": list(t.q.shape),
                        "ori_kv": list(t.ori_kv.shape),
                        "swa_page_table": list(t.swa_page_table.shape),
                        "swa_page_max": int(t.swa_page_table.max()),
                        "seqused_kv": list(t.seqused_kv.shape),
                    },
                    "pages": spec.__dict__,
                },
                indent=2,
            )
        )
        return 0

    prepare_npu(args)
    meta = build_metadata(cfg, t)

    if args.msprof:
        kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
        trace = run_with_msprof(
            lambda: run_swa_attn(t, meta, cfg),
            name=f"swa_attn_seq{cfg.seq_len}",
            **kw,
        )
        attn_hw = parse_op_summary(trace, "SparseAttnSharedkv")
        payload = {
            "kind": "swa",
            "mode": "msprof_hardware_only",
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "attn_hw": attn_hw,
            "trace_dir": str(trace),
        }
        write_payload(payload, args.out)
        return 0

    if args.sanity:
        run_sanity("swa", lambda: run_swa_attn(t, meta, cfg))

    result = bench_op(lambda: run_swa_attn(t, meta, cfg), cfg.warmup, cfg.repeat)
    payload = {
        "kind": "swa",
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "attn_us": result.to_dict(),
    }
    write_payload(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
