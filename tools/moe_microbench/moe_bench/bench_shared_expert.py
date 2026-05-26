from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_bench.config import add_common_bench_args, repo_root, resolve_cfg, timing_payload
from moe_bench.init_npu import init_custom_ops, log_versions, print_npu_smi, require_npu, setup_pythonpath
from moe_bench.ops_runner import run_shared_expert
from moe_bench.profile import maybe_profile
from moe_bench.sanity import sanity_check
from moe_bench.synthetic import assert_shapes, build_synthetic
from moe_bench.timing import bench_op


def _parse_shared_hw(trace):
    from moe_bench.msprof_runner import parse_op_summary, parse_op_summary_multi

    patterns = []
    for p in ("QuantBatchMatmul", "QuantMatmul", "DynamicQuant", "DynamicQuantize", "Silu", "Mul"):
        try:
            parse_op_summary(trace, p)
            patterns.append(p)
        except ValueError:
            continue
    if not patterns:
        raise ValueError("no shared expert ops in trace")
    if len(patterns) == 1:
        return parse_op_summary(trace, patterns[0])
    return parse_op_summary_multi(trace, patterns)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Shared expert dense W8A8 MLP")
    add_common_bench_args(parser)
    args = parser.parse_args(argv)
    setup_pythonpath(repo_root())
    cfg = resolve_cfg(args)
    if args.dry_run:
        from moe_bench.config import apply_overrides
        cfg = apply_overrides(cfg, device="cpu")
    t = build_synthetic(cfg, device="cpu" if args.dry_run else cfg.device)
    assert_shapes(cfg, t)
    if args.dry_run:
        print(json.dumps({"kind": "shared_expert", "shape": [cfg.num_tokens, cfg.hidden_size]}, indent=2))
        return 0
    init_custom_ops()
    dev = require_npu()
    log_versions(); print_npu_smi()

    if args.msprof:
        from moe_bench.msprof_runner import hw_payload, msprof_kwargs_from_cfg, run_with_msprof

        kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
        trace = run_with_msprof(lambda: run_shared_expert(t, cfg), "shared_expert", **kw)
        hw = _parse_shared_hw(trace)
        payload = hw_payload("shared_expert", hw, cfg=cfg, trace_dir=trace)
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    if args.sanity:
        o = run_shared_expert(t, cfg)
        rep = sanity_check("shared_expert", o)
        sp = Path(args.out).parent if args.out else Path("results")
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "sanity_shared_expert.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        if rep["has_nan"]:
            return 2
    if args.profile_dir:
        maybe_profile(args.profile_dir, "shared_expert", lambda: run_shared_expert(t, cfg))
    result = bench_op(lambda: run_shared_expert(t, cfg), cfg.warmup, cfg.repeat, device=dev)
    payload = {"kind": "shared_expert", "cfg": {"num_tokens": cfg.num_tokens, "H": cfg.hidden_size, "S": cfg.shared_intermediate_size}, "timing": timing_payload(result)}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
