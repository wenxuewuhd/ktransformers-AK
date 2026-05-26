from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_bench.config import add_common_bench_args, repo_root, resolve_cfg, timing_payload
from moe_bench.init_npu import init_custom_ops, log_versions, print_npu_smi, require_npu, setup_pythonpath
from moe_bench.ops_runner import make_silu_mul, run_act_quant, run_act_quant_mid, run_gemm_up
from moe_bench.profile import maybe_profile
from moe_bench.sanity import sanity_check
from moe_bench.synthetic import assert_shapes, build_synthetic
from moe_bench.timing import bench_op


def _parse_silu_hw(trace, *, fused: bool):
    from moe_bench.msprof_runner import list_op_types_in_trace, parse_op_summary, parse_op_summary_multi

    if fused:
        for pattern in ("Swiglu", "SwiGLU", "SiluMul", "DequantSwiglu"):
            try:
                return parse_op_summary(trace, pattern)
            except ValueError:
                continue
        raise ValueError(f"no fused silu op; types: {list_op_types_in_trace(trace)}")

    patterns = []
    for candidates in (
        ("Silu", "silu"),
        ("Mul", "mul"),
        ("DynamicQuant", "DynamicQuantize"),
    ):
        for p in candidates:
            try:
                parse_op_summary(trace, p)
                patterns.append(p)
                break
            except ValueError:
                continue
    if len(patterns) < 2:
        raise ValueError(f"silu sub-op patterns incomplete; saw {list_op_types_in_trace(trace)}")
    if "DynamicQuant" not in patterns and "DynamicQuantize" not in patterns:
        for p in ("DynamicQuant", "DynamicQuantize"):
            try:
                parse_op_summary(trace, p)
                patterns.append(p)
                break
            except ValueError:
                pass
    return parse_op_summary_multi(trace, patterns)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SiLU(gate)*up + mid quant")
    add_common_bench_args(parser)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--fused", action="store_true")
    grp.add_argument("--no-fused", action="store_true")
    args = parser.parse_args(argv)
    setup_pythonpath(repo_root())
    cfg = resolve_cfg(args)
    if args.dry_run:
        from moe_bench.config import apply_overrides
        cfg = apply_overrides(cfg, device="cpu")
    t = build_synthetic(cfg, device="cpu" if args.dry_run else cfg.device)
    assert_shapes(cfg, t)
    if args.dry_run:
        print(json.dumps({"kind": "silu_mul", "shape": [cfg.N, 2 * cfg.moe_intermediate_size]}, indent=2))
        return 0
    init_custom_ops()
    dev = require_npu()
    log_versions(); print_npu_smi()
    xi, xs = run_act_quant(t, cfg)
    t.x_int8, t.x_scale = xi, xs
    gate_up = run_gemm_up(t, cfg)
    force = True if args.fused else (False if args.no_fused else None)
    silu_fn, mode = make_silu_mul(cfg, force_fused=force)

    def fn():
        mid = silu_fn(gate_up)
        run_act_quant_mid(t, cfg, mid)

    if args.msprof:
        from moe_bench.msprof_runner import hw_payload, msprof_kwargs_from_cfg, run_with_msprof

        kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
        trace = run_with_msprof(fn, f"silu_mul_{mode}", **kw)
        hw = _parse_silu_hw(trace, fused=(mode == "fused"))
        payload = hw_payload(f"silu_mul_{mode}", hw, cfg=cfg, trace_dir=trace, extra={"mode": mode})
        payload["mode"] = "msprof_hardware_only"
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    if args.sanity:
        mid = silu_fn(gate_up)
        rep = sanity_check(f"silu_mul_{mode}", mid)
        sp = Path(args.out).parent if args.out else Path("results")
        sp.mkdir(parents=True, exist_ok=True)
        (sp / f"sanity_silu_mul_{mode}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        if rep["has_nan"]:
            return 2
    if args.profile_dir:
        maybe_profile(args.profile_dir, f"silu_mul_{mode}", fn)
    result = bench_op(fn, cfg.warmup, cfg.repeat, device=dev)
    payload = {"kind": f"silu_mul_{mode}", "cfg": {"N": cfg.N, "mode": mode}, "timing": timing_payload(result)}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
