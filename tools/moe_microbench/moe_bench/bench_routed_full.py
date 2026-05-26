from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_bench.config import add_common_bench_args, repo_root, resolve_cfg, timing_payload
from moe_bench.init_npu import init_custom_ops, log_versions, print_npu_smi, require_npu, setup_pythonpath
from moe_bench.ops_runner import (
    _unfused_silu_mul,
    run_act_quant,
    run_act_quant_mid,
    run_gemm_down,
    run_gemm_up,
    run_routed_post_dispatch,
)
from moe_bench.profile import maybe_profile
from moe_bench.sanity import sanity_check
from moe_bench.synthetic import assert_shapes, build_synthetic
from moe_bench.timing import bench_op


def _msprof_routed(cfg, t, args):
    from moe_bench.msprof_runner import (
        hw_payload,
        msprof_kwargs_from_cfg,
        parse_gemm_hw,
        parse_op_summary,
        parse_op_summary_multi,
        run_with_msprof,
    )

    kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
    base = Path(kw["out_dir"])

    xi, xs = run_act_quant(t, cfg)
    t.x_int8, t.x_scale = xi, xs
    up = run_gemm_up(t, cfg)
    mid = _unfused_silu_mul(up)
    mi8, ms = run_act_quant_mid(t, cfg, mid)
    t.mid_int8, t.mid_scale = mi8, ms

    segments = {}
    trace_aq = run_with_msprof(lambda: run_act_quant(t, cfg), "routed_act_quant", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    for pattern in ("DynamicQuant", "DynamicQuantize"):
        try:
            segments["act_quant_hw"] = hw_payload("act_quant", parse_op_summary(trace_aq, pattern), trace_dir=trace_aq)
            break
        except ValueError:
            continue

    trace_up = run_with_msprof(lambda: run_gemm_up(t, cfg), "routed_gemm_up", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    hw_up = parse_gemm_hw(trace_up, n_active=cfg.n_active_experts, active_steps=kw["active"])
    segments["gemm_up_hw"] = hw_payload("gemm_up", hw_up, trace_dir=trace_up)

    def silu_fn():
        m = _unfused_silu_mul(up)
        run_act_quant_mid(t, cfg, m)

    trace_silu = run_with_msprof(silu_fn, "routed_silu", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    silu_patterns = []
    for p in ("Silu", "Mul", "DynamicQuant", "DynamicQuantize"):
        try:
            parse_op_summary(trace_silu, p)
            if p not in silu_patterns:
                silu_patterns.append(p)
        except ValueError:
            pass
    segments["silu_hw"] = hw_payload("silu_mul", parse_op_summary_multi(trace_silu, silu_patterns[:3]), trace_dir=trace_silu)

    trace_dn = run_with_msprof(lambda: run_gemm_down(t, cfg), "routed_gemm_down", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    hw_dn = parse_gemm_hw(trace_dn, n_active=cfg.n_active_experts, active_steps=kw["active"])
    segments["gemm_down_hw"] = hw_payload("gemm_down", hw_dn, trace_dir=trace_dn)

    trace_pipe = run_with_msprof(lambda: run_routed_post_dispatch(t, cfg), "routed_full_pipeline", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    pipe_hw = None
    for pattern in ("GroupedMatmul", "QuantMatmul", "DynamicQuant", "Silu"):
        try:
            pipe_hw = parse_op_summary(trace_pipe, pattern)
            break
        except ValueError:
            continue
    if pipe_hw is None:
        from moe_bench.msprof_runner import list_op_types_in_trace
        types = list_op_types_in_trace(trace_pipe)
        seg_sum = sum(
            s.get("device_mean_us", 0)
            for s in segments.values()
            if isinstance(s, dict) and "device_mean_us" in s
        )
        pipe_hw = {"op_pattern": "segment_sum", "device_mean_us": seg_sum, "matched_rows": len(segments)}
        segments["pipeline_note"] = f"pipeline trace unaggregated; types={types[:5]}"

    segments["pipeline_hw"] = hw_payload("routed_full_pipeline", pipe_hw, trace_dir=trace_pipe)

    seg_total = (
        segments["act_quant_hw"]["device_mean_us"]
        + segments["gemm_up_hw"]["device_mean_us"]
        + segments["silu_hw"]["device_mean_us"]
        + segments["gemm_down_hw"]["device_mean_us"]
    )
    return {
        "kind": "routed_full",
        "mode": "msprof_hardware_only",
        "device_mean_us": seg_total,
        "pipeline_device_mean_us": segments["pipeline_hw"]["device_mean_us"],
        "segments_hw": segments,
        "cfg": {"N": cfg.N, "E_act": cfg.n_active_experts},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Routed MoE full path")
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
        print(json.dumps({"kind": "routed_full", "N": cfg.N}, indent=2))
        return 0
    init_custom_ops()
    dev = require_npu()
    log_versions(); print_npu_smi()

    if args.msprof:
        payload = _msprof_routed(cfg, t, args)
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    xi, xs = run_act_quant(t, cfg)
    t.x_int8, t.x_scale = xi, xs
    if args.sanity:
        o = run_routed_post_dispatch(t, cfg)
        rep = sanity_check("routed_full", o)
        sp = Path(args.out).parent if args.out else Path("results")
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "sanity_routed_full.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        if rep["has_nan"]:
            return 2
    if args.profile_dir:
        maybe_profile(args.profile_dir, "routed_full", lambda: run_routed_post_dispatch(t, cfg))
    from moe_bench.ops_runner import run_routed_compute_only
    r_compute_only = bench_op(lambda: run_routed_compute_only(t, cfg), cfg.warmup, cfg.repeat, device=dev)
    r_post = bench_op(lambda: run_routed_post_dispatch(t, cfg), cfg.warmup, cfg.repeat, device=dev)
    payload = {
        "kind": "routed_full",
        "cfg": {"N": cfg.N, "H": cfg.hidden_size, "I": cfg.moe_intermediate_size, "E_act": cfg.n_active_experts},
        "timing": timing_payload(r_post),
        "derived": {
            "routed_full_compute_only_us": r_compute_only.device_mean_us,
            "routed_full_post_dispatch_us": r_post.device_mean_us,
            "dispatch_overhead_us": r_post.host_mean_us - r_post.device_mean_us,
        },
        "segments": {
            "compute_only": timing_payload(r_compute_only),
            "post_dispatch": timing_payload(r_post),
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
