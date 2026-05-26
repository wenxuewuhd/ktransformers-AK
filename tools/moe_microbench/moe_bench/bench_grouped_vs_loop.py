from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_bench.config import add_common_bench_args, repo_root, resolve_cfg, timing_payload
from moe_bench.init_npu import init_custom_ops, log_versions, print_npu_smi, require_npu, setup_pythonpath
from moe_bench.ops_runner import (
    _unfused_silu_mul,
    make_loop_runner,
    run_act_quant,
    run_act_quant_mid,
    run_gemm_down,
    run_gemm_up,
)
from moe_bench.synthetic import assert_shapes, build_synthetic
from moe_bench.timing import bench_op


def _bench_target(t, cfg, dev, target: str):
    if target == "up":
        grouped_fn = lambda: run_gemm_up(t, cfg)
        loop_fn = make_loop_runner(t, cfg, target="up")
    else:
        grouped_fn = lambda: run_gemm_down(t, cfg)
        loop_fn = make_loop_runner(t, cfg, target="down")
    r_grouped = bench_op(grouped_fn, cfg.warmup, cfg.repeat, device=dev)
    r_loop = bench_op(loop_fn, cfg.warmup, cfg.repeat, device=dev)
    return {
        "target": target,
        "grouped": timing_payload(r_grouped),
        "loop": timing_payload(r_loop),
    }


def _msprof_grouped_loop(cfg, t, args, target: str):
    from moe_bench.msprof_runner import hw_payload, msprof_kwargs_from_cfg, parse_gemm_hw, run_with_msprof

    kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
    base = Path(kw["out_dir"])
    if target == "up":
        grouped_fn = lambda: run_gemm_up(t, cfg)
        loop_fn = make_loop_runner(t, cfg, target="up")
    else:
        grouped_fn = lambda: run_gemm_down(t, cfg)
        loop_fn = make_loop_runner(t, cfg, target="down")

    trace_g = run_with_msprof(grouped_fn, f"gvl_{target}_grouped", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    hw_g = parse_gemm_hw(trace_g, n_active=cfg.n_active_experts, active_steps=kw["active"])

    trace_l = run_with_msprof(loop_fn, f"gvl_{target}_loop", out_dir=str(base), **{k: v for k, v in kw.items() if k != "out_dir"})
    hw_l = parse_gemm_hw(trace_l, n_active=cfg.n_active_experts, active_steps=kw["active"])

    return {
        "target": target,
        "grouped_hw": hw_payload(f"grouped_{target}", hw_g, trace_dir=trace_g),
        "loop_hw": hw_payload(f"loop_{target}", hw_l, trace_dir=trace_l),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Grouped vs loop quant_matmul")
    add_common_bench_args(parser)
    parser.add_argument("--target", choices=["up", "down", "both"], default="both")
    args = parser.parse_args(argv)
    setup_pythonpath(repo_root())
    cfg = resolve_cfg(args)
    if args.dry_run:
        from moe_bench.config import apply_overrides
        cfg = apply_overrides(cfg, device="cpu")
    t = build_synthetic(cfg, device="cpu" if args.dry_run else cfg.device)
    assert_shapes(cfg, t)
    if args.dry_run:
        print(json.dumps({"kind": "grouped_vs_loop", "target": args.target}, indent=2))
        return 0
    init_custom_ops()
    dev = require_npu()
    log_versions(); print_npu_smi()
    xi, xs = run_act_quant(t, cfg)
    t.x_int8, t.x_scale = xi, xs
    up = run_gemm_up(t, cfg)
    mid = _unfused_silu_mul(up)
    mi8, ms = run_act_quant_mid(t, cfg, mid)
    t.mid_int8, t.mid_scale = mi8, ms
    targets = ["up", "down"] if args.target == "both" else [args.target]

    if args.msprof:
        results = [_msprof_grouped_loop(cfg, t, args, tg) for tg in targets]
        payload = {
            "kind": "grouped_vs_loop",
            "mode": "msprof_hardware_only",
            "cfg": {"N": cfg.N, "E_act": cfg.n_active_experts},
            "results": results,
        }
        if len(results) == 1:
            payload["grouped_hw"] = results[0]["grouped_hw"]
            payload["loop_hw"] = results[0]["loop_hw"]
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    results = [_bench_target(t, cfg, dev, tg) for tg in targets]
    payload = {"kind": "grouped_vs_loop", "cfg": {"N": cfg.N, "E_act": cfg.n_active_experts}, "results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
