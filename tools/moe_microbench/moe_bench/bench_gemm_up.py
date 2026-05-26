from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_bench.config import add_common_bench_args, repo_root, resolve_cfg, timing_payload
from moe_bench.init_npu import init_custom_ops, log_versions, print_npu_smi, require_npu, setup_pythonpath
from moe_bench.ops_runner import run_act_quant, run_gemm_up
from moe_bench.profile import maybe_profile
from moe_bench.sanity import sanity_check
from moe_bench.synthetic import assert_shapes, build_synthetic
from moe_bench.timing import bench_op


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="W8A8 grouped GEMM up")
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
        print(json.dumps({"kind": "gemm_up", "N": cfg.N}, indent=2))
        return 0
    init_custom_ops()
    dev = require_npu()
    log_versions(); print_npu_smi()
    xi, xs = run_act_quant(t, cfg)
    t.x_int8, t.x_scale = xi, xs

    if args.msprof:
        from moe_bench.msprof_runner import hw_payload, msprof_kwargs_from_cfg, parse_gemm_hw, run_with_msprof

        kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
        trace = run_with_msprof(lambda: run_gemm_up(t, cfg), "gemm_up", **kw)
        hw = parse_gemm_hw(trace, n_active=cfg.n_active_experts, active_steps=kw["active"])
        payload = hw_payload("gemm_up", hw, cfg=cfg, trace_dir=trace, extra={"gemm_path": hw.get("gemm_path")})
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    if args.sanity:
        o = run_gemm_up(t, cfg)
        rep = sanity_check("gemm_up", o)
        sp = Path(args.out).parent if args.out else Path("results")
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "sanity_gemm_up.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print("[sanity]", json.dumps(rep, indent=2))
        if rep["has_nan"]:
            return 2
    if args.profile_dir:
        maybe_profile(args.profile_dir, "gemm_up", lambda: run_gemm_up(t, cfg))
    result = bench_op(lambda: run_gemm_up(t, cfg), cfg.warmup, cfg.repeat, device=dev)
    payload = {"kind": "gemm_up", "cfg": {"N": cfg.N, "H": cfg.hidden_size, "I": cfg.moe_intermediate_size, "E_act": cfg.n_active_experts}, "timing": timing_payload(result)}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
