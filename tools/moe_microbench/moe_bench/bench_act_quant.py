from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from moe_bench.config import add_common_bench_args, repo_root, resolve_cfg, timing_payload
from moe_bench.init_npu import init_custom_ops, log_versions, print_npu_smi, require_npu, setup_pythonpath
from moe_bench.ops_runner import run_act_quant
from moe_bench.sanity import sanity_check
from moe_bench.synthetic import assert_shapes, build_synthetic
from moe_bench.timing import bench_op


def _run_one(cfg, t, *, pre_dispatch: bool, args) -> dict:
    dev = torch.device(cfg.device)

    def fn():
        run_act_quant(t, cfg, pre_dispatch=pre_dispatch)

    if args.sanity:
        x_int8, x_scale = run_act_quant(t, cfg, pre_dispatch=pre_dispatch)
        rep_x = sanity_check("x_int8", x_int8.float())
        rep_s = sanity_check("x_scale", x_scale.float())
        base = Path(args.out).parent if args.out else Path("results")
        sanity_path = base / f"sanity_act_quant_{'pre' if pre_dispatch else 'post'}.json"
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        sanity_path.write_text(json.dumps({"x_int8": rep_x, "x_scale": rep_s}, indent=2), encoding="utf-8")
        if rep_x["has_nan"]:
            raise RuntimeError("sanity failed: NaN in act_quant output")

    result = bench_op(fn, cfg.warmup, cfg.repeat, device=dev)
    kind = "act_quant_pre" if pre_dispatch else "act_quant_post"
    shape = [cfg.num_tokens, cfg.hidden_size] if pre_dispatch else [cfg.N, cfg.hidden_size]
    return {
        "kind": kind,
        "cfg": {"N": cfg.N, "num_tokens": cfg.num_tokens, "H": cfg.hidden_size, "pre_dispatch": pre_dispatch},
        "shape": shape,
        "timing": timing_payload(result),
    }


def _msprof_one(cfg, t, *, pre_dispatch: bool, args, suffix: str):
    from moe_bench.msprof_runner import hw_payload, msprof_kwargs_from_cfg, parse_op_summary, run_with_msprof

    kw = msprof_kwargs_from_cfg(cfg, args.msprof_out)
    name = f"act_quant_{suffix}"
    trace = run_with_msprof(
        lambda: run_act_quant(t, cfg, pre_dispatch=pre_dispatch),
        name,
        **kw,
    )
    for pattern in ("DynamicQuant", "DynamicQuantize", "npu_dynamic_quant"):
        try:
            hw = parse_op_summary(trace, pattern)
            break
        except ValueError:
            hw = None
    if hw is None:
        raise ValueError(f"no dynamic quant op in {trace}; see op_types_seen.txt")
    return hw_payload(
        f"act_quant_{suffix}",
        hw,
        cfg=cfg,
        trace_dir=trace,
        extra={"pre_dispatch": pre_dispatch},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-token activation dynamic quant")
    add_common_bench_args(parser)
    parser.add_argument("--pre-dispatch", action="store_true")
    args = parser.parse_args(argv)

    setup_pythonpath(repo_root())
    cfg = resolve_cfg(args)
    if args.dry_run:
        from moe_bench.config import apply_overrides
        cfg = apply_overrides(cfg, device="cpu")

    t = build_synthetic(cfg, device="cpu" if args.dry_run else cfg.device)
    assert_shapes(cfg, t)

    if args.dry_run:
        print(json.dumps([
            {"kind": "act_quant_post", "shape": list(t.x_bf16.shape)},
            {"kind": "act_quant_pre", "shape": [cfg.num_tokens, cfg.hidden_size]},
        ], indent=2))
        return 0

    init_custom_ops()
    require_npu()
    log_versions()
    print_npu_smi()

    if args.msprof:
        payload = {
            "kind": "act_quant",
            "mode": "msprof_hardware_only",
            "post_hw": _msprof_one(cfg, t, pre_dispatch=False, args=args, suffix="post"),
            "pre_hw": _msprof_one(cfg, t, pre_dispatch=True, args=args, suffix="pre"),
        }
        text = json.dumps(payload, indent=2)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    payloads = [_run_one(cfg, t, pre_dispatch=False, args=args), _run_one(cfg, t, pre_dispatch=True, args=args)]
    text = json.dumps(payloads, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
