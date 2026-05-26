from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    shared_intermediate_size: int
    num_hidden_layers: int
    device: str
    weight_dtype: str
    weight_scale_dtype: str
    weight_scale_strategy: str
    act_dtype: str
    act_scale_strategy: str
    accum_dtype: str
    out_dtype: str
    num_tokens: int
    top_k: int
    n_active_experts: int
    tokens_per_expert: int
    warmup: int
    repeat: int
    seed: int
    ops: dict
    sweep_n_active: list[int]
    sweep_n_active_extended: list[int]
    roofline_hbm_peak_tb_s: float
    roofline_hbm_effective_tb_s: float
    roofline_bytes_per_weight: int
    msprof: dict

    @property
    def N(self) -> int:
        """NPU grouped GEMM 实际 token 数 = n_active * tokens_per_expert (N2)."""
        return self.n_active_experts * self.tokens_per_expert


def microbench_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return microbench_root().parent.parent


def default_config_path() -> Path:
    return microbench_root() / "config" / "dsv4_flash_moe.yaml"


def load_config(path: str | Path | None = None) -> MoEConfig:
    cfg_path = Path(path) if path else default_config_path()
    with open(cfg_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    model = raw["model"]
    runtime = raw["runtime"]
    bench = raw["bench"]
    ops = raw.get("ops", {})
    sweep = raw.get("sweep", {})
    roofline = raw.get("roofline", {})
    msprof = raw.get("msprof", {})

    shared_inter = model.get("shared_intermediate_size")
    if shared_inter in (None, "null"):
        shared_inter = int(model["n_shared_experts"]) * int(model["moe_intermediate_size"])

    return MoEConfig(
        hidden_size=int(model["hidden_size"]),
        moe_intermediate_size=int(model["moe_intermediate_size"]),
        n_routed_experts=int(model["n_routed_experts"]),
        num_experts_per_tok=int(model["num_experts_per_tok"]),
        n_shared_experts=int(model["n_shared_experts"]),
        shared_intermediate_size=int(shared_inter),
        num_hidden_layers=int(model.get("num_hidden_layers", 43)),
        device=str(runtime["device"]),
        weight_dtype=str(runtime["weight_dtype"]),
        weight_scale_dtype=str(runtime["weight_scale_dtype"]),
        weight_scale_strategy=str(runtime.get("weight_scale_strategy", "channel")),
        act_dtype=str(runtime["act_dtype"]),
        act_scale_strategy=str(runtime.get("act_scale_strategy", "token")),
        accum_dtype=str(runtime["accum_dtype"]),
        out_dtype=str(runtime["out_dtype"]),
        num_tokens=int(bench["num_tokens"]),
        top_k=int(bench["top_k"]),
        n_active_experts=int(bench["n_active_experts"]),
        tokens_per_expert=int(bench["tokens_per_expert"]),
        warmup=int(bench["warmup"]),
        repeat=int(bench["repeat"]),
        seed=int(bench["seed"]),
        ops=ops,
        sweep_n_active=[int(x) for x in sweep.get("n_active_experts", [1, 2, 3, 4, 5, 6])],
        sweep_n_active_extended=[int(x) for x in sweep.get("n_active_extended", [])],
        roofline_hbm_peak_tb_s=float(roofline.get("hbm_peak_tb_s", 1.6)),
        roofline_hbm_effective_tb_s=float(roofline.get("hbm_effective_tb_s", 1.0)),
        roofline_bytes_per_weight=int(roofline.get("bytes_per_weight", 1)),
        msprof={
            "enabled": bool(msprof.get("enabled", False)),
            "profiler_level": str(msprof.get("profiler_level", "Level1")),
            "aic_metrics": str(msprof.get("aic_metrics", "PipeUtilization")),
            "skip_first": int(msprof.get("skip_first", 5)),
            "warmup": int(msprof.get("warmup", 2)),
            "active": int(msprof.get("active", 10)),
            "out_dir": str(msprof.get("out_dir", "./npu_results")),
            "record_shapes": bool(msprof.get("record_shapes", True)),
            "with_stack": bool(msprof.get("with_stack", False)),
        },
    )


def apply_overrides(cfg: MoEConfig, **kwargs: Any) -> MoEConfig:
    clean = {k: v for k, v in kwargs.items() if v is not None and hasattr(cfg, k)}
    return replace(cfg, **clean)


def timing_payload(result, extra: dict | None = None) -> dict:
    d = result.to_dict()
    d["dispatch_overhead_us"] = d["host_mean_us"] - d["device_mean_us"]
    if extra:
        d.update(extra)
    return d


def add_common_bench_args(parser) -> None:
    import argparse

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n-active-experts", type=int, default=None)
    parser.add_argument("--tokens-per-expert", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--quick-mode", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="")
    parser.add_argument("--msprof", action="store_true", help="msprof Level1 hardware device time")
    parser.add_argument("--msprof-out", default=None, help="msprof trace root; default cfg.msprof.out_dir")


def resolve_cfg(args) -> MoEConfig:
    cfg = load_config(args.config)
    repeat = 100 if args.quick_mode else args.repeat
    return apply_overrides(
        cfg,
        num_tokens=args.num_tokens,
        top_k=args.top_k,
        n_active_experts=args.n_active_experts,
        tokens_per_expert=args.tokens_per_expert,
        warmup=args.warmup,
        repeat=repeat,
    )
