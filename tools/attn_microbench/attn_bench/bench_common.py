from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attn_bench.config import apply_overrides, load_config, repo_root
from attn_bench.init_npu import init_custom_ops, log_versions, require_npu, setup_pythonpath
from attn_bench.sanity import require_sane, sanity_check
from attn_bench.synthetic import assert_shapes, build_synthetic


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--q-len", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--repeat", type=int, default=None)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--msprof", action="store_true",
                   help="Capture pure NPU hardware device time via torch_npu.profiler")
    p.add_argument("--msprof-out", default=None,
                   help="msprof trace root dir; default cfg.msprof.out_dir")


def load_cfg_from_args(args: argparse.Namespace):
    cfg = load_config(args.config)
    cfg = apply_overrides(
        cfg,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        q_len=args.q_len,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    if args.dry_run:
        cfg = apply_overrides(cfg, device="cpu")
    return cfg


def prepare_npu(args: argparse.Namespace):
    init_custom_ops()
    log_versions()
    require_npu()


def write_payload(payload: dict, out: str) -> None:
    text = json.dumps(payload, indent=2)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")


def run_sanity(name: str, fn) -> dict:
    report = sanity_check(name, fn())
    print(json.dumps({"sanity": report}, indent=2))
    require_sane(report)
    return report
