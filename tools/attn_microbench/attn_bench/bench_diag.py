"""P0/P1 diagnostics: seq scaling, c4_cols sweep, page-table modes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from attn_bench.config import apply_overrides, load_config, repo_root
from attn_bench.init_npu import setup_pythonpath
from attn_bench.metadata import build_metadata
from attn_bench.ops_runner import run_csa_indexer, run_swa_attn
from attn_bench.synthetic import assert_shapes, build_synthetic
from attn_bench.timing import TimingResult, bench_op


def _fmt_us(r: TimingResult) -> str:
    return (
        f"{r.device_mean_us:.1f} ± {r.device_std_us:.1f} "
        f"(p95={r.device_p95_us:.1f}, n={len(r.samples_device)})"
    )


def _row(test: str, cfg, t, idx: TimingResult, extra: dict, swa: TimingResult | None = None) -> dict:
    row = {
        "test": test,
        "seq_len": cfg.seq_len,
        "c4_cols": t.page_spec.c4_cols,
        "c4_pages": t.page_spec.c4_num_pages,
        "seqused_kv": int(t.seqused_kv[0]),
        "diag_snapshot": dict(cfg.diag),
        "indexer_us": idx.to_dict(),
        "indexer_summary": _fmt_us(idx),
        **extra,
    }
    if swa is not None:
        row["swa_attn_us"] = swa.to_dict()
        row["swa_summary"] = _fmt_us(swa)
    return row


def _bench(cfg, warmup: int, repeat: int) -> tuple:
    t = build_synthetic(cfg, swa_no_sink=False)
    assert_shapes(cfg, t)
    meta = build_metadata(cfg, t)
    idx = bench_op(lambda: run_csa_indexer(t, meta, cfg), warmup, repeat)
    swa = bench_op(lambda: run_swa_attn(t, meta, cfg), warmup, repeat)
    return t, idx, swa


def run_c4_cols_sweep(base_cfg, cols_list: list[int], warmup: int, repeat: int) -> list[dict]:
    """Each col in a fresh subprocess — small c4_cols can abort NPU without killing sweep."""
    rows: list[dict] = []
    for cols in cols_list:
        cfg = apply_overrides(base_cfg, diag={"override_c4_cols": cols})
        env = os.environ.copy()
        env["DIAG_OVERRIDE_C4_COLS"] = str(cols)
        out_tmp = f"/tmp/attn_diag_c4_{cols}.json"
        cmd = [
            sys.executable,
            "-m",
            "attn_bench.bench_diag",
            "--scenario",
            "c4_cols_single",
            "--seq-len",
            str(cfg.seq_len),
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--out",
            out_tmp,
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0 or not Path(out_tmp).is_file():
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            rows.append(
                {
                    "test": "c4_cols_sweep",
                    "override_c4_cols": cols,
                    "error": err[-1] if err else f"exit={proc.returncode}",
                    "diag_snapshot": dict(cfg.diag),
                }
            )
            continue
        try:
            payload = json.loads(Path(out_tmp).read_text(encoding="utf-8"))
            rows.extend(payload.get("results", []))
        except json.JSONDecodeError:
            rows.append(
                {
                    "test": "c4_cols_sweep",
                    "override_c4_cols": cols,
                    "error": "invalid subprocess json",
                }
            )
    return rows


def run_c4_cols_single(base_cfg, cols_list: list[int], warmup: int, repeat: int) -> list[dict]:
    rows = []
    for cols in cols_list:
        cfg = apply_overrides(base_cfg, diag={"override_c4_cols": cols})
        try:
            t, idx, swa = _bench(cfg, warmup, repeat)
            rows.append(_row("c4_cols_sweep", cfg, t, idx, {"override_c4_cols": cols}, swa))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "test": "c4_cols_sweep",
                    "override_c4_cols": cols,
                    "error": str(exc),
                    "diag_snapshot": dict(cfg.diag),
                }
            )
    return rows


def run_unique_pages(base_cfg, warmup: int, repeat: int) -> list[dict]:
    rows = []
    for label, unique in (("default_modulo", False), ("unique_pages", True)):
        cfg = apply_overrides(base_cfg, diag={"page_table_unique_pages": unique})
        t, idx, swa = _bench(cfg, warmup, repeat)
        rows.append(
            _row(
                "unique_pages",
                cfg,
                t,
                idx,
                {"label": label, "page_table_unique_pages": unique},
                swa,
            )
        )
    return rows


def run_seq_sweep(base_cfg, warmup: int, repeat: int) -> list[dict]:
    rows = []
    for seq_len in (1024, 4096, 8192, 16384, 32768):
        cfg = apply_overrides(base_cfg, seq_len=seq_len)
        t, idx, swa = _bench(cfg, warmup, repeat)
        rows.append(_row("seq_sweep", cfg, t, idx, {}, swa))
    return rows


def run_extreme_seqused(base_cfg, warmup: int, repeat: int) -> list[dict]:
    rows = []
    base = apply_overrides(base_cfg, seq_len=32768)
    for label, seqused in (("baseline", 32768), ("extreme_128", 128)):
        cfg = apply_overrides(base, diag={"override_seqused_kv": seqused})
        t, idx, _ = _bench(cfg, warmup, repeat)
        rows.append(_row("extreme_seqused_kv", cfg, t, idx, {"label": label}))
    return rows


def run_key_len_variant(base_cfg, warmup: int, repeat: int) -> list[dict]:
    rows = []
    base = apply_overrides(base_cfg, seq_len=32768)
    for label, key_len in (("token_len", 32768), ("c4_len", 8192)):
        cfg = apply_overrides(base, diag={"override_seqused_kv": key_len})
        t, idx, _ = _bench(cfg, warmup, repeat)
        rows.append(
            _row("key_len_variant", cfg, t, idx, {"label": label, "key_len": key_len})
        )
    return rows


def _floor_verdict(results: list[dict]) -> dict:
    means = [
        r["indexer_us"]["device_mean_us"]
        for r in results
        if "indexer_us" in r and "error" not in r
    ]
    errors = [r.get("override_c4_cols") for r in results if r.get("error")]
    if not means:
        return {"verdict": "no_data", "kernel_errors": errors}
    lo, hi = min(means), max(means)
    spread_pct = (hi - lo) / lo * 100 if lo else 0.0
    return {
        "indexer_min_us": lo,
        "indexer_max_us": hi,
        "spread_pct": spread_pct,
        "floor_confirmed": spread_pct < 5.0,
        "verdict": "floor confirmed" if spread_pct < 5.0 else "floor refuted",
        "kernel_errors": errors,
        "successful_cols": [
            r.get("override_c4_cols", r.get("c4_cols"))
            for r in results
            if "indexer_us" in r and "error" not in r
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Attention microbench diagnostics")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seq-len", type=int, default=32768)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="c4_cols_sweep|c4_cols_single|unique_pages|seq_sweep|extreme|key_len|all|p0",
    )
    p.add_argument("--out", type=str, default="")
    args = p.parse_args(argv)

    setup_pythonpath(repo_root())
    from attn_bench.bench_common import prepare_npu

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(
        base_cfg, seq_len=args.seq_len, warmup=args.warmup, repeat=args.repeat
    )
    prepare_npu(argparse.Namespace(dry_run=False))

    scenario = args.scenario
    results: list[dict] = []

    if scenario in ("c4_cols_sweep", "all", "p1"):
        cols_env = os.environ.get("DIAG_OVERRIDE_C4_COLS", "")
        cols_list = (
            [int(x) for x in cols_env.split()]
            if cols_env
            else [4, 16, 64, 256, 1024, 8192]
        )
        results.extend(run_c4_cols_sweep(base_cfg, cols_list, args.warmup, args.repeat))

    if scenario == "c4_cols_single":
        cols_env = os.environ.get("DIAG_OVERRIDE_C4_COLS", "8192")
        cols_list = [int(x) for x in cols_env.split()]
        results.extend(run_c4_cols_single(base_cfg, cols_list, args.warmup, args.repeat))

    if scenario in ("unique_pages", "all", "p1"):
        results.extend(run_unique_pages(base_cfg, args.warmup, args.repeat))

    if scenario in ("seq_sweep", "all", "p0", "p1"):
        results.extend(run_seq_sweep(base_cfg, args.warmup, args.repeat))

    if scenario in ("extreme", "all", "p0"):
        results.extend(run_extreme_seqused(base_cfg, args.warmup, args.repeat))

    if scenario in ("key_len", "all", "p0"):
        results.extend(run_key_len_variant(base_cfg, args.warmup, args.repeat))

    payload = {
        "scenario": scenario,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": results,
    }
    if scenario in ("c4_cols_sweep", "all", "p1") and any(
        r["test"] == "c4_cols_sweep" for r in results
    ):
        payload["p1_1_floor_evidence"] = _floor_verdict(
            [r for r in results if r["test"] == "c4_cols_sweep"]
        )

    out_path = args.out or {
        "c4_cols_sweep": "results/diag_c4_cols_sweep.json",
        "unique_pages": "results/diag_unique_pages.json",
        "seq_sweep": "results/diag_seq_scaling.json",
    }.get(scenario, "results/diag_combined.json")

    text = json.dumps(payload, indent=2)
    print(text)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"[diag] wrote {out_path}", flush=True)

    if "p1_1_floor_evidence" in payload:
        fe = payload["p1_1_floor_evidence"]
        fe_path = Path("results/p1_1_floor_evidence.json")
        fe_path.parent.mkdir(parents=True, exist_ok=True)
        fe_path.write_text(json.dumps(fe, indent=2), encoding="utf-8")
        print(f"[diag] wrote {fe_path}", flush=True)
        if "spread_pct" in fe:
            print(
                f"P1.1 done: c4_cols spread = {fe['spread_pct']:.1f}%; "
                f"floor假说 {fe['verdict']}",
                flush=True,
            )
        else:
            print(f"P1.1 done: {fe.get('verdict', 'no_data')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
