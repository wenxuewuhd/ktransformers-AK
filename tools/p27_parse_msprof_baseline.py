#!/usr/bin/env python3
"""Parse msprof R1/R2 baseline: kernel_details, trace_view, operator_details."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as e:
    print("pandas required:", e, file=sys.stderr)
    sys.exit(2)


def find_profiler_output(run_dir: Path) -> Path | None:
    candidates = list(run_dir.rglob("ASCEND_PROFILER_OUTPUT"))
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[-1]
    # msprof application 导出到 mindstudio_profiler_output
    for name in ("kernel_details.csv", "operator_details.csv", "api_statistic.csv"):
        hits = list(run_dir.rglob(name))
        if hits:
            return hits[0].parent
    hits = list(run_dir.rglob("mindstudio_profiler_output"))
    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[-1]
    return None


def read_csv(path: Path) -> pd.DataFrame:
    # kernel_details 数字列可能带 tab
    df = pd.read_csv(path, skipinitialspace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace("\t", "", regex=False).str.strip()
    return df


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def parse_f2_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(errors="replace")
    steady = re.findall(r"\[steady\] avg tok/s \(prompt 2\+4\) = ([0-9.]+)", text)
    per_prompt = re.findall(
        r"\[timing\] prompt=(\d+) elapsed=([0-9.]+)s tokens=(\d+) throughput=([0-9.]+) tok/s",
        text,
    )
    return {
        "steady_tok_s_prompt_2_4": float(steady[-1]) if steady else None,
        "per_prompt": [
            {
                "prompt_id": int(p),
                "elapsed_s": float(e),
                "completion_tokens": int(t),
                "tok_per_s": float(tp),
            }
            for p, e, t, tp in per_prompt
        ],
    }


def analyze_kernel_details(df: pd.DataFrame) -> dict[str, Any]:
    dur_col = "Duration(us)" if "Duration(us)" in df.columns else "Duration (us)"
    wait_col = "Wait Time(us)" if "Wait Time(us)" in df.columns else "Wait Time (us)"
    name_col = "Name" if "Name" in df.columns else "Op Name"

    df = df.copy()
    df["_dur"] = to_float(df[dur_col])
    df["_wait"] = to_float(df[wait_col])

    total_dur_us = float(df["_dur"].sum())
    total_wait_us = float(df["_wait"].sum())

    # Top kernels by duration
    by_name_dur = (
        df.groupby(name_col, dropna=False)["_dur"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    by_name_wait = (
        df.groupby(name_col, dropna=False)["_wait"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # Per-step (decode batch proxy): group consecutive kernel bursts by time gap
    if "Start Time(us)" in df.columns:
        df["_start"] = to_float(df["Start Time(us)"])
        df_sorted = df.sort_values("_start")
        gaps = df_sorted["_start"].diff().fillna(0)
        # >50ms gap → new step
        step_id = (gaps > 50_000).cumsum()
        df_sorted = df_sorted.assign(_step=step_id)
        steps = []
        for sid, g in df_sorted.groupby("_step"):
            steps.append(
                {
                    "step_idx": int(sid),
                    "kernel_count": int(len(g)),
                    "duration_ms": float(g["_dur"].sum() / 1000.0),
                    "wait_ms": float(g["_wait"].sum() / 1000.0),
                    "wall_span_ms": float(
                        (g["_start"].max() - g["_start"].min() + g["_dur"].max()) / 1000.0
                    )
                    if len(g) > 1
                    else float(g["_dur"].sum() / 1000.0),
                    "start_us": float(g["_start"].min()),
                }
            )
        steps.sort(key=lambda x: x["start_us"])
        # drop obvious prefill / capture outliers (very long span or low kernel count)
        decode_steps = [s for s in steps if s["kernel_count"] >= 50 and s["wall_span_ms"] < 2000]
        if len(decode_steps) >= 3:
            cold = decode_steps[: min(13, len(decode_steps) // 3)]
            steady = decode_steps[max(len(decode_steps) // 3, 13) :]
            cold_wait = sum(s["wait_ms"] for s in cold) / max(len(cold), 1)
            steady_wait = sum(s["wait_ms"] for s in steady) / max(len(steady), 1)
        else:
            cold_wait = steady_wait = None
    else:
        decode_steps = []
        cold_wait = steady_wait = None

    return {
        "total_kernel_duration_ms": total_dur_us / 1000.0,
        "total_kernel_wait_ms": total_wait_us / 1000.0,
        "top10_duration_ms": [
            {"name": str(k), "duration_ms": float(v / 1000.0)} for k, v in by_name_dur.items()
        ],
        "top10_wait_ms": [
            {"name": str(k), "wait_ms": float(v / 1000.0)} for k, v in by_name_wait.items()
        ],
        "decode_steps_sampled": len(decode_steps),
        "cold_avg_wait_ms": cold_wait,
        "steady_avg_wait_ms": steady_wait,
        "decode_step_stats": decode_steps[:5] + decode_steps[-3:] if decode_steps else [],
    }


def analyze_trace_view(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        events = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(events, list):
        return {}

    event_wait_us = 0.0
    event_wait_count = 0
    wait_durs: list[float] = []
    memcpy_us = 0.0
    host_sched_us = 0.0

    for ev in events:
        name = ev.get("name") or ""
        cat = ev.get("cat") or ""
        dur = float(ev.get("dur") or 0)
        if name in ("EVENT_WAIT", "Event::wait", "wait_event"):
            event_wait_us += dur
            event_wait_count += 1
            wait_durs.append(dur)
        if "memcpy" in name.lower() or "Memcpy" in name:
            memcpy_us += dur
        if cat == "cpu_op" and name not in ("EVENT_WAIT", "Event::wait", "wait_event"):
            host_sched_us += dur

    wait_durs_sorted = sorted(wait_durs, reverse=True)
    return {
        "event_wait_total_ms": event_wait_us / 1000.0,
        "event_wait_count": event_wait_count,
        "top10_event_wait_ms": [d / 1000.0 for d in wait_durs_sorted[:10]],
        "host_cpu_op_ms": host_sched_us / 1000.0,
        "memcpy_ms": memcpy_us / 1000.0,
    }


def analyze_operator_details(df: pd.DataFrame) -> dict[str, Any]:
    cols = {c.lower(): c for c in df.columns}
    name = cols.get("name", "Name")
    host_self = cols.get("host self duration(us)", "Host Self Duration(us)")
    dev_self = cols.get("device self duration(us)", "Device Self Duration(us)")

    df = df.copy()
    df["_host"] = to_float(df[host_self])
    df["_dev"] = to_float(df[dev_self])

    item_rows = df[df[name].astype(str).str.contains("item", case=False, na=False)]
    acl_rows = df[df[name].astype(str).str.contains("acl|launch|runtime", case=False, na=False)]

    return {
        "total_host_self_ms": float(df["_host"].sum() / 1000.0),
        "total_device_self_ms": float(df["_dev"].sum() / 1000.0),
        "aten_item_host_ms": float(item_rows["_host"].sum() / 1000.0),
        "aten_item_count": int(len(item_rows)),
        "acl_launch_host_ms": float(acl_rows["_host"].sum() / 1000.0),
    }


def build_step_breakdown(
    kernel: dict[str, Any],
    trace: dict[str, Any],
    f2: dict[str, Any],
) -> dict[str, Any]:
    steady_tok = f2.get("steady_tok_s_prompt_2_4")
    wall_ms_per_token = (1000.0 / steady_tok) if steady_tok and steady_tok > 0 else None

    npu_compute = kernel.get("total_kernel_duration_ms", 0.0)
    npu_wait = kernel.get("total_kernel_wait_ms") or trace.get("event_wait_total_ms", 0.0)
    host_memcpy = trace.get("host_cpu_op_ms", 0.0) + trace.get("memcpy_ms", 0.0)

    # Per-decode-step normalization when we have step samples
    steps = kernel.get("decode_steps_sampled") or 0
    if steps > 0 and kernel.get("decode_step_stats"):
        stats = [
            s
            for s in kernel.get("decode_step_stats", [])
            if isinstance(s, dict) and "duration_ms" in s
        ]
        if stats:
            npu_compute = sum(s["duration_ms"] for s in stats) / len(stats)
            npu_wait = sum(s["wait_ms"] for s in stats) / len(stats)

    total = wall_ms_per_token or (npu_compute + npu_wait + host_memcpy)
    other = max(total - npu_compute - npu_wait - host_memcpy, 0.0)

    def pct(x: float) -> float:
        return round(100.0 * x / total, 1) if total > 0 else 0.0

    return {
        "wall_ms_per_token": wall_ms_per_token,
        "npu_compute_ms": round(npu_compute, 2),
        "npu_wait_ms": round(npu_wait, 2),
        "host_sched_memcpy_ms": round(host_memcpy, 2),
        "other_ms": round(other, 2),
        "total_ms": round(total, 2),
        "pct": {
            "npu_compute": pct(npu_compute),
            "npu_wait": pct(npu_wait),
            "host_sched_memcpy": pct(host_memcpy),
            "other": pct(other),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--label", default="R1")
    ap.add_argument("--f2-log", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--write-json", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    prof = find_profiler_output(run_dir)
    if prof is None:
        print(f"[parse] ERROR: no ASCEND_PROFILER_OUTPUT under {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[parse] ASCEND_PROFILER_OUTPUT={prof}")

    kernel_path = prof / "kernel_details.csv"
    trace_path = prof / "trace_view.json"
    op_path = prof / "operator_details.csv"

    summary: dict[str, Any] = {
        "label": args.label,
        "run_dir": str(run_dir),
        "profiler_output": str(prof),
    }

    if kernel_path.is_file():
        summary["kernel"] = analyze_kernel_details(read_csv(kernel_path))
    else:
        print(f"[parse] WARN: missing {kernel_path}")

    if trace_path.is_file():
        summary["trace"] = analyze_trace_view(trace_path)
    else:
        print(f"[parse] WARN: missing {trace_path}")

    if op_path.is_file():
        summary["operator"] = analyze_operator_details(read_csv(op_path))

    f2 = parse_f2_log(args.f2_log) if args.f2_log else {}
    summary["throughput"] = f2
    summary["step_breakdown"] = build_step_breakdown(
        summary.get("kernel", {}),
        summary.get("trace", {}),
        f2,
    )

    if args.meta and args.meta.is_file():
        summary["meta"] = json.loads(args.meta.read_text())

    out_json = args.write_json or (run_dir / "parsed_summary.json")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[parse] wrote {out_json}")

    sb = summary["step_breakdown"]
    print("\n=== Step breakdown (steady decode) ===")
    print(f"  wall_ms/token     {sb.get('wall_ms_per_token')}")
    print(f"  NPU compute       {sb.get('npu_compute_ms')} ms ({sb['pct']['npu_compute']}%)")
    print(f"  NPU EVENT_WAIT    {sb.get('npu_wait_ms')} ms ({sb['pct']['npu_wait']}%)")
    print(f"  Host/Memcpy       {sb.get('host_sched_memcpy_ms')} ms ({sb['pct']['host_sched_memcpy']}%)")
    print(f"  Other             {sb.get('other_ms')} ms ({sb['pct']['other']}%)")
    if f2.get("steady_tok_s_prompt_2_4"):
        print(f"  steady tok/s      {f2['steady_tok_s_prompt_2_4']}")


if __name__ == "__main__":
    main()
