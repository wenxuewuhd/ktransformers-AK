# NOTE: keep in sync with tools/attn_microbench/attn_bench/msprof_runner.py
# Changes here should be mirrored manually (no shared module by design).

"""Pure NPU hardware op timing via torch_npu.profiler Level1 + op_summary CSV."""

from __future__ import annotations

import csv
import glob
from pathlib import Path
from typing import Callable

import torch_npu


def _glob_csv(trace_dir: Path, rel: str) -> list[str]:
    return sorted(glob.glob(str(trace_dir / "**" / rel), recursive=True))


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _find_profiler_csvs(trace_dir: Path) -> dict[str, list[str]]:
    td = Path(trace_dir)
    return {
        "op_summary": _glob_csv(td, "ASCEND_PROFILER_OUTPUT/op_summary_*.csv"),
        "kernel_details": _glob_csv(td, "ASCEND_PROFILER_OUTPUT/kernel_details.csv"),
        "op_statistic": _glob_csv(td, "ASCEND_PROFILER_OUTPUT/op_statistic.csv"),
        "operator_details": _glob_csv(td, "ASCEND_PROFILER_OUTPUT/operator_details.csv"),
    }


def _stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("no duration samples")
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n if n > 1 else 0.0
    return {
        "device_mean_us": mean,
        "device_std_us": var**0.5,
        "device_p50_us": q(0.50),
        "device_p95_us": q(0.95),
        "device_p99_us": q(0.99),
        "device_max_us": s[-1],
        "device_min_us": s[0],
        "matched_rows": n,
    }


def _match_text(text: str, pattern: str) -> bool:
    return pattern.lower() in (text or "").lower()


def _parse_kernel_details(path: str, op_pattern: str) -> list[float]:
    rows = _read_csv(path)
    dur: list[float] = []
    for row in rows:
        name = row.get("Name") or row.get("Type") or ""
        if not _match_text(name, op_pattern):
            continue
        step = (row.get("Step Id") or "").strip()
        if not step:
            continue
        col = "Duration(us)"
        if col not in row:
            continue
        val = row[col].strip().replace("\t", "")
        if val:
            dur.append(float(val))
    return dur


def _parse_op_summary(path: str, op_pattern: str) -> list[float]:
    rows = _read_csv(path)
    op_col = next((c for c in ("OP Type", "Op Type", "op_type") if rows and c in rows[0]), None)
    if not op_col:
        return []
    dur_col = next(
        (c for c in ("Task Duration(us)", "Duration(us)", "Task Duration (us)") if c in rows[0]),
        None,
    )
    if not dur_col:
        return []
    out: list[float] = []
    for row in rows:
        if _match_text(row.get(op_col, ""), op_pattern):
            out.append(float(row[dur_col]))
    return out


def _parse_operator_details(path: str, op_pattern: str) -> list[float]:
    rows = _read_csv(path)
    out: list[float] = []
    for row in rows:
        name = row.get("Name") or ""
        if not _match_text(name, op_pattern):
            continue
        for col in (
            "Device Self Duration With AICore(us)",
            "Device Self Duration(us)",
            "Device Total Duration With AICore(us)",
            "Device Total Duration(us)",
        ):
            if col in row and row[col]:
                try:
                    out.append(float(str(row[col]).strip()))
                    break
                except ValueError:
                    continue
    return out


def _collect_op_types(csvs: dict[str, list[str]]) -> list[str]:
    types: set[str] = set()
    for path in csvs.get("kernel_details", []):
        for row in _read_csv(path):
            if row.get("Name"):
                types.add(row["Name"])
    for path in csvs.get("op_statistic", []):
        for row in _read_csv(path):
            if row.get("OP Type"):
                types.add(row["OP Type"])
    for path in csvs.get("operator_details", []):
        for row in _read_csv(path):
            if row.get("Name"):
                types.add(row["Name"])
    for path in csvs.get("op_summary", []):
        for row in _read_csv(path):
            for col in ("OP Type", "Op Type"):
                if row.get(col):
                    types.add(row[col])
    return sorted(types)


def run_with_msprof(
    fn: Callable[[], None],
    name: str,
    out_dir: str,
    skip_first: int = 5,
    warmup: int = 2,
    active: int = 10,
    profiler_level: str = "Level1",
    aic_metrics: str = "PipeUtilization",
    record_shapes: bool = True,
    with_stack: bool = False,
) -> Path:
    """Run fn (skip_first + warmup + active) times under msprof; return trace root."""
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)

    level = getattr(torch_npu.profiler.ProfilerLevel, profiler_level)
    metrics = getattr(torch_npu.profiler.AiCMetrics, aic_metrics)
    cfg = torch_npu.profiler._ExperimentalConfig(
        profiler_level=level,
        aic_metrics=metrics,
    )
    sched = torch_npu.profiler.schedule(
        wait=0,
        warmup=warmup,
        active=active,
        repeat=1,
        skip_first=skip_first,
    )

    try:
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=sched,
            experimental_config=cfg,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            record_shapes=record_shapes,
            with_stack=with_stack,
        ) as prof:
            for _ in range(skip_first + warmup + active):
                fn()
                prof.step()
            torch_npu.npu.synchronize()
    except (TypeError, AttributeError):
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            experimental_config=cfg,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            record_shapes=record_shapes,
            with_stack=with_stack,
        ) as prof:
            for _ in range(skip_first + warmup + active):
                fn()
            torch_npu.npu.synchronize()

    return target


def list_op_types_in_trace(trace_dir: Path) -> list[str]:
    """Debug: list all op type names seen in trace CSVs."""
    return _collect_op_types(_find_profiler_csvs(Path(trace_dir)))


def parse_op_summary(
    trace_dir: Path,
    op_pattern: str,
    *,
    sub_ops_per_call: int | None = None,
    active_steps: int | None = None,
) -> dict:
    """Parse profiler CSVs; fuzzy-match op name against op_pattern."""
    trace_dir = Path(trace_dir)
    csvs = _find_profiler_csvs(trace_dir)
    durations: list[float] = []
    source = ""

    for path in csvs.get("op_summary", []):
        d = _parse_op_summary(path, op_pattern)
        if d:
            durations = d
            source = f"op_summary:{path}"
            break

    if not durations:
        for path in csvs.get("kernel_details", []):
            d = _parse_kernel_details(path, op_pattern)
            if d:
                durations = d
                source = f"kernel_details:{path}"
                break

    if not durations:
        for path in csvs.get("operator_details", []):
            d = _parse_operator_details(path, op_pattern)
            if d:
                durations = d[-10:] if len(d) > 10 else d
                source = f"operator_details:{path}"
                break

    if not durations:
        avail = _collect_op_types(csvs)
        types_dump = trace_dir / "op_types_seen.txt"
        types_dump.write_text("\n".join(avail), encoding="utf-8")
        raise ValueError(
            f"no op match {op_pattern!r}; "
            f"saw {len(avail)} unique op types, dumped to {types_dump}"
        )

    out = _stats(durations)
    out["op_pattern"] = op_pattern
    out["source_csv"] = source

    if sub_ops_per_call and sub_ops_per_call > 1:
        n = len(durations)
        steps = active_steps or (n // sub_ops_per_call if n % sub_ops_per_call == 0 else None)
        if steps and steps > 0 and n >= steps:
            if n == steps * sub_ops_per_call:
                per_call = [
                    sum(durations[i : i + sub_ops_per_call])
                    for i in range(0, n, sub_ops_per_call)
                ]
            else:
                per_call = [sum(durations) / steps]
            out = _stats(per_call)
            out["device_mean_us_per_sub_op"] = sum(durations) / n
            out["sub_ops_per_call"] = sub_ops_per_call
            out["op_pattern"] = op_pattern
            out["source_csv"] = source
        else:
            out["device_mean_us_per_sub_op"] = out["device_mean_us"]
            out["device_mean_us"] = out["device_mean_us"] * sub_ops_per_call
            out["sub_ops_per_call"] = sub_ops_per_call

    return out


def parse_op_summary_multi(trace_dir: Path, patterns: list[str]) -> dict:
    """Sum device_mean_us across multiple op patterns (e.g. Silu + Mul + DynamicQuant)."""
    parts = []
    total = 0.0
    for p in patterns:
        s = parse_op_summary(trace_dir, p)
        parts.append(s)
        total += s["device_mean_us"]
    agg = _stats([total])
    agg["op_pattern"] = "+".join(patterns)
    agg["sub_ops"] = parts
    agg["matched_rows"] = sum(p["matched_rows"] for p in parts)
    return agg


def parse_gemm_hw(trace_dir: Path, *, n_active: int, active_steps: int | None = None) -> dict:
    """Try GroupedMatmul first, then QuantMatmul loop fallback with sub-op scaling."""
    for pattern in ("GroupedMatmul", "GroupedMatMul", "GroupMatmul"):
        try:
            hw = parse_op_summary(trace_dir, pattern)
            hw["gemm_path"] = "grouped"
            return hw
        except ValueError:
            continue

    for pattern in ("QuantBatchMatmul", "QuantMatmul", "QuantBatchMatMul"):
        try:
            hw = parse_op_summary(
                trace_dir,
                pattern,
                sub_ops_per_call=max(n_active, 1),
                active_steps=active_steps,
            )
            hw["gemm_path"] = "loop_quant_matmul"
            return hw
        except ValueError:
            continue

    avail = list_op_types_in_trace(trace_dir)
    dump = Path(trace_dir) / "op_types_seen.txt"
    dump.write_text("\n".join(avail), encoding="utf-8")
    raise ValueError(f"no gemm op match; dumped {len(avail)} types to {dump}")


def msprof_kwargs_from_cfg(cfg, msprof_out: str | None = None) -> dict:
    """Build run_with_msprof kwargs from MoEConfig.msprof + optional CLI override."""
    m = cfg.msprof
    return {
        "out_dir": msprof_out or m.get("out_dir", "./npu_results"),
        "skip_first": int(m.get("skip_first", 5)),
        "warmup": int(m.get("warmup", 2)),
        "active": int(m.get("active", 10)),
        "profiler_level": str(m.get("profiler_level", "Level1")),
        "aic_metrics": str(m.get("aic_metrics", "PipeUtilization")),
        "record_shapes": bool(m.get("record_shapes", True)),
        "with_stack": bool(m.get("with_stack", False)),
    }


def hw_payload(kind: str, hw: dict, *, cfg=None, trace_dir: Path | None = None, extra: dict | None = None) -> dict:
    """Standard msprof JSON envelope."""
    payload = {
        "kind": kind,
        "mode": "msprof_hardware_only",
        "op_pattern": hw.get("op_pattern"),
        "device_mean_us": hw["device_mean_us"],
        "device_std_us": hw.get("device_std_us", 0.0),
        "device_p50_us": hw.get("device_p50_us"),
        "device_p95_us": hw.get("device_p95_us"),
        "device_p99_us": hw.get("device_p99_us"),
        "device_max_us": hw.get("device_max_us"),
        "device_min_us": hw.get("device_min_us"),
        "matched_rows": hw.get("matched_rows"),
        "source_csv": hw.get("source_csv"),
    }
    if trace_dir is not None:
        payload["trace_dir"] = str(trace_dir)
    if cfg is not None:
        payload["cfg"] = {
            "N": cfg.N,
            "n_active_experts": cfg.n_active_experts,
            "H": cfg.hidden_size,
            "I": cfg.moe_intermediate_size,
        }
    if extra:
        payload.update(extra)
    return payload
