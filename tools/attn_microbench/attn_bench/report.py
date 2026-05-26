from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from attn_bench.config import load_config
from attn_bench.roofline import (
    csa_attn_cmp_kv_bytes,
    effective_tb_s,
    hca_cmp_kv_bytes,
    hw_lower_bound_us,
    indexer_kv_bytes,
    swa_kv_bytes,
    util_vs_achievable,
)


def _block(data: dict, key: str) -> dict | None:
    return data.get(key)


def _fmt_mean_std(block: dict | None) -> str:
    if not block:
        return "-"
    mean = block.get("device_mean_us", block.get("mean_us"))
    std = block.get("device_std_us")
    if mean is None:
        return "-"
    if std is not None:
        return f"{mean:.1f} ± {std:.1f}"
    return f"{mean:.1f}"


def _mean_us(data: dict, key: str) -> float | None:
    block = _block(data, key)
    if not block:
        return None
    return block.get("device_mean_us", block.get("mean_us"))


def _hw_mean(data: dict, key: str = "attn_hw") -> float | None:
    block = _block(data, key)
    if not block:
        return None
    return block.get("device_mean_us")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _overhead_row(
    name: str,
    py_mean: float | None,
    py_std: float | None,
    hw_mean: float | None,
    hw_std: float | None,
) -> dict:
    if py_mean is None or hw_mean is None:
        return {
            "op": name,
            "python_event_us": "-",
            "msprof_device_us": "-",
            "launch_overhead_us": "-",
            "overhead_pct": "-",
            "py_mean": py_mean,
            "hw_mean": hw_mean,
        }
    overhead = py_mean - hw_mean
    pct = (overhead / py_mean * 100.0) if py_mean else 0.0
    py_fmt = f"{py_mean:.1f} ± {py_std:.1f}" if py_std is not None else f"{py_mean:.1f}"
    hw_fmt = f"{hw_mean:.1f} ± {hw_std:.1f}" if hw_std is not None else f"{hw_mean:.1f}"
    return {
        "op": name,
        "python_event_us": py_fmt,
        "msprof_device_us": hw_fmt,
        "launch_overhead_us": f"{overhead:.1f}",
        "overhead_pct": f"{pct:.1f}%",
        "py_mean": py_mean,
        "hw_mean": hw_mean,
        "overhead_pct_val": pct,
    }


def _verdict_overhead(rows: list[dict]) -> str:
    vals = [r["overhead_pct_val"] for r in rows if "overhead_pct_val" in r]
    if not vals:
        return "数据不足"
    avg = sum(vals) / len(vals)
    if avg > 50:
        return "Python launch 主导"
    if avg < 30:
        return "NPU kernel 主导"
    return "中间（两边都贡献）"


def _summary_mode(args) -> str:
    rows = []
    swa_attn = None
    for path in args.inputs:
        data = _load_json(path)
        kind = data["kind"]
        row = {
            "kind": kind,
            "seq_len": data.get("seq_len"),
            "batch_size": data.get("batch_size", 1),
            "indexer": _fmt_mean_std(_block(data, "indexer_us")),
            "attn": _fmt_mean_std(_block(data, "attn_us")),
            "isolated_sum": _fmt_mean_std(
                _block(data, "isolated_device_sum_us") or _block(data, "total_us")
            ),
            "attn_mean_raw": _mean_us(data, "attn_us"),
            "attn_host": (
                data.get("attn_us", {}).get("host_mean_us") if data.get("attn_us") else None
            ),
            "repeat_n": (data.get("attn_us") or {}).get("n"),
        }
        if kind == "swa":
            swa_attn = row["attn_mean_raw"]
        rows.append(row)

    lines = [
        "# Synthetic Attention Microbench Summary",
        "",
        "> ⚠️ Read `results/diag_seq_scaling.json` before trusting seq_len scaling.",
        "> `attn_us` for csa/hca **includes SWA branch**; do not add with swa.",
        "> `isolated_sum_us` = indexer + attn **independent** timings (not fused E2E).",
        "",
        "| kind | seq_len | batch | n | indexer (µs) | attn (µs) | attn_compressed_only | isolated_sum (µs) | attn_host (µs) |",
        "|------|---------|-------|---|--------------|-----------|----------------------|-------------------|----------------|",
    ]
    for r in sorted(rows, key=lambda x: x["kind"]):
        compressed = "-"
        if r["kind"] in ("csa", "hca") and swa_attn is not None and r["attn_mean_raw"] is not None:
            compressed = f"{r['attn_mean_raw'] - swa_attn:.1f}"
        lines.append(
            f"| {r['kind']} | {r['seq_len']} | {r['batch_size']} | {r.get('repeat_n', '-')} | "
            f"{r['indexer']} | {r['attn']} | {compressed} | {r['isolated_sum']} | "
            f"{r.get('attn_host', '-')} |"
        )
    return "\n".join(lines) + "\n"


def _comparison_mode(args) -> str:
    msprof_paths = args.msprof_jsons or []
    event_paths = args.event_jsons or []
    if len(msprof_paths) < 3 or len(event_paths) < 3:
        raise SystemExit("comparison mode needs 3 msprof JSONs and 3 event JSONs")

    by_kind_msprof = {_load_json(p)["kind"]: _load_json(p) for p in msprof_paths}
    by_kind_event = {_load_json(p)["kind"]: _load_json(p) for p in event_paths}
    cfg = load_config()

    swa_e = by_kind_event["swa"]
    csa_e = by_kind_event["csa"]
    hca_e = by_kind_event["hca"]
    swa_m = by_kind_msprof["swa"]
    csa_m = by_kind_msprof["csa"]
    hca_m = by_kind_msprof["hca"]

    rows = [
        _overhead_row(
            "swa_attn",
            _mean_us(swa_e, "attn_us"),
            swa_e.get("attn_us", {}).get("device_std_us"),
            _hw_mean(swa_m, "attn_hw"),
            swa_m.get("attn_hw", {}).get("device_std_us"),
        ),
        _overhead_row(
            "csa_indexer",
            _mean_us(csa_e, "indexer_us"),
            csa_e.get("indexer_us", {}).get("device_std_us"),
            _hw_mean(csa_m, "indexer_hw"),
            csa_m.get("indexer_hw", {}).get("device_std_us"),
        ),
        _overhead_row(
            "csa_attn",
            _mean_us(csa_e, "attn_us"),
            csa_e.get("attn_us", {}).get("device_std_us"),
            _hw_mean(csa_m, "attn_hw"),
            csa_m.get("attn_hw", {}).get("device_std_us"),
        ),
        _overhead_row(
            "hca_attn",
            _mean_us(hca_e, "attn_us"),
            hca_e.get("attn_us", {}).get("device_std_us"),
            _hw_mean(hca_m, "attn_hw"),
            hca_m.get("attn_hw", {}).get("device_std_us"),
        ),
    ]

    tb = effective_tb_s(cfg)
    roof_rows = [
        ("swa_attn", _hw_mean(swa_m, "attn_hw"), swa_kv_bytes(cfg)),
        ("csa_indexer", _hw_mean(csa_m, "indexer_hw"), indexer_kv_bytes(cfg)),
        ("csa_attn", _hw_mean(csa_m, "attn_hw"), csa_attn_cmp_kv_bytes(cfg)),
        ("hca_attn", _hw_mean(hca_m, "attn_hw"), hca_cmp_kv_bytes(cfg)),
    ]

    lines = [
        "# msprof vs Python Event Comparison",
        "",
        "> Python Event = eager end-to-end per op call",
        "> msprof device time = NPU hardware kernel time (Level1 op_summary)",
        "",
        "## Launch Overhead 拆解",
        "",
        "| op | python_event_us (mean±std) | msprof_device_us (mean±std) | launch_overhead_us | overhead_pct |",
        "|----|----------------------------|------------------------------|--------------------|--------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['op']} | {r['python_event_us']} | {r['msprof_device_us']} | "
            f"{r['launch_overhead_us']} | {r['overhead_pct']} |"
        )

    verdict = _verdict_overhead(rows)
    lines.extend([
        "",
        "判定:",
        f"- overhead_pct > 50% → Python launch 主导, 优化方向 = NPUGraph / op fusion",
        f"- overhead_pct < 30% → NPU kernel 主导, 优化方向 = kernel 内部 / 算法",
        f"- 中间 → 两边都要看",
        f"- **本次判定: {verdict}**",
        "",
        "## Roofline 对照 (hw 层, effective {:.1f} TB/s)".format(tb),
        "",
        "| op | device_us (msprof) | hw_lower_bound_us | util_vs_achievable |",
        "|----|--------------------|-------------------|---------------------|",
    ])
    for name, dev_us, nbytes in roof_rows:
        lb = hw_lower_bound_us(nbytes, tb)
        util = util_vs_achievable(dev_us or 0.0, lb) if dev_us else 0.0
        lines.append(
            f"| {name} | {dev_us:.1f} | {lb:.2f} | {util:.3f} |"
            if dev_us is not None
            else f"| {name} | - | {lb:.2f} | - |"
        )
    lines.extend([
        "",
        "> util_vs_achievable < 0.1 → kernel 远未打满带宽",
        "> util_vs_achievable > 0.5 → kernel 已接近带宽极限",
        "",
        "## 顶部速览 (overhead_pct)",
    ])
    for r in rows:
        lines.append(f"- {r['op']}: {r['overhead_pct']}")
    return "\n".join(lines) + "\n"


def _msprof_sweep_mode(args) -> str:
    paths = sorted(glob.glob(args.inputs[0]) if len(args.inputs) == 1 and "*" in args.inputs[0] else args.inputs)
    rows = []
    base_idx = None
    for p in paths:
        data = _load_json(p)
        seq = int(data.get("seq_len", 0))
        idx = _hw_mean(data, "indexer_hw")
        attn = _hw_mean(data, "attn_hw")
        if seq == 1024:
            base_idx = idx
        scaling = (idx / base_idx) if base_idx and idx and seq != 1024 else (1.0 if seq == 1024 else None)
        rows.append({
            "seq_len": seq,
            "indexer_hw": idx,
            "indexer_std": data.get("indexer_hw", {}).get("device_std_us"),
            "attn_hw": attn,
            "attn_std": data.get("attn_hw", {}).get("device_std_us"),
            "scaling": scaling,
        })
    rows.sort(key=lambda x: x["seq_len"])

    max_scaling = max((r["scaling"] or 1.0) for r in rows)
    if max_scaling > 2.0:
        sweep_verdict = "硬件层随 seq_len 单调 — P1.5 平坦结论被推翻（Python overhead 平坦）"
    elif max_scaling < 1.2:
        sweep_verdict = "硬件层也平坦 — kernel-internal floor 实锤"
    else:
        sweep_verdict = f"硬件层部分缩放 (max ratio {max_scaling:.2f}×) — 需分项分析"

    lines = [
        "# msprof CSA seq sweep (hardware only)",
        "",
        "| seq_len | indexer_hw_us (mean±std) | attn_hw_us (mean±std) | scaling_vs_1k |",
        "|---------|--------------------------|------------------------|---------------|",
    ]
    for r in rows:
        idx_fmt = (
            f"{r['indexer_hw']:.1f} ± {r['indexer_std']:.1f}"
            if r["indexer_hw"] is not None and r["indexer_std"] is not None
            else "-"
        )
        attn_fmt = (
            f"{r['attn_hw']:.1f} ± {r['attn_std']:.1f}"
            if r["attn_hw"] is not None and r["attn_std"] is not None
            else "-"
        )
        sc = f"{r['scaling']:.2f}" if r["scaling"] is not None else "-"
        lines.append(f"| {r['seq_len']} | {idx_fmt} | {attn_fmt} | {sc} |")

    lines.extend(["", "判定:", f"- {sweep_verdict}"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize SWA/CSA/HCA bench JSON")
    p.add_argument("--mode", choices=("summary", "comparison", "msprof-sweep"), default="summary")
    p.add_argument("--inputs", nargs="+", default=[])
    p.add_argument("--msprof-jsons", nargs="+", default=None)
    p.add_argument("--event-jsons", nargs="+", default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args(argv)

    if args.mode == "comparison":
        text = _comparison_mode(args)
    elif args.mode == "msprof-sweep":
        if not args.inputs:
            raise SystemExit("msprof-sweep requires --inputs")
        text = _msprof_sweep_mode(args)
    else:
        if not args.inputs:
            raise SystemExit("summary mode requires --inputs")
        text = _summary_mode(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
