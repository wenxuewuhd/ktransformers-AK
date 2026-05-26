from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from moe_bench.config import apply_overrides, load_config
from moe_bench.roofline import lower_bound_us, routed_weight_bytes, shared_weight_bytes, utilizations


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _timing(d):
    if not d:
        return {}
    if isinstance(d, list):
        return {}
    return d.get("timing") or d.get("segments", {}).get("post_dispatch") or {}


def _mean(d, key="device_mean_us"):
    t = _timing(d)
    return t.get(key)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def _hw_us(obj) -> float | None:
    if not obj:
        return None
    if isinstance(obj.get("device_mean_us"), (int, float)):
        return obj["device_mean_us"]
    if obj.get("mode") == "msprof_hardware_only":
        return obj.get("device_mean_us")
    if "post_hw" in obj:
        return obj["post_hw"].get("device_mean_us")
    if "grouped_hw" in obj:
        return obj["grouped_hw"].get("device_mean_us")
    return None


def _event_us(data: dict, kind: str) -> float | None:
    if kind == "act_quant_post":
        obj = data.get("act_quant_post")
        if obj is None and isinstance(data.get("act_quant"), list):
            for item in data["act_quant"]:
                if item.get("kind") == "act_quant_post":
                    obj = item
        return _mean(obj)
    if kind == "act_quant_pre":
        obj = data.get("act_quant_pre")
        if obj is None and isinstance(data.get("act_quant"), list):
            for item in data.get("act_quant", []):
                if item.get("kind") == "act_quant_pre":
                    obj = item
        return _mean(obj)
    obj = data.get(kind)
    return _mean(obj)


def _roofline_row(cfg, kind, actual_us):
    if actual_us is None:
        return ("-", "-", "-", "-")
    if kind == "gemm_up":
        wb, _ = routed_weight_bytes(cfg)
        u = utilizations(actual_us, wb, cfg)
    elif kind == "gemm_down":
        _, wb = routed_weight_bytes(cfg)
        u = utilizations(actual_us, wb, cfg)
    elif kind == "routed_full":
        up_b, dn_b = routed_weight_bytes(cfg)
        u = utilizations(actual_us, up_b + dn_b, cfg)
    elif kind == "shared_expert":
        gu, dn = shared_weight_bytes(cfg)
        u = utilizations(actual_us, gu + dn, cfg)
    else:
        return ("(n/a)", "(n/a)", "(n/a)", "(n/a)")
    return (_fmt(u["lb_peak_us"]), _fmt(u["util_vs_peak"]), _fmt(u["lb_effective_us"]), _fmt(u["util_vs_achievable"]))


def _load_msprof_map(paths: list[str]) -> dict:
    out = {}
    for p in paths:
        obj = _load(Path(p))
        if not obj:
            continue
        kind = obj.get("kind", Path(p).stem.replace("_msprof", ""))
        out[kind] = obj
        if kind.startswith("silu_mul"):
            out["silu_mul_unfused"] = obj
        if kind == "act_quant":
            out["act_quant_post_hw"] = obj.get("post_hw")
            out["act_quant_pre_hw"] = obj.get("pre_hw")
    return out


def _load_event_map(paths: list[str]) -> dict:
    data = {}
    for ip in paths:
        obj = _load(Path(ip))
        if obj is None:
            continue
        if isinstance(obj, list):
            for item in obj:
                data[item.get("kind", Path(ip).stem)] = item
        else:
            data[obj.get("kind", Path(ip).stem)] = obj
    return data


def _comparison_row(label: str, py_us: float | None, hw_us: float | None) -> dict:
    if py_us is None or hw_us is None:
        return {"segment": label, "python_event_us": py_us, "msprof_device_us": hw_us}
    overhead = py_us - hw_us
    pct = (overhead / py_us * 100) if py_us > 0 else 0.0
    return {
        "segment": label,
        "python_event_us": py_us,
        "msprof_device_us": hw_us,
        "launch_overhead_us": overhead,
        "overhead_pct": pct,
    }


def _comparison_mode(args) -> str:
    cfg = load_config(args.config)
    msprof = _load_msprof_map(args.msprof_jsons or [])
    events = _load_event_map(args.event_jsons or [])

    rows = [
        _comparison_row("act_quant_post", _event_us(events, "act_quant_post"), _hw_us(msprof.get("act_quant", {}).get("post_hw") if msprof.get("act_quant") else msprof.get("act_quant_post_hw"))),
        _comparison_row("act_quant_pre", _event_us(events, "act_quant_pre"), _hw_us(msprof.get("act_quant", {}).get("pre_hw") if msprof.get("act_quant") else msprof.get("act_quant_pre_hw"))),
        _comparison_row("gemm_up", _mean(events.get("gemm_up")), _hw_us(msprof.get("gemm_up"))),
        _comparison_row("silu_mul", _mean(events.get("silu_mul_unfused")) or _mean(events.get("silu_mul_unfused")), _hw_us(msprof.get("silu_mul_unfused"))),
        _comparison_row("gemm_down", _mean(events.get("gemm_down")), _hw_us(msprof.get("gemm_down"))),
        _comparison_row("routed_full", _mean(events.get("routed_full")), msprof.get("routed_full", {}).get("device_mean_us")),
        _comparison_row("shared_expert", _mean(events.get("shared_expert")), _hw_us(msprof.get("shared_expert"))),
    ]

    lines = [
        "# msprof vs Python Event Comparison (MoE)",
        "",
        "> Python Event = eager end-to-end per op call (D4)",
        "> msprof device time = NPU hardware kernel time (Level1, D5)",
        "> 两者差额 = launch overhead",
        "",
        "## Launch Overhead 拆解",
        "",
        "| segment | python_event_us | msprof_device_us | launch_overhead_us | overhead_pct |",
        "|---------|-----------------|------------------|--------------------|--------------|",
    ]
    for r in rows:
        if r.get("python_event_us") is None and r.get("msprof_device_us") is None:
            continue
        lines.append(
            f"| {r['segment']} | {_fmt(r.get('python_event_us'))} | {_fmt(r.get('msprof_device_us'))} | "
            f"{_fmt(r.get('launch_overhead_us'))} | {_fmt(r.get('overhead_pct'))}% |"
        )

    lines += [
        "",
        "## Roofline 对照 (硬件层, measured @ {:.2f} TB/s)".format(cfg.roofline_hbm_effective_tb_s),
        "",
        "| segment | device_us (hw) | lb @ measured | util_vs_measured |",
        "|---------|----------------|---------------|------------------|",
    ]
    roof = [
        ("gemm_up", _hw_us(msprof.get("gemm_up")), routed_weight_bytes(cfg)[0]),
        ("gemm_down", _hw_us(msprof.get("gemm_down")), routed_weight_bytes(cfg)[1]),
        ("shared_expert", _hw_us(msprof.get("shared_expert")), sum(shared_weight_bytes(cfg))),
    ]
    for name, hw, wb in roof:
        lb = lower_bound_us(wb, cfg.roofline_hbm_effective_tb_s)
        util = (lb / hw * 100) if hw and hw > 0 else None
        lines.append(f"| {name} | {_fmt(hw)} | {_fmt(lb)} | {_fmt(util)}% |")

    gvl = msprof.get("grouped_vs_loop")
    if gvl:
        up = next((r for r in gvl.get("results", []) if r.get("target") == "up"), gvl)
        py_g = events.get("grouped_vs_loop", {}).get("results", [{}])[0].get("grouped", {}).get("device_mean_us")
        py_l = events.get("grouped_vs_loop", {}).get("results", [{}])[0].get("loop", {}).get("device_mean_us")
        hw_g = up.get("grouped_hw", {}).get("device_mean_us")
        hw_l = up.get("loop_hw", {}).get("device_mean_us")
        lines += [
            "",
            "## Grouped vs Loop 硬件层确认 (D4 谜团)",
            "",
            "| path | python_event_us | hw_device_us | overhead_us | overhead_pct |",
            "|------|-----------------|--------------|-------------|--------------|",
        ]
        for label, py, hw in (("grouped", py_g, hw_g), ("loop", py_l, hw_l)):
            row = _comparison_row(label, py, hw)
            lines.append(
                f"| {label} | {_fmt(row.get('python_event_us'))} | {_fmt(row.get('msprof_device_us'))} | "
                f"{_fmt(row.get('launch_overhead_us'))} | {_fmt(row.get('overhead_pct'))}% |"
            )
        if hw_g and hw_l:
            if hw_g <= hw_l * 1.15:
                verdict = "hw_grouped ≈ hw_loop：差异主要来自 grouped fallback 的 Python 路径开销"
            elif hw_g > hw_l * 1.5:
                verdict = "hw_grouped > hw_loop × 1.5：NPU kernel 层面 grouped 实现欠优化"
            else:
                verdict = "hw 层有一定差距，Python overhead 与 kernel 差异并存"
            lines += ["", f"判定: {verdict}"]

    lines += [
        "",
        "## 顶部速览 (overhead_pct)",
    ]
    for r in rows:
        if r.get("overhead_pct") is not None:
            lines.append(f"- {r['segment']}: {r['overhead_pct']:.1f}%")

    gd = _hw_us(msprof.get("gemm_down"))
    _, wb_dn = routed_weight_bytes(cfg)
    lb_dn = lower_bound_us(wb_dn, cfg.roofline_hbm_effective_tb_s)
    util_dn = (lb_dn / gd) if gd and gd > 0 else None
    if util_dn is not None:
        if util_dn > 0.8:
            gd_verdict = "gemm_down 真打满带宽"
        elif util_dn < 0.5:
            gd_verdict = "D4 Event 99.78% 含大量 Python overhead，硬件 util 远低于 Event 报数"
        else:
            gd_verdict = "gemm_down 中等 util，Event 与硬件层均有优化空间"
        lines += ["", f"gemm_down util_vs_measured = {util_dn*100:.1f}% (D4 Event util_eff=99.78%)；判定: {gd_verdict}"]

    return "\n".join(lines) + "\n"


def _msprof_n_active_sweep(args) -> str:
    cfg = load_config(args.config)
    by_na: dict[int, dict] = {}
    for p in args.inputs:
        m = re.search(r"_n(\d+)\.json$", p)
        if not m:
            continue
        na = int(m.group(1))
        obj = _load(Path(p))
        if not obj:
            continue
        kind = obj.get("kind", "").replace("_msprof", "")
        by_na.setdefault(na, {})[kind] = obj

    lines = [
        "# msprof n_active sweep (hardware only)",
        "",
        "| n_active | gemm_up_hw | gemm_up_util | gemm_down_hw | gemm_down_util | routed_full_hw |",
        "|---------:|-----------:|-------------:|-------------:|---------------:|---------------:|",
    ]
    for na in sorted(by_na):
        scfg = apply_overrides(cfg, n_active_experts=na)
        up_b, dn_b = routed_weight_bytes(scfg)
        lb_up = lower_bound_us(up_b, scfg.roofline_hbm_effective_tb_s)
        lb_dn = lower_bound_us(dn_b, scfg.roofline_hbm_effective_tb_s)
        up_hw = _hw_us(by_na[na].get("gemm_up"))
        dn_hw = _hw_us(by_na[na].get("gemm_down"))
        rf_hw = by_na[na].get("routed_full", {}).get("device_mean_us")
        up_util = (lb_up / up_hw * 100) if up_hw else None
        dn_util = (lb_dn / dn_hw * 100) if dn_hw else None
        lines.append(
            f"| {na} | {_fmt(up_hw)} | {_fmt(up_util)}% | {_fmt(dn_hw)} | {_fmt(dn_util)}% | {_fmt(rf_hw)} |"
        )
    lines += ["", "> act_quant(pre) 不随 n_active 变化（固定 [num_tokens, H]）。"]
    return "\n".join(lines) + "\n"


def _summary_mode(args) -> str:
    cfg = load_config(args.config)
    data = {}
    for ip in args.inputs:
        obj = _load(Path(ip))
        if obj is None:
            continue
        if isinstance(obj, list):
            for item in obj:
                data[item.get("kind", Path(ip).stem)] = item
        else:
            data[obj.get("kind", Path(ip).stem)] = obj

    lines = [
        "# NPU MoE Microbench Summary",
        "",
        "> 仅 NPU 段，不含 gating / dispatch / combine / CPU MoE。",
        "",
        "| segment | dev_mean_us | p99 | host_mean_us | dispatch_overhead_us | lb_peak | util_peak | lb_eff | util_eff |",
        "|---------|-------------|-----|--------------|----------------------|---------|-----------|--------|----------|",
    ]

    rows = [
        ("act_quant_post", data.get("act_quant_post")),
        ("act_quant_pre", data.get("act_quant_pre")),
        ("gemm_up", data.get("gemm_up")),
    ]
    silu_keys = [k for k in data if k.startswith("silu_mul")]
    for k in sorted(silu_keys):
        rows.append((k, data[k]))
    rows += [
        ("gemm_down", data.get("gemm_down")),
        ("routed_full", data.get("routed_full")),
        ("shared_expert", data.get("shared_expert")),
    ]

    for kind, obj in rows:
        if not obj:
            continue
        t = _timing(obj)
        lb_p, u_p, lb_e, u_e = ("(n/a)", "(n/a)", "(n/a)", "(n/a)") if "act_quant" in kind or "silu" in kind else _roofline_row(cfg, kind.replace("_post", "").replace("_pre", ""), t.get("device_mean_us"))
        if kind == "routed_full":
            lb_p, u_p, lb_e, u_e = _roofline_row(cfg, "routed_full", t.get("device_mean_us"))
        dispatch = t.get("dispatch_overhead_us", (t.get("host_mean_us") or 0) - (t.get("device_mean_us") or 0))
        lines.append(
            f"| {kind} | {_fmt(t.get('device_mean_us'))} | {_fmt(t.get('device_p99_us'))} | "
            f"{_fmt(t.get('host_mean_us'))} | {_fmt(dispatch)} | {lb_p} | {u_p} | {lb_e} | {u_e} |"
        )

    rf = data.get("routed_full")
    if rf and "derived" in rf:
        d = rf["derived"]
        lines += [
            "",
            f"- routed_full_compute_only_us: **{d.get('routed_full_compute_only_us', '-')}**",
            f"- routed_full_post_dispatch_us: **{d.get('routed_full_post_dispatch_us', '-')}**",
        ]

    gvl = data.get("grouped_vs_loop")
    if gvl:
        lines += ["", "### Grouped vs Loop (S2)", "", "| target | grouped_us | loop_us | grouped_dispatch | loop_dispatch |", "|--------|------------|---------|------------------|---------------|"]
        for r in gvl.get("results", []):
            g, l = r.get("grouped", {}), r.get("loop", {})
            lines.append(
                f"| {r.get('target')} | {_fmt(g.get('device_mean_us'))} | {_fmt(l.get('device_mean_us'))} | "
                f"{_fmt(g.get('dispatch_overhead_us'))} | {_fmt(l.get('dispatch_overhead_us'))} |"
            )

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else None
    if sweep_dir and sweep_dir.exists():
        lines += ["", "### n_active sweep (A5)", "", "| n_active | tpe | N | gemm_up_us | gemm_down_us | routed_full_us | util_eff |", "|---------:|----:|--:|-----------:|-------------:|---------------:|---------:|"]
        for na in cfg.sweep_n_active:
            sp = sweep_dir / f"n_active_{na}" / "routed_full.json"
            gp = sweep_dir / f"n_active_{na}" / "gemm_up.json"
            dp = sweep_dir / f"n_active_{na}" / "gemm_down.json"
            ro = _load(sp); go = _load(gp); do = _load(dp)
            n = na * cfg.tokens_per_expert
            scfg = apply_overrides(cfg, n_active_experts=na)
            up_b, dn_b = routed_weight_bytes(scfg)
            rf_us = _mean(ro)
            u_eff = utilizations(rf_us, up_b + dn_b, scfg)["util_vs_achievable"] if rf_us else None
            lines.append(
                f"| {na} | {cfg.tokens_per_expert} | {n} | {_fmt(_mean(go))} | {_fmt(_mean(do))} | {_fmt(rf_us)} | {_fmt(u_eff)} |"
            )
        lines += ["", "> act_quant(pre) 不随 n_active 变化（固定 [num_tokens, H]）。"]

    lines += [
        "",
        "> ⚠ **shared_expert ‖ routed_full 并行** (A8)：端到端 ≈ max(shared, routed)，不能相加。",
        "> ⚠ **util_vs_achievable > 0.75** 算打满 (N4)；表中 util_eff 列即 util_vs_achievable。",
        "> ⚠ **act_quant pre vs post** (N5)：post 是上限；pre ≈ post / n_active。",
        "",
        "### HBM 带宽实测 (N4)",
        f"- hbm_peak_tb_s = {cfg.roofline_hbm_peak_tb_s}",
        f"- hbm_effective_tb_s = {cfg.roofline_hbm_effective_tb_s}",
        "",
        "### FP4 vs W8 roofline (N8)",
        "| path | routed weight | @peak 1.6TB/s | @eff 1.0TB/s |",
        "|------|---------------|--------------|--------------|",
        "| W8 (本计划, n_active=6) | 144 MB | 90 μs | 144 μs |",
        "| FP4 (论文估计) | ~76 MB | ~48 μs | ~76 μs |",
        "| gap | ~1.9× HBM traffic | | |",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("summary", "comparison", "msprof-n-active-sweep"), default="summary")
    p.add_argument("--inputs", nargs="+", default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--sweep-dir", type=str, default="")
    p.add_argument("--msprof-jsons", nargs="+", default=None)
    p.add_argument("--event-jsons", nargs="+", default=None)
    args = p.parse_args(argv)

    if args.mode == "comparison":
        text = _comparison_mode(args)
    elif args.mode == "msprof-n-active-sweep":
        if not args.inputs:
            raise SystemExit("msprof-n-active-sweep requires --inputs")
        text = _msprof_n_active_sweep(args)
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
