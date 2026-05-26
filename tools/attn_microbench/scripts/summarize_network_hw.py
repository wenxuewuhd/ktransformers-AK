#!/usr/bin/env python3
"""Regenerate P1.7 network hardware estimate from msprof JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MICRO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = MICRO_ROOT / "results"

# Main-stack layer counts (compress_ratios; L43 excluded — TBD)
N_SWA = 2
N_CSA = 20
N_HCA = 20
N_LAYERS = N_SWA + N_CSA + N_HCA  # 42


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hw_mean(data: dict, key: str = "attn_hw") -> tuple[float, float]:
    block = data.get(key) or {}
    return float(block.get("device_mean_us", 0)), float(block.get("device_std_us", 0))


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize network attention hw from msprof JSON")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--out", type=Path, default=RESULTS / "network_hw_estimate.md")
    args = p.parse_args()
    rd = args.results_dir

    swa = _load(rd / "swa_msprof.json")
    hca = _load(rd / "hca_msprof.json")
    swa_us, swa_std = _hw_mean(swa)
    hca_us, hca_std = _hw_mean(hca)

    seq_lens = [1024, 4096, 8192, 16384, 32768]
    sweep_rows = []
    for s in seq_lens:
        path = rd / f"csa_msprof_seq_{s}.json"
        if not path.is_file():
            continue
        d = _load(path)
        i_us, i_std = _hw_mean(d, "indexer_hw")
        a_us, a_std = _hw_mean(d, "attn_hw")
        csa_layer = i_us + a_us
        total = N_SWA * swa_us + N_CSA * csa_layer + N_HCA * hca_us
        sweep_rows.append(
            {
                "seq_len": s,
                "indexer_hw": (i_us, i_std),
                "attn_hw": (a_us, a_std),
                "csa_layer": csa_layer,
                "network_hw_us": total,
            }
        )

    base_total = sweep_rows[0]["network_hw_us"] if sweep_rows else 0

    # Python Event @ 32k (optional)
    py_32k = None
    csa_py = rd / "seq_32768" / "csa.json"
    swa_py = rd / "seq_32768" / "swa.json"
    hca_py = rd / "seq_32768" / "hca.json"
    if all(p.is_file() for p in (csa_py, swa_py, hca_py)):
        csa = _load(csa_py)
        swa_p = _load(swa_py)
        hca_p = _load(hca_py)
        py_32k = (
            N_SWA * swa_p["attn_us"]["mean"]
            + N_CSA * (csa["indexer_us"]["mean"] + csa["attn_us"]["mean"])
            + N_HCA * hca_p["attn_us"]["mean"]
        )

    r32 = sweep_rows[-1] if sweep_rows else None
    lines = [
        "# Network Attention Hardware Estimate (auto-generated)",
        "",
        "> 由 `scripts/summarize_network_hw.py` 从 `results/*_msprof*.json` 生成。",
        "> **完整复现 / 解析说明**：见 [`P1_7_analysis_guide.md`](./P1_7_analysis_guide.md)",
        "> **重新生成**：`python scripts/summarize_network_hw.py --out results/network_hw_estimate.md`",
        "> 层数：SWA×2 + CSA×20 + HCA×20 = 42（不含 L43）。",
        "> **@32k CSA 用 seq sweep 的 `csa_msprof_seq_32768.json`**（与 `csa_msprof.json` 独立两次 run，差 ~5% 正常）。",
        "",
        "## 单层硬件 @ 32k（msprof kernel_details）",
        "",
        "| 层类型 | Indexer (µs) | Attn (µs) | 层合计 (µs) |",
        "|--------|--------------|-----------|-------------|",
    ]
    if r32:
        i, _ = r32["indexer_hw"]
        a, _ = r32["attn_hw"]
        lines.append(f"| SWA | — | {swa_us:.1f} ± {swa_std:.1f} | {swa_us:.1f} |")
        lines.append(f"| CSA | {i:.1f} | {a:.1f} | {i + a:.1f} |")
        lines.append(f"| HCA | — | {hca_us:.1f} ± {hca_std:.1f} | {hca_us:.1f} |")

    if r32:
        total = r32["network_hw_us"]
        lines.extend(
            [
                "",
                "## 整网 @ 32k",
                "",
                "```",
                f"Total = {N_SWA}×SWA + {N_CSA}×CSA + {N_HCA}×HCA",
                f"      = {N_SWA}×{swa_us:.1f} + {N_CSA}×{r32['csa_layer']:.1f} + {N_HCA}×{hca_us:.1f}",
                f"      ≈ {total/1000:.2f} ms",
                "```",
                "",
                "| 组成部分 | 耗时 | 占比 |",
                "|----------|------|------|",
            ]
        )
        parts = [
            ("SWA×2", N_SWA * swa_us),
            ("CSA indexer×20", N_CSA * r32["indexer_hw"][0]),
            ("CSA attn×20", N_CSA * r32["attn_hw"][0]),
            ("HCA attn×20", N_HCA * hca_us),
        ]
        for name, us in parts:
            lines.append(f"| {name} | {us:.0f} µs | {us/total*100:.1f}% |")
        lines.append(f"| **合计** | **{total:.0f} µs** | 100% |")

    lines.extend(["", "## Seq sweep（CSA 可变，SWA/HCA 用 32k 常数）", ""])
    lines.append("| seq_len | indexer_hw | attn_hw | CSA×20 | 整网 hw | vs 1024 |")
    lines.append("|---------|------------|---------|--------|---------|---------|")
    for row in sweep_rows:
        i, is_ = row["indexer_hw"]
        a, as_ = row["attn_hw"]
        sc = row["network_hw_us"] / base_total if base_total else 1.0
        lines.append(
            f"| {row['seq_len']} | {i:.1f}±{is_:.1f} | {a:.1f}±{as_:.1f} | "
            f"{N_CSA * row['csa_layer']:.0f} µs | {row['network_hw_us']/1000:.2f} ms | {sc:.2f}× |"
        )

    if py_32k and r32:
        lines.extend(
            [
                "",
                "## Python Event @ 32k（对比）",
                "",
                f"- 硬件：{r32['network_hw_us']/1000:.2f} ms",
                f"- Python Event（repeat=1000）：{py_32k/1000:.2f} ms",
                f"- Launch overhead：{(py_32k - r32['network_hw_us'])/py_32k*100:.1f}%",
            ]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summarize] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
