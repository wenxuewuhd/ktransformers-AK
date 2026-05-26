from __future__ import annotations

import torch


def sanity_check(name: str, o: torch.Tensor) -> dict:
    finite = torch.isfinite(o)
    has_nan = bool((~finite).any().item())
    report = {
        "kind": name,
        "shape": list(o.shape),
        "dtype": str(o.dtype),
        "has_nan": has_nan,
    }
    if not has_nan:
        report["mean"] = float(o.float().mean().item())
        report["std"] = float(o.float().std().item())
        report["abs_max"] = float(o.float().abs().max().item())
    else:
        report["mean"] = float("nan")
        report["std"] = float("nan")
        report["abs_max"] = float("inf")
    return report


def require_sane(report: dict) -> None:
    if report.get("has_nan"):
        raise RuntimeError(f"sanity failed for {report['kind']}: NaN/Inf in output")
