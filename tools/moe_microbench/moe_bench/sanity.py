from __future__ import annotations

import torch


def sanity_check(name: str, o: torch.Tensor) -> dict:
    finite = torch.isfinite(o)
    all_finite = bool(finite.all().item())
    return {
        "kind": name,
        "shape": list(o.shape),
        "dtype": str(o.dtype),
        "nan_count": int((~finite).sum().item()) if o.numel() else 0,
        "has_nan": not all_finite,
        "mean": float(o.float().mean().item()) if all_finite else float("nan"),
        "std": float(o.float().std().item()) if all_finite else float("nan"),
        "abs_max": float(o.float().abs().max().item()) if all_finite else float("inf"),
    }
