from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def setup_pythonpath(repo_root: Path) -> None:
    sglang_py = repo_root / "third_party" / "sglang" / "python"
    microbench = repo_root / "tools" / "moe_microbench"
    for p in (str(microbench), str(sglang_py)):
        if p not in sys.path:
            sys.path.insert(0, p)


def init_custom_ops() -> None:
    import torch_npu  # noqa: F401

    try:
        import sgl_kernel_npu  # noqa: F401
    except ImportError:
        print("[moe_microbench] sgl_kernel_npu not installed (ok if using torch_npu.* only)")

    assert hasattr(torch_npu, "npu_dynamic_quant"), "torch_npu.npu_dynamic_quant missing"
    assert hasattr(torch_npu, "npu_grouped_matmul"), "torch_npu.npu_grouped_matmul missing"


def require_npu():
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU not available; check ASCEND_RT_VISIBLE_DEVICES and driver")
    return torch.device("npu:0")


def log_versions() -> None:
    import torch_npu

    print(f"[moe_microbench] torch_npu={torch_npu.__version__}")
    try:
        import sgl_kernel_npu

        print(f"[moe_microbench] sgl_kernel_npu={getattr(sgl_kernel_npu, '__version__', 'unknown')}")
    except ImportError:
        pass


def print_npu_smi() -> None:
    """Best-effort: 打印 npu-smi info 简表 + ASCEND_RT_VISIBLE_DEVICES (N9)."""
    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>")
    print(f"[moe_microbench] ASCEND_RT_VISIBLE_DEVICES={vis}")
    if shutil.which("npu-smi") is None:
        print("[moe_microbench] npu-smi not on PATH; cannot show device map")
        return
    try:
        out = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
        print("[moe_microbench] npu-smi info (head):")
        print(out[:2000])
    except Exception as e:
        print(f"[moe_microbench] npu-smi info failed: {e}")
