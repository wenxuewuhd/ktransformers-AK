from __future__ import annotations

import sys
from pathlib import Path


def setup_pythonpath(repo_root: Path) -> None:
    sglang_py = repo_root / "third_party" / "sglang" / "python"
    microbench = repo_root / "tools" / "attn_microbench"
    for p in (str(microbench), str(sglang_py)):
        if p not in sys.path:
            sys.path.insert(0, p)


def init_custom_ops() -> None:
    import torch

    import torch_npu  # noqa: F401

    # DSv4 sparse attention 算子在 CANN custom_ops 包（与 p27 / deepseek_v4.py 一致），
    # 不是 sgl_kernel_npu.so 注册。
    import custom_ops  # noqa: F401

    try:
        import sgl_kernel_npu  # noqa: F401
    except ImportError:
        pass

    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    if not hasattr(torch.ops, "custom"):
        raise RuntimeError(
            "torch.ops.custom not registered — source env.sh (LD_LIBRARY_PATH + CANN set_env) "
            "then import custom_ops"
        )
    for name in (
        "npu_sparse_attn_sharedkv",
        "npu_sparse_attn_sharedkv_metadata",
        "npu_quant_lightning_indexer",
        "npu_quant_lightning_indexer_metadata",
    ):
        if not hasattr(torch.ops.custom, name):
            raise RuntimeError(f"missing torch.ops.custom.{name}")


def log_versions() -> None:
    import torch

    import torch_npu

    try:
        import custom_ops
    except ImportError as exc:
        print(f"[microbench][WARN] custom_ops import failed: {exc}")
        return

    try:
        import sgl_kernel_npu

        sk_ver = getattr(sgl_kernel_npu, "__version__", "unknown")
    except ImportError:
        sk_ver = "not installed"

    co_ver = getattr(custom_ops, "__version__", "unknown")
    print(f"[microbench] torch_npu={torch_npu.__version__}")
    print(f"[microbench] custom_ops={co_ver}")
    print(f"[microbench] sgl_kernel_npu={sk_ver}")
    print(
        "[microbench] custom ops: "
        f"sparse_attn={hasattr(torch.ops.custom, 'npu_sparse_attn_sharedkv')}, "
        f"li={hasattr(torch.ops.custom, 'npu_quant_lightning_indexer')}"
    )
    visible = __import__("os").environ.get("ASCEND_RT_VISIBLE_DEVICES", "(all)")
    print(f"[microbench] ASCEND_RT_VISIBLE_DEVICES={visible}")


def require_npu():
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU not available")
    return torch.device("npu:0")
