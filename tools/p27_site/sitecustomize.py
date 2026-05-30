"""Auto-imported when tools/p27_site is first on PYTHONPATH."""

from __future__ import annotations

import builtins


def _patch_server_args() -> None:
    try:
        from sglang.srt.server_args import ServerArgs

        if hasattr(ServerArgs, "get_hf_config"):
            return
        ServerArgs.get_hf_config = lambda self: self.get_model_config().hf_config  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_server_args()

_orig_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
    mod = _orig_import(name, globals, locals, fromlist, level)
    if name == "sglang.srt.server_args" or (
        fromlist and "server_args" in fromlist and name.startswith("sglang")
    ):
        _patch_server_args()
    return mod


builtins.__import__ = _import
