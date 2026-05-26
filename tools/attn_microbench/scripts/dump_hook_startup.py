"""PYTHONSTARTUP: lazy-patch Indexer.forward_npu_dsv4_fusion when nsa_indexer loads."""
from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

OUT = os.environ.get("ATTN_DUMP_INDEXER_PATH")
if not OUT:
    raise SystemExit(0)

_out_path = Path(OUT)
_dumped = False
_orig_import = builtins.__import__


def _encode(v):
    import torch

    if isinstance(v, torch.Tensor):
        return {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "device": str(v.device),
            "sample": v.flatten()[:8].tolist() if v.numel() else [],
        }
    return str(v)


def _patch_module(mod) -> None:
    global _dumped
    cls = mod.Indexer
    if getattr(cls.forward_npu_dsv4_fusion, "_attn_dump_patched", False):
        return
    orig = cls.forward_npu_dsv4_fusion

    def wrapped(self, q, k, k_scale, weights, forward_batch):
        global _dumped
        if not _dumped:
            import torch
            import torch_npu

            q_dq, q_scale = torch_npu.npu_dynamic_quant(q)
            li_quant_metadata = forward_batch.attn_backend.forward_metadata.kernel_metadata[
                "li_quant_metadata"
            ]
            kwargs = {
                "query": q_dq,
                "key": k,
                "key_dequant_scale": k_scale.squeeze(-2),
                "actual_seq_lengths_query": forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q,
                "actual_seq_lengths_key": forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv,
                "block_table": forward_batch.attn_backend.forward_metadata.c4_page_table,
                "layout_query": "TND",
                "layout_key": "PA_BSND",
                "weights": weights.to(torch.float16),
                "query_dequant_scale": q_scale.to(torch.float16),
                "cmp_ratio": 4,
                "query_quant_mode": 0,
                "key_quant_mode": 0,
                "sparse_mode": 3,
                "sparse_count": self.index_topk,
                "metadata": li_quant_metadata,
            }
            payload = {
                "source": "production_forward_npu_dsv4_fusion",
                "kwargs": {k: _encode(v) for k, v in kwargs.items()},
            }
            _out_path.parent.mkdir(parents=True, exist_ok=True)
            _out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _dumped = True
        return orig(self, q, k, k_scale, weights, forward_batch)

    wrapped._attn_dump_patched = True  # type: ignore[attr-defined]
    cls.forward_npu_dsv4_fusion = wrapped


def _hook_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    if name == "sglang.srt.layers.attention.nsa.nsa_indexer" or (
        fromlist and "nsa_indexer" in fromlist and hasattr(mod, "Indexer")
    ):
        try:
            _patch_module(mod)
        except Exception:
            pass
    return mod


builtins.__import__ = _hook_import
