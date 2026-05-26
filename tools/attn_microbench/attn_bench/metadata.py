from __future__ import annotations

from dataclasses import dataclass

import torch

from attn_bench.config import BenchConfig
from attn_bench.synthetic import SyntheticTensors


@dataclass
class KernelMetadata:
    c1a: torch.Tensor
    c4a: torch.Tensor
    c128a: torch.Tensor
    li_quant: torch.Tensor


def build_metadata(cfg: BenchConfig, t: SyntheticTensors) -> KernelMetadata:
    mk = cfg.metadata_keys
    fa_common = {
        "cu_seqlens_q": t.cu_seqlens_q_pa,
        "seqused_kv": t.seqused_kv,
        "ori_mask_mode": int(mk.get("ori_mask_mode", 4)),
        "cmp_mask_mode": int(mk.get("cmp_mask_mode", 3)),
        "ori_win_left": cfg.sliding_window_size - 1,
        "ori_win_right": 0,
        "layout_q": str(mk.get("layout_q", "TND")),
        "layout_kv": str(mk.get("layout_kv", "PA_ND")),
    }
    base = {
        "batch_size": cfg.batch_size,
        "num_heads_q": cfg.num_heads_q,
        "num_heads_kv": cfg.num_heads_kv,
        "head_dim": cfg.head_dim,
        "has_ori_kv": True,
        **fa_common,
    }

    c1a = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
        has_cmp_kv=False,
        cmp_ratio=1,
        **base,
    )
    c4a = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
        has_cmp_kv=True,
        cmp_ratio=4,
        cmp_topk=cfg.index_topk,
        **base,
    )
    c128a = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
        has_cmp_kv=True,
        cmp_ratio=128,
        **base,
    )
    li_quant = torch.ops.custom.npu_quant_lightning_indexer_metadata(
        device=str(t.q.device),
        actual_seq_lengths_query=t.actual_seq_lengths_q,
        actual_seq_lengths_key=t.seqused_kv,
        layout_key=str(mk.get("layout_li_key", "PA_BSND")),
        sparse_count=cfg.index_topk,
        sparse_mode=int(mk.get("sparse_mode", 3)),
        layout_query=str(mk.get("layout_q", "TND")),
        cmp_ratio=4,
        key_quant_mode=int(mk.get("key_quant_mode", 0)),
        query_quant_mode=int(mk.get("query_quant_mode", 0)),
        num_heads_q=cfg.index_n_heads,
        num_heads_k=cfg.num_heads_kv,
        head_dim=cfg.index_head_dim,
    )
    return KernelMetadata(c1a, c4a, c128a, li_quant)
