from __future__ import annotations

import torch

from attn_bench.config import BenchConfig
from attn_bench.metadata import KernelMetadata
from attn_bench.synthetic import SyntheticTensors


def _attn_common(t: SyntheticTensors, cfg: BenchConfig) -> dict:
    return {
        "q": t.q,
        "ori_kv": t.ori_kv,
        "ori_block_table": t.swa_page_table,
        "sinks": t.sinks,
        "softmax_scale": cfg.softmax_scale,
        "cu_seqlens_q": t.cu_seqlens_q_pa,
        "seqused_kv": t.seqused_kv,
        "ori_mask_mode": 4,
        "ori_win_left": cfg.sliding_window_size - 1,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
    }


def run_swa_attn(t: SyntheticTensors, meta: KernelMetadata, cfg: BenchConfig) -> torch.Tensor:
    out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
        metadata=meta.c1a,
        **_attn_common(t, cfg),
    )
    return out


def run_hca_attn(t: SyntheticTensors, meta: KernelMetadata, cfg: BenchConfig) -> torch.Tensor:
    out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
        metadata=meta.c128a,
        cmp_ratio=128,
        cmp_mask_mode=3,
        cmp_kv=t.cmp_kv_c128,
        cmp_sparse_indices=None,
        cmp_block_table=t.c128_page_table,
        **_attn_common(t, cfg),
    )
    return out


def run_csa_indexer(
    t: SyntheticTensors, meta: KernelMetadata, cfg: BenchConfig
) -> torch.Tensor:
    q, q_scale = torch_npu_dynamic_quant(t.li_query)
    topk, _ = torch.ops.custom.npu_quant_lightning_indexer(
        query=q,
        key=t.li_key,
        key_dequant_scale=t.li_key_scale.squeeze(-2),
        actual_seq_lengths_query=t.actual_seq_lengths_q,
        actual_seq_lengths_key=t.seqused_kv,
        block_table=t.c4_page_table,
        layout_query="TND",
        layout_key="PA_BSND",
        weights=t.li_weights.to(torch.float16),
        query_dequant_scale=q_scale.to(torch.float16),
        cmp_ratio=4,
        query_quant_mode=0,
        key_quant_mode=0,
        sparse_mode=3,
        sparse_count=cfg.index_topk,
        metadata=meta.li_quant,
    )
    return topk.view(-1, cfg.index_topk)


def torch_npu_dynamic_quant(x: torch.Tensor):
    import torch_npu

    return torch_npu.npu_dynamic_quant(x)


def run_csa_attn(
    t: SyntheticTensors,
    meta: KernelMetadata,
    cfg: BenchConfig,
    topk: torch.Tensor,
) -> torch.Tensor:
    topk_view = topk.view(-1, 1, cfg.index_topk)
    out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(
        metadata=meta.c4a,
        cmp_ratio=4,
        cmp_mask_mode=3,
        cmp_kv=t.cmp_kv_c4,
        cmp_sparse_indices=topk_view,
        cmp_block_table=t.c4_page_table,
        **_attn_common(t, cfg),
    )
    return out
