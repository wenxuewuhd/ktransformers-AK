from __future__ import annotations

from dataclasses import dataclass

import torch

from attn_bench.config import BenchConfig
from attn_bench.page_table import (
    PageTableSpec,
    build_c128_page_table,
    build_c4_page_table,
    build_swa_page_table,
    compute_page_spec,
)


@dataclass
class SyntheticTensors:
    q: torch.Tensor
    sinks: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv_c4: torch.Tensor
    cmp_kv_c128: torch.Tensor
    swa_page_table: torch.Tensor
    c4_page_table: torch.Tensor
    c128_page_table: torch.Tensor
    seqused_kv: torch.Tensor
    cu_seqlens_q_pa: torch.Tensor
    actual_seq_lengths_q: torch.Tensor
    li_query: torch.Tensor
    li_key: torch.Tensor
    li_key_scale: torch.Tensor
    li_weights: torch.Tensor
    page_spec: PageTableSpec


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "int8":
        return torch.int8
    return torch.float32


def build_topk_random(
    cfg: BenchConfig, c4_cols: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """[B, index_topk] int32 valid indices in [0, c4_cols)."""
    k = min(cfg.index_topk, c4_cols)
    idx = torch.randint(
        0, c4_cols, (cfg.batch_size, k), generator=generator, device=device
    )
    if k < cfg.index_topk:
        pad = torch.full(
            (cfg.batch_size, cfg.index_topk - k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        idx = torch.cat([idx, pad], dim=1)
    return idx.to(torch.int32)


def build_synthetic(cfg: BenchConfig, *, swa_no_sink: bool = True) -> SyntheticTensors:
    device = torch.device(cfg.device)
    dtype = _dtype_from_name(cfg.dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    c4_override = cfg.diag.get("override_c4_cols")
    spec = compute_page_spec(
        cfg.seq_len,
        cfg.page_size,
        cfg.batch_size,
        c4_cols_override=int(c4_override) if c4_override is not None else None,
    )
    num_tokens = cfg.num_tokens

    q = torch.randn(
        (num_tokens, cfg.num_heads_q, cfg.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    )
    if swa_no_sink:
        sinks = torch.full(
            (cfg.num_heads_q,), float("-inf"), dtype=torch.float32, device=device
        )
    else:
        sinks = torch.zeros(cfg.num_heads_q, dtype=torch.float32, device=device)

    ori_kv = torch.randn(
        (spec.swa_num_pages, cfg.page_size, cfg.num_heads_kv, cfg.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    )
    cmp_kv_c4 = torch.randn(
        (spec.c4_num_pages, cfg.page_size, cfg.num_heads_kv, cfg.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    )
    cmp_kv_c128 = torch.randn(
        (spec.c128_num_pages, cfg.page_size, cfg.num_heads_kv, cfg.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    )

    swa_page_table = build_swa_page_table(spec, device)
    c4_page_table = build_c4_page_table(spec, device, cfg)
    c128_page_table = build_c128_page_table(spec, device, cfg)

    seqused_kv = torch.full(
        (cfg.batch_size,), cfg.effective_seqused_kv(), dtype=torch.int32, device=device
    )
    cu_seqlens_q_pa = torch.arange(
        0, cfg.batch_size + 1, dtype=torch.int32, device=device
    )
    actual_seq_lengths_q = torch.arange(
        1, cfg.batch_size + 1, dtype=torch.int32, device=device
    )

    li_query = torch.randn(
        (num_tokens, cfg.index_n_heads, cfg.index_head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    )
    li_key = torch.randint(
        -127,
        128,
        (spec.c4_num_pages, cfg.page_size, cfg.num_heads_kv, cfg.index_head_dim),
        dtype=torch.int8,
        device=device,
        generator=gen,
    )
    li_key_scale = torch.full(
        (spec.c4_num_pages, cfg.page_size, cfg.num_heads_kv, 1),
        0.01,
        dtype=torch.float16,
        device=device,
    )
    li_weights = torch.randn(
        (num_tokens, cfg.index_n_heads),
        dtype=torch.float16,
        device=device,
        generator=gen,
    )

    return SyntheticTensors(
        q=q,
        sinks=sinks,
        ori_kv=ori_kv,
        cmp_kv_c4=cmp_kv_c4,
        cmp_kv_c128=cmp_kv_c128,
        swa_page_table=swa_page_table,
        c4_page_table=c4_page_table,
        c128_page_table=c128_page_table,
        seqused_kv=seqused_kv,
        cu_seqlens_q_pa=cu_seqlens_q_pa,
        actual_seq_lengths_q=actual_seq_lengths_q,
        li_query=li_query,
        li_key=li_key,
        li_key_scale=li_key_scale,
        li_weights=li_weights,
        page_spec=spec,
    )


def assert_shapes(cfg: BenchConfig, t: SyntheticTensors) -> None:
    spec = t.page_spec
    num_tokens = cfg.num_tokens

    assert t.q.shape == (num_tokens, cfg.num_heads_q, cfg.head_dim), t.q.shape
    assert t.sinks.shape == (cfg.num_heads_q,)
    assert t.ori_kv.shape == (
        spec.swa_num_pages,
        cfg.page_size,
        cfg.num_heads_kv,
        cfg.head_dim,
    )
    assert t.cmp_kv_c4.shape == (
        spec.c4_num_pages,
        cfg.page_size,
        cfg.num_heads_kv,
        cfg.head_dim,
    )
    assert t.cmp_kv_c128.shape == (
        spec.c128_num_pages,
        cfg.page_size,
        cfg.num_heads_kv,
        cfg.head_dim,
    )
    assert t.swa_page_table.shape == (cfg.batch_size, spec.swa_cols)
    assert int(t.swa_page_table.max()) < spec.swa_num_pages, (
        f"swa_page_table max {t.swa_page_table.max()} >= {spec.swa_num_pages}"
    )
    assert t.c4_page_table.shape == (cfg.batch_size, spec.c4_num_pages), t.c4_page_table.shape
    assert int(t.c4_page_table.max()) < spec.c4_num_pages
    assert t.c128_page_table.shape == (cfg.batch_size, spec.c128_num_pages), (
        t.c128_page_table.shape
    )
    assert int(t.c128_page_table.max()) < spec.c128_num_pages
    assert t.seqused_kv.shape == (cfg.batch_size,)
    expected_seqused = cfg.effective_seqused_kv()
    assert int(t.seqused_kv[0]) == expected_seqused, (
        f"seqused_kv {t.seqused_kv[0]} != expected {expected_seqused}"
    )
    assert t.cu_seqlens_q_pa.tolist()[0] == 0
    assert t.cu_seqlens_q_pa.tolist()[-1] == cfg.batch_size
    assert t.li_key.shape == (
        spec.c4_num_pages,
        cfg.page_size,
        cfg.num_heads_kv,
        cfg.index_head_dim,
    )
    assert t.li_key.dtype == torch.int8
    assert t.li_key_scale.shape == (
        spec.c4_num_pages,
        cfg.page_size,
        cfg.num_heads_kv,
        1,
    )
    assert t.li_query.shape == (num_tokens, cfg.index_n_heads, cfg.index_head_dim)
