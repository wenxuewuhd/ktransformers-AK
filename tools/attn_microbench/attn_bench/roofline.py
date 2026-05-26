"""Roofline helpers for attention microbench (HBM lower-bound estimates)."""

from __future__ import annotations

from attn_bench.config import BenchConfig


def effective_tb_s(cfg: BenchConfig) -> float:
    return float(cfg.roofline.get("hbm_effective_tb_s", 1.0))


def swa_kv_bytes(cfg: BenchConfig) -> int:
    """SWA window: 128 tokens × 1 kv head × 512 dim × 2 B."""
    win = min(cfg.sliding_window_size, cfg.seq_len)
    return win * cfg.num_heads_kv * cfg.head_dim * 2


def indexer_kv_bytes(cfg: BenchConfig) -> int:
    """Indexer int8 KV: c4_cols × 1 × index_head_dim × 1 B."""
    c4_cols = cfg.effective_c4_cols()
    return c4_cols * cfg.num_heads_kv * cfg.index_head_dim * 1


def csa_attn_cmp_kv_bytes(cfg: BenchConfig) -> int:
    """CSA compressed attn bf16 KV read: c4_cols × 1 × head_dim × 2 B."""
    c4_cols = cfg.effective_c4_cols()
    return c4_cols * cfg.num_heads_kv * cfg.head_dim * 2


def hca_cmp_kv_bytes(cfg: BenchConfig) -> int:
    """HCA compressed attn bf16 KV: c128_cols × 1 × head_dim × 2 B."""
    c128_cols = max(1, cfg.seq_len // 128)
    return c128_cols * cfg.num_heads_kv * cfg.head_dim * 2


def hw_lower_bound_us(num_bytes: int, tb_s: float = 1.0) -> float:
    """Minimum device time (µs) if memory-bound at effective HBM bandwidth."""
    if num_bytes <= 0 or tb_s <= 0:
        return 0.0
    return num_bytes / (tb_s * 1e6)


def util_vs_achievable(device_us: float, lower_bound_us: float) -> float:
    """Fraction of peak HBM bandwidth achieved (lb / device), capped at 1."""
    if device_us <= 0 or lower_bound_us <= 0:
        return 0.0
    return min(1.0, lower_bound_us / device_us)
