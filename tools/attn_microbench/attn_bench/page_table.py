from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from attn_bench.config import BenchConfig


@dataclass(frozen=True)
class PageTableSpec:
    seq_len: int
    page_size: int
    batch_size: int
    swa_cols: int
    c4_cols: int
    c128_cols: int
    swa_num_pages: int
    c4_num_pages: int
    c128_num_pages: int


def compute_page_spec(
    seq_len: int,
    page_size: int,
    batch_size: int = 1,
    *,
    c4_cols_override: int | None = None,
) -> PageTableSpec:
    """Synthetic decode page geometry aligned with ascend_backend init_forward_metadata."""
    swa_cols = seq_len
    c4_cols = c4_cols_override if c4_cols_override is not None else max(1, seq_len // 4)
    c128_cols = max(1, seq_len // 128)

    swa_num_pages = max(1, (2 * batch_size * page_size) // page_size)
    c4_num_pages = max(1, math.ceil(c4_cols / page_size))
    c128_num_pages = max(1, math.ceil(c128_cols / page_size))

    return PageTableSpec(
        seq_len=seq_len,
        page_size=page_size,
        batch_size=batch_size,
        swa_cols=swa_cols,
        c4_cols=c4_cols,
        c128_cols=c128_cols,
        swa_num_pages=swa_num_pages,
        c4_num_pages=c4_num_pages,
        c128_num_pages=c128_num_pages,
    )


def build_swa_page_table(spec: PageTableSpec, device: torch.device) -> torch.Tensor:
    table = torch.zeros(
        (spec.batch_size, spec.swa_cols), dtype=torch.int32, device=device
    )
    win = min(spec.page_size * spec.swa_num_pages, spec.seq_len)
    start = spec.seq_len - win
    rel = torch.arange(win, device=device, dtype=torch.int32)
    table[0, start:] = (rel // spec.page_size) % spec.swa_num_pages
    return table


def _strided_page_table(
    spec: PageTableSpec,
    device: torch.device,
    cols: int,
    num_pages: int,
    *,
    unique_pages: bool,
) -> torch.Tensor:
    """Build strided block_table like ascend_backend (cache loc → page id).

    Production: ``table[:, ::page_size] // page_size`` after filling cache locs.
    """
    pos = torch.arange(cols, device=device, dtype=torch.int32)
    if unique_pages:
        page_id = (pos // spec.page_size).clamp(max=num_pages - 1)
    else:
        page_id = (pos // spec.page_size) % num_pages
    cache_loc = page_id * spec.page_size + (pos % spec.page_size)
    if spec.page_size > 1:
        table = cache_loc[:: spec.page_size] // spec.page_size
    else:
        table = cache_loc
    return table.unsqueeze(0)


def build_c4_page_table(
    spec: PageTableSpec,
    device: torch.device,
    cfg: BenchConfig | None = None,
) -> torch.Tensor:
    """[B, c4_num_pages] int32 — strided physical page ids (ascend_backend decode)."""
    unique = bool(cfg is not None and cfg.diag.get("page_table_unique_pages", False))
    return _strided_page_table(
        spec, device, spec.c4_cols, spec.c4_num_pages, unique_pages=unique
    )


def build_c128_page_table(
    spec: PageTableSpec,
    device: torch.device,
    cfg: BenchConfig | None = None,
) -> torch.Tensor:
    """[B, c128_num_pages] int32 — strided physical page ids."""
    unique = bool(cfg is not None and cfg.diag.get("page_table_unique_pages", False))
    return _strided_page_table(
        spec, device, spec.c128_cols, spec.c128_num_pages, unique_pages=unique
    )
