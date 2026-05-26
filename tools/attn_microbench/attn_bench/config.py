from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchConfig:
    seq_len: int
    page_size: int
    batch_size: int
    q_len: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    sliding_window_size: int
    warmup: int
    repeat: int
    seed: int
    device: str
    dtype: str
    dtypes: dict[str, str]
    softmax_scale: float
    indexer_kv_dtype: str
    indexer_weight_dtype: str
    swa_layer_id: int
    csa_layer_id: int
    hca_layer_id: int
    metadata_keys: dict[str, Any] = field(default_factory=dict)
    invariants: dict[str, Any] = field(default_factory=dict)
    diag: dict[str, Any] = field(default_factory=dict)
    roofline: dict[str, Any] = field(default_factory=dict)
    msprof: dict[str, Any] = field(default_factory=dict)
    quick_mode_repeat: int = 100
    quick_mode_warmup: int = 10

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.q_len

    def effective_seqused_kv(self) -> int:
        v = self.diag.get("override_seqused_kv")
        return int(v) if v is not None else self.seq_len

    def effective_c4_cols(self) -> int:
        v = self.diag.get("override_c4_cols")
        if v is not None:
            return int(v)
        return max(1, self.seq_len // 4)


def microbench_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return microbench_root().parent.parent


def default_config_path() -> Path:
    return microbench_root() / "config" / "dsv4_flash.yaml"


def _merge_diag(base: dict, overrides: dict | None) -> dict:
    out = dict(base)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                out[k] = v
    # env overrides (P1 CLI)
    if os.environ.get("DIAG_OVERRIDE_C4_COLS"):
        out["override_c4_cols"] = int(os.environ["DIAG_OVERRIDE_C4_COLS"])
    if os.environ.get("DIAG_OVERRIDE_SEQUSED_KV"):
        out["override_seqused_kv"] = int(os.environ["DIAG_OVERRIDE_SEQUSED_KV"])
    if os.environ.get("DIAG_PAGE_TABLE_UNIQUE", "").lower() in ("1", "true", "yes"):
        out["page_table_unique_pages"] = True
    return out


def load_config(path: str | Path | None = None, diag_overrides: dict | None = None) -> BenchConfig:
    cfg_path = Path(path) if path else default_config_path()
    with open(cfg_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    model = raw["model"]
    runtime = raw["runtime"]
    bench = raw["bench"]
    layers = raw.get("layers", {})
    metadata_keys = raw.get("metadata_keys", {})
    invariants = raw.get("invariants", {})
    diag = _merge_diag(raw.get("diag", {}), diag_overrides)
    roofline = raw.get("roofline", {})
    msprof = raw.get("msprof", {})
    dtypes = runtime.get("dtypes", {})

    dtype = dtypes.get("q", runtime.get("dtype", "bfloat16"))
    softmax_scale = float(
        model.get("softmax_scale", 1.0 / math.sqrt(int(model["head_dim"])))
    )

    return BenchConfig(
        seq_len=int(bench["seq_len"]),
        page_size=int(runtime["page_size"]),
        batch_size=int(runtime["batch_size"]),
        q_len=int(bench.get("q_len", 1)),
        num_heads_q=int(model["num_heads_q"]),
        num_heads_kv=int(model["num_heads_kv"]),
        head_dim=int(model["head_dim"]),
        index_n_heads=int(model["index_n_heads"]),
        index_head_dim=int(model["index_head_dim"]),
        index_topk=int(model["index_topk"]),
        sliding_window_size=int(model["sliding_window_size"]),
        warmup=int(bench["warmup"]),
        repeat=int(bench["repeat"]),
        seed=int(bench["seed"]),
        device=str(runtime["device"]),
        dtype=str(dtype),
        dtypes=dict(dtypes) if dtypes else {"q": str(dtype)},
        softmax_scale=softmax_scale,
        indexer_kv_dtype=str(dtypes.get("indexer_kv", runtime.get("indexer_kv_dtype", "int8"))),
        indexer_weight_dtype=str(
            dtypes.get("indexer_weight", runtime.get("indexer_weight_dtype", "float16"))
        ),
        swa_layer_id=int(layers.get("swa_layer_id", 0)),
        csa_layer_id=int(layers.get("csa_layer_id", 2)),
        hca_layer_id=int(layers.get("hca_layer_id", 3)),
        metadata_keys=dict(metadata_keys),
        invariants=dict(invariants),
        diag=diag,
        roofline=dict(roofline),
        msprof=dict(msprof),
        quick_mode_repeat=int(bench.get("quick_mode_repeat", 100)),
        quick_mode_warmup=int(bench.get("quick_mode_warmup", 10)),
    )


def apply_overrides(cfg: BenchConfig, **kwargs: Any) -> BenchConfig:
    data = cfg.__dict__.copy()
    diag_extra = kwargs.pop("diag", None)
    for k, v in kwargs.items():
        if v is not None and k in data:
            data[k] = v
    if diag_extra:
        merged = dict(data.get("diag", {}))
        merged.update({k: v for k, v in diag_extra.items() if v is not None})
        data["diag"] = merged
    return BenchConfig(**data)
