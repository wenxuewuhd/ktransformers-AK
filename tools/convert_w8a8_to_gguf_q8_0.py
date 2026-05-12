#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash W8A8 routed-expert weights to GGUF Q8_0.

This is Phase 1.1 of the NPU + CPU expert offload project. The output GGUF files
are consumed by kt-kernel's LlamafileMoEWrapper (which loads tensors named
``blk.{layer}.ffn_{gate,up,down}_exps.weight`` via
``GGUFLoader.get_undequanted_tensor_and_ggml_type``).

INPUT (DSv4-Flash W8A8 safetensors layout — verified against shard 22):
    layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight        int8 (out, in)
    layers.{L}.ffn.experts.{E}.{w1,w2,w3}.weight_scale  fp32 (out, 1)

Alias convention (kt-kernel internal):
    w1 = gate_proj   shape (moe_intermediate_size, hidden_size)        = (2048, 4096)
    w3 = up_proj     shape (moe_intermediate_size, hidden_size)        = (2048, 4096)
    w2 = down_proj   shape (hidden_size, moe_intermediate_size)        = (4096, 2048)

OUTPUT (one .gguf per layer):
    <dst>/dsv4_flash_q8_0-layer_{L:03d}.gguf
        tensors:
            blk.{L}.ffn_gate_exps.weight  Q8_0  shape (E, 2048, 4096)
            blk.{L}.ffn_up_exps.weight    Q8_0  shape (E, 2048, 4096)
            blk.{L}.ffn_down_exps.weight  Q8_0  shape (E, 4096, 2048)
    <dst>/manifest.json
        per-(layer, expert, tensor) records with cosine_sim, max_abs_diff, bytes.

CONVERSION PIPELINE (per expert tensor):
    int8 W8A8 weight + fp32 per-channel scale
        → fp32 dequant  (weight.astype(f32) * scale, broadcast (N,1)*(N,M))
        → Q8_0 quantize (gguf.quants.Q8_0.quantize — matches llama.cpp byte-for-byte)
        → Q8_0 dequant  (for cosine-sim verification only)
    cosine(W8A8_dequant_fp32, Q8_0_dequant_fp32) ≥ --verify-cosine (default 0.9995).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from safetensors import safe_open

# Make this script runnable both as a module (`python -m tools.convert...`) and as a
# standalone file (`python tools/convert_...py`).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _w8a8_dequant import dequant_w8a8_numpy  # noqa: E402

# gguf is from `pip install gguf` — provides byte-for-byte reference Q8_0 quantize.
from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402
from gguf.quants import Q8_0  # noqa: E402

logger = logging.getLogger("convert_w8a8_to_gguf_q8_0")


# ---------------------------------------------------------------------------
# Constants — sourced from DSv4-Flash config.json
# ---------------------------------------------------------------------------

DEFAULT_NUM_LAYERS = 43
DEFAULT_NUM_EXPERTS = 256
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_MOE_INTER = 2048
EXPERT_TENSOR_NAMES = ("w1", "w3", "w2")  # gate, up, down

# Map kt-kernel internal alias → output GGUF tensor name suffix
_W2GGUF_SUFFIX = {
    "w1": "ffn_gate_exps",
    "w3": "ffn_up_exps",
    "w2": "ffn_down_exps",
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class TensorRecord:
    """One (layer, expert, alias) record in the output manifest.json."""
    layer: int
    expert: int
    tensor: str  # one of: w1, w3, w2
    gguf_name: str
    shape: tuple[int, int]  # (out, in)
    cosine_sim: float
    max_abs_diff: float
    bytes: int  # post-Q8_0 byte size for this expert's slice (one expert only)


# ---------------------------------------------------------------------------
# Safetensors index + multi-shard tensor accessor
# ---------------------------------------------------------------------------


class WeightSource:
    """Lazily opens safetensors shards on demand, one handle cached per file.

    Holds open file handles for the lifetime of the object. Cheap on memory
    because safetensors `safe_open` is mmap-based, not load-into-RAM.
    """

    def __init__(self, src_dir: Path) -> None:
        self.src_dir = src_dir
        index_path = src_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"safetensors index not found at {index_path}; "
                "is --src pointing at the model directory?"
            )
        with index_path.open() as f:
            index = json.load(f)
        # weight_map: tensor name → shard file name (relative).
        self._weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def _handle(self, name: str):
        fname = self._weight_map[name]
        h = self._handles.get(fname)
        if h is None:
            h = safe_open(str(self.src_dir / fname), framework="pt", device="cpu")
            self._handles[fname] = h
        return h

    def get(self, name: str) -> torch.Tensor:
        if name not in self._weight_map:
            raise KeyError(name)
        return self._handle(name).get_tensor(name)

    def list_expert_keys(self) -> list[str]:
        return [k for k in self._weight_map if ".ffn.experts." in k]

    def close(self) -> None:
        self._handles.clear()


# ---------------------------------------------------------------------------
# Per-expert quantization (the actual numerical work)
# ---------------------------------------------------------------------------


def quantize_one_expert_tensor(
    weight_int8: np.ndarray,
    scale_fp32: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Run W8A8 → fp32 dequant → Q8_0 quantize and compute roundtrip stats.

    Args:
        weight_int8: shape (out, in), dtype int8.
        scale_fp32:  shape (out, 1), dtype fp32.

    Returns:
        q8_0_bytes:  raw Q8_0 bytes, shape (out, in/32*34) uint8 (the trailing dim grows
                     from `in` fp32 elements to `in/32 * 34` bytes after quantization).
        cosine_sim:  cosine similarity between W8A8 dequant fp32 and Q8_0 dequant fp32.
        max_abs_diff: max |diff| between the two fp32 reconstructions.
    """
    in_features = weight_int8.shape[1]
    if in_features % 32 != 0:
        raise ValueError(
            f"Q8_0 requires in-features divisible by 32 (block size); got {in_features}"
        )

    # 1. W8A8 dequant to fp32.
    fp32 = dequant_w8a8_numpy(weight_int8, scale_fp32)

    # 2. Q8_0 quantize (block-wise along the last dim, 32 elements/block).
    #    Output shape: same leading dims, last dim becomes in/32 * 34 (uint8).
    q8 = Q8_0.quantize(fp32)

    # 3. Roundtrip dequant for verification.
    fp32_rt = Q8_0.dequantize(q8)

    # 4. Stats.
    cos = _cosine_sim(fp32, fp32_rt)
    max_diff = float(np.abs(fp32 - fp32_rt).max())

    return q8, cos, max_diff


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == 0.0 and nb == 0.0 else 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Per-expert worker (per-process; data flows out via memmap, not IPC)
# ---------------------------------------------------------------------------
#
# Design rationale
# ----------------
# Going from layer-level work units down to expert-level lets us scale to
# ``--workers > num_layers`` (e.g. 46 workers on remote, near the shard count)
# and gives finer-grain retry/progress. The catch is that returning every
# expert's 25 MB of Q8_0 bytes through ProcessPoolExecutor's pickle pipeline
# would cost ~275 GB of IPC for the full model — slower than the quantize work
# itself.
#
# Instead, we pre-allocate per-(layer, alias) memmap files in ``--tmp-dir``
# (one file = stacked shape ``(n_experts, out_features, in_features/32*34)``
# uint8). Workers seek into their assigned ``slot_idx`` and write Q8_0 bytes
# directly into the kernel page cache. The main process opens the same files
# read-only (after every expert in that layer is done), streams them through
# ``GGUFWriter``, and immediately deletes the tmp files to keep disk peak at
# one layer's worth (~6.5 GB) above the produced .gguf output.
#
# Workers cache one ``WeightSource`` and one ``np.memmap`` per (layer, alias)
# in module-level globals so the second through Nth tasks on the same worker
# pay zero open-file/parse cost.

_worker_source: Optional["WeightSource"] = None
_worker_memmaps: dict = {}  # (tmp_dir, layer, alias) -> np.memmap


def _get_worker_source(src_dir: str) -> "WeightSource":
    global _worker_source
    if _worker_source is None or str(_worker_source.src_dir) != str(src_dir):
        _worker_source = WeightSource(Path(src_dir))
    return _worker_source


def _get_worker_memmap(tmp_dir: str, layer: int, alias: str, shape: tuple[int, int, int]) -> np.memmap:
    key = (tmp_dir, layer, alias)
    mm = _worker_memmaps.get(key)
    if mm is None:
        path = Path(tmp_dir) / _memmap_filename(layer, alias)
        mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)
        _worker_memmaps[key] = mm
    return mm


def _format_layer_filename(layer: int) -> str:
    return f"dsv4_flash_q8_0-layer_{layer:03d}.gguf"


def _memmap_filename(layer: int, alias: str) -> str:
    return f"layer_{layer:03d}_{alias}.bin"


def _gguf_tensor_name(layer: int, alias: str) -> str:
    return f"blk.{layer}.{_W2GGUF_SUFFIX[alias]}.weight"


@dataclass
class ExpertResult:
    """What a worker returns to the main process after one (layer, expert) call.

    Stays small — only stats and small bookkeeping — so IPC is cheap.
    """
    layer: int
    expert: int
    slot_idx: int
    elapsed_sec: float
    # per-alias (cosine, max_abs_diff, bytes)
    per_alias: dict[str, tuple[float, float, int]]


def convert_one_expert(
    src_dir: str,
    tmp_dir: Optional[str],
    layer: int,
    expert: int,
    slot_idx: int,
    n_experts: int,
    moe_inter: int,
    hidden: int,
) -> ExpertResult:
    """Worker entry point: quantize one expert's three tensors.

    Args:
        src_dir:    DSv4-Flash safetensors directory.
        tmp_dir:    Where the per-(layer, alias) memmap files live. ``None``
                    means dry-run: compute Q8_0 and stats but skip the disk
                    write. The shape of the buffer is still computed so the
                    cosine/diff numbers match a real run.
        layer:      Decoder layer index (already filtered to MoE-only).
        expert:     Routed expert index in this layer.
        slot_idx:   Zero-based position of this expert within
                    ``expert_ids`` — i.e. its slot inside the per-layer stack.
        n_experts:  Total experts being written for this layer (the memmap's
                    first-dim size).
        moe_inter:  ``config.moe_intermediate_size`` (2048 for V4-Flash).
        hidden:     ``config.hidden_size`` (4096 for V4-Flash).

    Returns:
        ExpertResult with timing + per-alias (cosine, max_abs_diff, bytes).
    """
    t0 = time.time()
    source = _get_worker_source(src_dir)

    shape_per_alias = {
        "w1": (moe_inter, hidden),
        "w3": (moe_inter, hidden),
        "w2": (hidden, moe_inter),
    }

    per_alias: dict[str, tuple[float, float, int]] = {}

    for alias in EXPERT_TENSOR_NAMES:
        out, in_f = shape_per_alias[alias]
        byte_in = (in_f // 32) * 34

        w_key = f"layers.{layer}.ffn.experts.{expert}.{alias}.weight"
        s_key = f"layers.{layer}.ffn.experts.{expert}.{alias}.weight_scale"
        if not source.has(w_key) or not source.has(s_key):
            raise KeyError(
                f"missing weight or scale for layer={layer} expert={expert} alias={alias}; "
                f"keys: {w_key!r} / {s_key!r}"
            )
        weight = source.get(w_key).numpy()
        scale = source.get(s_key).numpy()
        if weight.shape != (out, in_f):
            raise ValueError(
                f"layer={layer} expert={expert} alias={alias}: "
                f"weight shape {weight.shape} != expected {(out, in_f)}"
            )

        q8, cos, max_diff = quantize_one_expert_tensor(weight, scale)

        if tmp_dir is not None:
            mm = _get_worker_memmap(tmp_dir, layer, alias, (n_experts, out, byte_in))
            mm[slot_idx] = q8
        # else: dry-run, q8 is discarded by GC

        per_alias[alias] = (cos, max_diff, int(q8.nbytes))

    return ExpertResult(
        layer=layer,
        expert=expert,
        slot_idx=slot_idx,
        elapsed_sec=time.time() - t0,
        per_alias=per_alias,
    )


# ---------------------------------------------------------------------------
# Main-process orchestration: memmap pre-alloc + .gguf assembly
# ---------------------------------------------------------------------------


def _preallocate_layer_memmaps(
    tmp_dir: Path,
    layer: int,
    n_experts: int,
    moe_inter: int,
    hidden: int,
) -> dict[str, int]:
    """Pre-allocate the 3 memmap files for one layer using ftruncate.

    ftruncate creates a sparse file: actual disk usage grows as workers write
    into it, so the up-front cost is O(syscalls), not O(bytes).

    Returns:
        {alias: total_bytes} for logging/reporting.
    """
    shape_per_alias = {
        "w1": (moe_inter, hidden),
        "w3": (moe_inter, hidden),
        "w2": (hidden, moe_inter),
    }
    sizes: dict[str, int] = {}
    for alias in EXPERT_TENSOR_NAMES:
        out, in_f = shape_per_alias[alias]
        byte_in = (in_f // 32) * 34
        total = n_experts * out * byte_in
        path = tmp_dir / _memmap_filename(layer, alias)
        if path.is_file() and path.stat().st_size == total:
            sizes[alias] = total
            continue
        # ftruncate to the right size. Open in 'wb' would zero-truncate;
        # we want O_CREAT but preserve existing data only if size matches.
        with open(path, "wb") as fp:
            fp.truncate(total)
        sizes[alias] = total
    return sizes


def _assemble_and_write_layer_gguf(
    layer: int,
    tmp_dir: Path,
    dst_dir: Path,
    expert_ids: list[int],
    moe_inter: int,
    hidden: int,
) -> Path:
    """Open the three layer memmaps, write one .gguf, then delete the memmaps.

    Uses ``np.memmap`` (not a full RAM read) so the writer streams bytes from
    the page cache straight into the .gguf — no 6.5 GB allocation here.
    """
    shape_per_alias = {
        "w1": (moe_inter, hidden),
        "w3": (moe_inter, hidden),
        "w2": (hidden, moe_inter),
    }
    out_path = dst_dir / _format_layer_filename(layer)
    writer = GGUFWriter(str(out_path), arch="deepseek2")
    writer.add_uint32("dsv4_flash.converter.layer", layer)
    writer.add_uint32("dsv4_flash.converter.num_experts", len(expert_ids))
    writer.add_uint32("dsv4_flash.converter.moe_intermediate_size", moe_inter)
    writer.add_uint32("dsv4_flash.converter.hidden_size", hidden)

    mms: list[np.memmap] = []
    for alias in EXPERT_TENSOR_NAMES:
        out, in_f = shape_per_alias[alias]
        byte_in = (in_f // 32) * 34
        path = tmp_dir / _memmap_filename(layer, alias)
        mm = np.memmap(path, dtype=np.uint8, mode="r",
                       shape=(len(expert_ids), out, byte_in))
        mms.append(mm)
        writer.add_tensor(
            name=_gguf_tensor_name(layer, alias),
            tensor=mm,
            raw_dtype=GGMLQuantizationType.Q8_0,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Release the writer-side mmaps before unlinking — on Linux unlink works
    # while files are still open, but releasing first plays nicer with NFS.
    for mm in mms:
        del mm
    for alias in EXPERT_TENSOR_NAMES:
        (tmp_dir / _memmap_filename(layer, alias)).unlink(missing_ok=True)

    return out_path


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _parse_range(spec: str, lo: int, hi: int) -> list[int]:
    """Parse a range spec like '3-42' or '0,2,4-6' into a sorted unique int list.

    `lo`, `hi` are inclusive bounds used for the 'all' case and for clamping.
    """
    if spec == "all":
        return list(range(lo, hi + 1))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _load_config(src: Path) -> dict:
    cfg = src / "config.json"
    if cfg.is_file():
        with cfg.open() as f:
            return json.load(f)
    return {}


def _resolve_moe_layers(args: argparse.Namespace, cfg: dict) -> list[int]:
    """Resolve which decoder layers contain a routed-expert MoE block."""
    n_layers = cfg.get("num_hidden_layers", DEFAULT_NUM_LAYERS)
    first_dense = cfg.get("first_k_dense_replace", 0)
    all_moe = list(range(first_dense, n_layers))
    requested = _parse_range(args.layers, 0, n_layers - 1)
    out = [L for L in requested if L in all_moe]
    skipped = [L for L in requested if L not in all_moe]
    if skipped:
        logger.info(
            "skipping non-MoE / out-of-range layers: %s (first_k_dense_replace=%d, num_hidden_layers=%d)",
            skipped, first_dense, n_layers,
        )
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert DSv4-Flash W8A8 routed experts to GGUF Q8_0.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True,
                   help="Path to DSv4-Flash safetensors model directory (contains config.json + index).")
    p.add_argument("--dst", required=True,
                   help="Output directory for .gguf files + manifest.json.")
    p.add_argument("--layers", default="all",
                   help="Layer selector: 'all', '3-42', '0,3,5-7', etc. "
                        "Dense layers (first_k_dense_replace) are silently skipped.")
    p.add_argument("--experts", default="all",
                   help="Expert selector inside each layer, same syntax as --layers.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes. Each work unit is "
                        "one (layer, expert), so workers > num_layers is useful "
                        "(set this near the number of safetensors shards, ~46).")
    p.add_argument("--tmp-dir", default=None,
                   help="Where to put intermediate per-(layer, alias) memmap files. "
                        "Default: <dst>/_tmp_q8_chunks. Each layer needs ~6.5 GB "
                        "of tmp space, freed as soon as that layer's .gguf is written.")
    p.add_argument("--dry-run", action="store_true",
                   help="Do everything except writing .gguf files. Still computes cosine.")
    p.add_argument("--verify-cosine", type=float, default=0.9995,
                   help="Per-tensor cosine-sim threshold. Below = WARN + counted; "
                        "if >10 tensors fall below, the converter exits non-zero.")
    p.add_argument("--log-every", type=int, default=64,
                   help="Log progress every N completed expert work units.")
    p.add_argument("--num-experts-total", type=int, default=DEFAULT_NUM_EXPERTS,
                   help="Total expert count from config.n_routed_experts.")
    p.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE,
                   help="config.hidden_size.")
    p.add_argument("--moe-intermediate-size", type=int, default=DEFAULT_MOE_INTER,
                   help="config.moe_intermediate_size.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    src = Path(args.src)
    if not src.is_dir():
        logger.error("--src %s is not a directory", src)
        return 2
    dst = Path(args.dst)
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(src)
    # Allow config.json to override defaults non-destructively.
    if cfg:
        args.num_experts_total = cfg.get("n_routed_experts", args.num_experts_total)
        args.hidden_size = cfg.get("hidden_size", args.hidden_size)
        args.moe_intermediate_size = cfg.get("moe_intermediate_size", args.moe_intermediate_size)

    moe_layers = _resolve_moe_layers(args, cfg)
    expert_ids = _parse_range(args.experts, 0, args.num_experts_total - 1)
    expert_ids = [e for e in expert_ids if 0 <= e < args.num_experts_total]

    if not moe_layers:
        logger.error("no MoE layers selected after filtering; nothing to do")
        return 3
    if not expert_ids:
        logger.error("no experts selected after filtering; nothing to do")
        return 3

    n_work_units = len(moe_layers) * len(expert_ids)
    logger.info(
        "plan: %d MoE layers × %d experts = %d work units (3 tensors each), "
        "workers=%d, dry_run=%s, verify_cosine=%.4f",
        len(moe_layers), len(expert_ids), n_work_units,
        args.workers, args.dry_run, args.verify_cosine,
    )

    # --- tmp dir + memmap pre-allocation -----------------------------------
    tmp_dir: Optional[Path] = None
    if not args.dry_run:
        tmp_dir = Path(args.tmp_dir) if args.tmp_dir else (dst / "_tmp_q8_chunks")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("tmp memmaps will live under: %s", tmp_dir)
        for L in moe_layers:
            sizes = _preallocate_layer_memmaps(
                tmp_dir, L, len(expert_ids),
                args.moe_intermediate_size, args.hidden_size,
            )
            logger.debug(
                "preallocated layer %d memmaps (sparse): %s",
                L, {a: f"{n/1024/1024:.1f} MB" for a, n in sizes.items()},
            )

    # --- per-layer accounting (assembled as expert work units complete) ----
    # records[L] holds the TensorRecord list as experts come in; len == 3 *
    # n_experts means the layer is fully quantized and ready to .gguf-write.
    records: dict[int, list[TensorRecord]] = {L: [] for L in moe_layers}
    elapsed_by_layer: dict[int, float] = collections.defaultdict(float)
    completion: dict[int, int] = {L: 0 for L in moe_layers}
    layer_gguf_paths: dict[int, str] = {}
    layer_below_threshold: dict[int, int] = {L: 0 for L in moe_layers}
    layer_write_elapsed: dict[int, float] = {}

    total_below = 0
    n_completed = 0
    t_global = time.time()

    def _ingest(result: ExpertResult) -> bool:
        """Apply one ExpertResult; return True if the global stop condition fired."""
        nonlocal total_below, n_completed
        L, E = result.layer, result.expert
        elapsed_by_layer[L] += result.elapsed_sec
        for alias, (cos, max_diff, nbytes) in result.per_alias.items():
            if cos < args.verify_cosine:
                layer_below_threshold[L] += 1
                total_below += 1
                logger.warning(
                    "low cosine: layer=%d expert=%d alias=%s cos=%.6f max_diff=%.3e",
                    L, E, alias, cos, max_diff,
                )
            out, in_f = (
                (args.moe_intermediate_size, args.hidden_size) if alias in ("w1", "w3")
                else (args.hidden_size, args.moe_intermediate_size)
            )
            records[L].append(TensorRecord(
                layer=L,
                expert=E,
                tensor=alias,
                gguf_name=_gguf_tensor_name(L, alias),
                shape=(out, in_f),
                cosine_sim=cos,
                max_abs_diff=max_diff,
                bytes=nbytes,
            ))
        completion[L] += 1
        n_completed += 1
        if n_completed % args.log_every == 0:
            logger.info(
                "progress: %d/%d expert units done (%.1f%%), elapsed %.1fs",
                n_completed, n_work_units, 100 * n_completed / n_work_units,
                time.time() - t_global,
            )

        # If this layer just got its last expert, assemble + write .gguf.
        if completion[L] == len(expert_ids) and not args.dry_run:
            t_write = time.time()
            gguf_path = _assemble_and_write_layer_gguf(
                L, tmp_dir, dst, expert_ids,
                args.moe_intermediate_size, args.hidden_size,
            )
            layer_gguf_paths[L] = str(gguf_path)
            layer_write_elapsed[L] = time.time() - t_write
            logger.info(
                "layer %d done: 3 tensors × %d experts in %.1fs quantize + %.1fs gguf-write, "
                "%d tensors below cosine threshold, -> %s",
                L, len(expert_ids), elapsed_by_layer[L], layer_write_elapsed[L],
                layer_below_threshold[L], gguf_path,
            )
        elif completion[L] == len(expert_ids):
            # dry-run path: log layer completion without writing .gguf
            layer_gguf_paths[L] = ""
            layer_write_elapsed[L] = 0.0
            logger.info(
                "layer %d done [dry-run]: 3 tensors × %d experts in %.1fs, "
                "%d tensors below cosine threshold",
                L, len(expert_ids), elapsed_by_layer[L], layer_below_threshold[L],
            )

        return total_below > 10

    # --- dispatch -----------------------------------------------------------
    if args.workers <= 1:
        # Serial path — keep available for debugging / tiny test runs.
        for L in moe_layers:
            for slot_idx, E in enumerate(expert_ids):
                res = convert_one_expert(
                    src_dir=args.src,
                    tmp_dir=str(tmp_dir) if tmp_dir else None,
                    layer=L, expert=E, slot_idx=slot_idx,
                    n_experts=len(expert_ids),
                    moe_inter=args.moe_intermediate_size,
                    hidden=args.hidden_size,
                )
                if _ingest(res):
                    logger.error("too many tensors below verify_cosine; aborting")
                    break
            if total_below > 10:
                break
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {}
            for L in moe_layers:
                for slot_idx, E in enumerate(expert_ids):
                    fut = exe.submit(
                        convert_one_expert,
                        args.src,
                        str(tmp_dir) if tmp_dir else None,
                        L, E, slot_idx, len(expert_ids),
                        args.moe_intermediate_size, args.hidden_size,
                    )
                    futures[fut] = (L, E)
            stop_now = False
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception:
                    logger.exception("worker raised; cancelling remaining work")
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    raise
                if _ingest(res):
                    logger.error("too many tensors below verify_cosine; cancelling remaining work")
                    stop_now = True
                    break
            if stop_now:
                for f in futures:
                    if not f.done():
                        f.cancel()

    total_elapsed = time.time() - t_global

    # --- cleanup tmp dir (only meaningful if all layers were assembled) ----
    if tmp_dir is not None and tmp_dir.is_dir():
        try:
            tmp_dir.rmdir()  # only succeeds if empty
        except OSError:
            logger.warning(
                "tmp dir not empty at %s (some layers may not have been written); leaving in place",
                tmp_dir,
            )

    # --- manifest -----------------------------------------------------------
    if not args.dry_run:
        manifest_path = dst / "manifest.json"
        manifest = {
            "src": str(src),
            "dst": str(dst),
            "num_hidden_layers": cfg.get("num_hidden_layers", DEFAULT_NUM_LAYERS),
            "first_k_dense_replace": cfg.get("first_k_dense_replace", 0),
            "n_routed_experts": args.num_experts_total,
            "hidden_size": args.hidden_size,
            "moe_intermediate_size": args.moe_intermediate_size,
            "verify_cosine": args.verify_cosine,
            "workers": args.workers,
            "total_elapsed_sec": total_elapsed,
            "total_tensors_below_threshold": total_below,
            "layers": [
                {
                    "layer": L,
                    "gguf_path": layer_gguf_paths.get(L, ""),
                    "quantize_elapsed_sec": elapsed_by_layer.get(L, 0.0),
                    "gguf_write_elapsed_sec": layer_write_elapsed.get(L, 0.0),
                    "n_below_threshold": layer_below_threshold.get(L, 0),
                    "tensors": [asdict(t) for t in sorted(
                        records[L], key=lambda r: (r.expert, r.tensor),
                    )],
                }
                for L in moe_layers if completion[L] == len(expert_ids)
            ],
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("wrote manifest: %s", manifest_path)

    fully_done = sum(1 for L in moe_layers if completion[L] == len(expert_ids))
    logger.info(
        "done: %d/%d layers fully converted in %.1fs, %d tensors below cosine threshold",
        fully_done, len(moe_layers), total_elapsed, total_below,
    )
    return 0 if total_below <= 10 and fully_done == len(moe_layers) else 4


if __name__ == "__main__":
    sys.exit(main())
