#!/usr/bin/env python3
"""
Convert one DeepSeek-V4-Flash (W8A8) MoE layer from HuggingFace safetensors to a
minimal GGUF containing only the three stacked expert tensors expected by
kt-kernel LlamafileMoEWrapper:

  blk.{L}.ffn_gate_exps.weight   (Q8_0)
  blk.{L}.ffn_up_exps.weight    (Q8_0)
  blk.{L}.ffn_down_exps.weight   (Q8_0)

Layout matches llama.cpp / GGUFLoader conventions:
  - File tensor dims follow ggml order (reversed relative to the PyTorch view in GGUFLoader).
  - Gate/up: ggml shape (n_embd, n_ff, n_expert) = (4096, 2048, E)
  - Down:    ggml shape (n_ff, n_embd, n_expert) = (2048, 4096, E)

W8A8 dequant: weight_fp32 = int8_weight.to(float32) * weight_scale
(per output channel; scale shape (Nout, 1) broadcasts).

Memory: stacks experts in batches of --expert-batch (default 32) in FP32 before
Q8_0 quantization to cap peak RAM (~1 GiB per gate/up batch on default sizes).

单层转换见同目录 `batch_convert_w8a8_layers_mp.py`（层间多进程批量）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GGUF_PY = _REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if _GGUF_PY.is_dir():
    sys.path.insert(0, str(_GGUF_PY))
else:
    raise RuntimeError(f"Expected gguf-py at {_GGUF_PY}")

import gguf  # noqa: E402
from safetensors import safe_open  # noqa: E402


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")
    data = json.loads(index_path.read_text())
    return data["weight_map"]


def _detect_experts_uri(weight_map: dict[str, str], layer_idx: int) -> tuple[str, tuple[str, str, str]]:
    """
    Returns (experts_prefix, (gate_w, up_w, down_w)) where full keys are
    f"{experts_prefix}.{e}.{proj}.weight`.
    """
    layer_prefixes = (
        f"model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    )
    sample = None
    for layer_prefix in layer_prefixes:
        for k in weight_map:
            if not k.startswith(layer_prefix):
                continue
            # Routed MoE only: require ".experts.<id>." (exclude shared_experts.*)
            if ".experts." not in k or ".shared_experts" in k:
                continue
            if k.endswith(".weight") and ".experts.0." in k:
                sample = k
                break
        if sample is not None:
            break
    if sample is None:
        raise ValueError(
            "No per-expert .weight keys found for layer "
            f"{layer_idx} (tried prefixes {layer_prefixes!r})"
        )

    before, _ = sample.split(".experts.0.", 1)
    experts_prefix = before + ".experts"

    if ".w1.weight" in sample:
        return experts_prefix, ("w1", "w3", "w2")
    if ".gate_proj.weight" in sample:
        return experts_prefix, ("gate_proj", "up_proj", "down_proj")
    raise ValueError(f"Cannot infer projection names from sample key: {sample!r}")


def _dequant_int8(w_int8: torch.Tensor, scale_fp32: torch.Tensor) -> torch.Tensor:
    s = scale_fp32
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    return w_int8.to(torch.float32) * s.to(torch.float32)


def _open_shard(model_dir: Path, weight_map: dict[str, str], cache: dict[str, object], key: str):
    if key not in weight_map:
        raise KeyError(key)
    shard = weight_map[key]
    if shard not in cache:
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        cache[shard] = safe_open(path, framework="pt")
    return cache[shard]


def _fp32_to_bf16_ndarray(arr_fp32: np.ndarray) -> np.ndarray:
    """np.float32 ndarray -> bf16 ndarray (uint16, same shape).

    Bit pattern matches ggml ggml_compute_fp32_to_bf16 (round-to-nearest-even).
    Caller hands the uint16 ndarray to gguf.GGUFWriter.add_tensor(..., raw_dtype=BF16);
    writer uses tensor.shape as logical shape and 2 bytes/elem as type size.
    """
    if arr_fp32.dtype != np.float32:
        raise TypeError(f"expected np.float32, got {arr_fp32.dtype}")
    if not arr_fp32.flags["C_CONTIGUOUS"]:
        arr_fp32 = np.ascontiguousarray(arr_fp32)
    u32 = arr_fp32.view(np.uint32)
    bias = (u32 >> 16) & 1
    rounded = (u32 + 0x7FFF + bias).astype(np.uint32)
    bf16 = (rounded >> 16).astype(np.uint16)
    return bf16


def _build_moe_tensor(
    model_dir: Path,
    weight_map: dict[str, str],
    experts_prefix: str,
    proj_name: str,
    num_experts: int,
    expert_batch: int,
    layout: str,
    quant: str,
) -> tuple[np.ndarray, "gguf.GGMLQuantizationType"]:
    """
    layout:
      'gate_up' — safetensors (n_ff, n_embd) per expert -> numpy (E, n_ff, n_embd)
                  i.e. ggml ne = (n_embd, n_ff, E) with hidden innermost
      'down'    — safetensors (n_embd, n_ff) per expert -> numpy (E, n_embd, n_ff)
                  i.e. ggml ne = (n_ff, n_embd, E) with intermediate innermost
    quant: 'q8_0' | 'bf16'

    KT C++ ``LLAMA_MOE_TP::forward_one`` uses pointer arithmetic that assumes:
      - gate/up per-expert layout (intermediate, hidden) row-major, hidden inner
      - down  per-expert layout (hidden,  intermediate) row-major, intermediate inner

    Q8_0 quantizes along the **last numpy dim**, so the last dim must be the
    one we want as the GEMM K (the inner dim of each row in KT's view):
      - gate/up: last numpy dim = hidden (4096, %32==0 ✓)
      - down:    last numpy dim = intermediate (2048, %32==0 ✓)

    A previous version of this script applied ``permute(2,1,0)`` which placed E
    as the innermost dim. That broke KT's pointer math (each expert's bytes
    were no longer contiguous) and caused: Q8_0 -> NaN, BF16 -> finite but
    cosine ~ 0 vs reference. See doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md.
    """
    cache: dict[str, object] = {}
    chunks: list[np.ndarray] = []

    for batch_lo in range(0, num_experts, expert_batch):
        batch_hi = min(batch_lo + expert_batch, num_experts)
        tensors: list[torch.Tensor] = []
        for e in range(batch_lo, batch_hi):
            wk = f"{experts_prefix}.{e}.{proj_name}.weight"
            sk = f"{experts_prefix}.{e}.{proj_name}.weight_scale"
            h = _open_shard(model_dir, weight_map, cache, wk)
            w = _dequant_int8(h.get_tensor(wk), h.get_tensor(sk))
            tensors.append(w)

        stacked = torch.stack(tensors, dim=0).contiguous()
        del tensors

        if layout == "gate_up":
            # safetensors per-expert (intermediate, hidden) is exactly KT's
            # expected per-expert layout. Stacked (E, intermediate, hidden) is
            # C-contiguous with hidden inner — what KT and Q8_0 both want.
            x = stacked
        elif layout == "down":
            # safetensors per-expert (hidden, intermediate). Stacked
            # (E, hidden, intermediate) is C-contiguous with intermediate inner
            # — exactly KT's down-proj layout and Q8_0's row direction.
            x = stacked
        else:
            raise ValueError(layout)

        arr = x.numpy().astype(np.float32)
        del stacked, x

        if quant == "q8_0":
            if not gguf.can_quantize_to_q8_0(arr):
                raise ValueError(f"Q8_0 requires last dim % 32 == 0, got shape {arr.shape}")
            chunks.append(gguf.quantize_q8_0(arr))
        elif quant == "bf16":
            # Same shape as fp32, dtype uint16 (raw bf16 bit pattern); concat along E axis below.
            chunks.append(_fp32_to_bf16_ndarray(arr))
        else:
            raise ValueError(quant)

    # Experts are batched along axis 0 (outer, slowest dim) now that
    # `_build_moe_tensor` keeps numpy layout as (E, ...). Concat along axis 0.
    out = np.concatenate(chunks, axis=0)
    if quant == "q8_0":
        return out, gguf.GGMLQuantizationType.Q8_0
    return out, gguf.GGMLQuantizationType.BF16


# Backward compat — keep the old name so other batch tools still import.
def _build_q80_moe_tensor(*args, **kwargs):
    arr, _ = _build_moe_tensor(*args, quant="q8_0", **kwargs)
    return arr


def convert_layer(
    model_dir: Path,
    layer_idx: int,
    output_path: Path,
    num_experts: int,
    expert_batch: int,
    hidden_size: int,
    moe_intermediate_size: int,
    quant: str = "q8_0",
) -> None:
    if quant not in ("q8_0", "bf16"):
        raise ValueError(f"--quant must be one of q8_0|bf16, got {quant!r}")

    weight_map = _load_weight_map(model_dir)
    experts_prefix, (gate_n, up_n, down_n) = _detect_experts_uri(weight_map, layer_idx)

    if quant == "q8_0" and num_experts % 32 != 0:
        raise ValueError(f"num_experts ({num_experts}) must be a multiple of 32 for Q8_0 blocks along expert axis")

    # Smoke keys
    probe = f"{experts_prefix}.0.{gate_n}.weight"
    if probe not in weight_map:
        raise KeyError(f"Missing {probe!r}")

    print(f"[convert] layer={layer_idx} experts_prefix={experts_prefix!r} proj=({gate_n},{up_n},{down_n}) quant={quant}")

    gate_arr, ggml_type = _build_moe_tensor(
        model_dir, weight_map, experts_prefix, gate_n, num_experts, expert_batch, "gate_up", quant=quant
    )
    print(f"[convert] gate shape {gate_arr.shape} dtype={gate_arr.dtype} ggml_type={ggml_type}")

    up_arr, _ = _build_moe_tensor(
        model_dir, weight_map, experts_prefix, up_n, num_experts, expert_batch, "gate_up", quant=quant
    )
    print(f"[convert] up   shape {up_arr.shape} dtype={up_arr.dtype}")

    down_arr, _ = _build_moe_tensor(
        model_dir, weight_map, experts_prefix, down_n, num_experts, expert_batch, "down", quant=quant
    )
    print(f"[convert] down shape {down_arr.shape} dtype={down_arr.dtype}")

    arch = "deepseek2"
    writer = gguf.GGUFWriter(str(output_path), arch)
    writer.add_quantization_version(2)
    if quant == "q8_0":
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    else:
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_name(f"dsv4-w8a8-layer{layer_idx}-moe-{quant}")
    writer.add_uint32(gguf.Keys.LLM.EXPERT_COUNT.format(arch=arch), num_experts)
    writer.add_uint32(gguf.Keys.LLM.EXPERT_USED_COUNT.format(arch=arch), 6)
    writer.add_uint32(gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=arch), hidden_size)
    writer.add_uint32(gguf.Keys.LLM.EXPERT_FEED_FORWARD_LENGTH.format(arch=arch), moe_intermediate_size)

    base = f"blk.{layer_idx}"
    writer.add_tensor(f"{base}.ffn_gate_exps.weight", gate_arr, raw_dtype=ggml_type)
    writer.add_tensor(f"{base}.ffn_up_exps.weight", up_arr, raw_dtype=ggml_type)
    writer.add_tensor(f"{base}.ffn_down_exps.weight", down_arr, raw_dtype=ggml_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer.write_header_to_file(str(output_path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"[convert] wrote {output_path} ({output_path.stat().st_size / 1e9:.3f} GB)")


def _verify_with_reader(path: Path) -> None:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    names = [t.name for t in reader.tensors]
    print(f"[verify] tensors: {names}")
    for t in reader.tensors:
        print(f"  {t.name} shape={t.shape} type={t.tensor_type}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="HF model dir with safetensors + index.json")
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Output .gguf path")
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--expert-batch", type=int, default=32, help="Experts per FP32 batch before Q8_0")
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument(
        "--quant",
        type=str,
        default="q8_0",
        choices=["q8_0", "bf16"],
        help="Output element type: q8_0 (default, prod path) or bf16 (debug fallback for aarch64 Q8_0-NaN bug)",
    )
    ap.add_argument("--verify-reader", action="store_true", help="Print tensor table after write")
    args = ap.parse_args()

    model_dir = args.input.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"--input must be a directory: {model_dir}")

    convert_layer(
        model_dir=model_dir,
        layer_idx=args.layer_idx,
        output_path=args.output.expanduser().resolve(),
        num_experts=args.num_experts,
        expert_batch=args.expert_batch,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        quant=args.quant,
    )

    if args.verify_reader:
        _verify_with_reader(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
