#!/usr/bin/env python3
"""P3 数值对账（MXFP4 版）：KTMoEWrapper(LLAMAFILE, MXFP4 GGUF) vs torch 参考。

与 ``p27_cpu_moe_reference_check.py`` 的关键区别：参考权重直接 dequant **同一份原生
MXFP4 权重**（不是 W8A8）。这样 cand 与 ref 用的是完全相同的母权重，唯一的数值损失源
是 kernel 内部把激活量化到 Q8_0（ggml_vec_dot_mxfp4_q8_0 的 vec_dot_type=Q8_0）。
因此阈值收紧到 cosine >= 0.999。

dequant 语义见 ``verify_mxfp4_layer.dequant_native``：value = FP4_TABLE[nibble] *
2^(e-127)，byte i -> Kpos 2i(lo),2i+1(hi)（原生 consecutive 排布，与转换器一致）。

用法::
  ${PYTHON_BIN} tools/p27_cpu_moe_reference_check_mxfp4.py \
    --model-dir /workspace/models/DeepSeekV4/DeepSeek-V4-Flash \
    --gguf /workspace/models/cache/dsv4_layer16_mxfp4.gguf \
    --layer-idx 16 --batch 4 --seed 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Offline single-layer check: the NPU stream-callback submit path (bypass=False)
# does NOT deliver CPU MoE output back to the returned tensor in this isolated
# context (verified: Q8_0 baseline returns all-zero too). Force the synchronous
# submit/sync path so the output_cpu->output_gpu copy actually lands. Production
# serving uses the real graph stream path and is unaffected.
os.environ.setdefault("KT_FORCE_SYNC_SUBMIT", "1")

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# 复用已有 helper（pin_memory patch、reference forward、cosine、device/stream）
from p27_cpu_moe_reference_check import (  # noqa: E402
    _ensure_pin_memory_or_patch,
    _resolve_device_and_stream,
    cosine_sim,
    reference_moe_forward,
)
from convert_mxfp4_layer_to_gguf import (  # noqa: E402
    _load_weight_map,
    _detect_experts_prefix,
    _open_shard,
    _as_u8,
)
from verify_mxfp4_layer import dequant_native  # noqa: E402

_ensure_pin_memory_or_patch()


def load_routed_experts_fp32(model_dir: Path, layer_idx: int, expert_ids):
    """Native MXFP4 -> fp32 dicts {eid: tensor} for only the routed experts (fast)."""
    weight_map = _load_weight_map(model_dir)
    prefix = _detect_experts_prefix(weight_map, layer_idx)
    cache: dict = {}
    w1d, w3d, w2d = {}, {}, {}
    for e in sorted(set(int(x) for x in expert_ids)):
        for proj, dst in (("w1", w1d), ("w3", w3d), ("w2", w2d)):
            wk = f"{prefix}.{e}.{proj}.weight"
            sk = f"{prefix}.{e}.{proj}.scale"
            h = _open_shard(model_dir, weight_map, cache, wk)
            dst[e] = torch.from_numpy(dequant_native(_as_u8(h.get_tensor(wk)), _as_u8(h.get_tensor(sk))))
    return w1d, w3d, w2d


def reference_moe_forward_dict(hidden, topk_ids, topk_weights, w1d, w3d, w2d):
    """Pure-PyTorch MoE forward using per-expert dicts (only routed experts present)."""
    import torch.nn.functional as F
    B, H = hidden.shape
    K = topk_ids.shape[1]
    h = hidden.float()
    out = torch.zeros(B, H, dtype=torch.float32)
    for i in range(B):
        for k in range(K):
            e = int(topk_ids[i, k].item())
            wf = float(topk_weights[i, k].item())
            gate = h[i] @ w1d[e].t()
            up = h[i] @ w3d[e].t()
            out[i].add_(wf * ((F.silu(gate) * up) @ w2d[e].t()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-experts-per-tok", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cpuinfer-threads", type=int, default=24)
    ap.add_argument("--threadpool-count", type=int, default=8)
    ap.add_argument("--chunked-prefill-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="npu", choices=("npu", "cuda", "cpu"))
    ap.add_argument("--npu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cos-min", type=float, default=0.999)
    ap.add_argument("--rel-tol", type=float, default=0.03)
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    gguf_file = args.gguf.expanduser().resolve()
    if not gguf_file.is_file():
        print(f"ERROR: GGUF 不存在: {gguf_file}", file=sys.stderr)
        return 2

    device, stream_handle = _resolve_device_and_stream(args.device, args.npu_id)
    print(f"[env] device={device} stream_handle={stream_handle}")
    if stream_handle == 0 and device.type != "cpu":
        print("ERROR: stream_handle=0 非 cpu device → task 不触发", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    hidden_cpu = torch.randn(args.batch, args.hidden_size, dtype=torch.bfloat16)
    topk_ids_cpu = torch.randint(0, args.num_experts, (args.batch, args.num_experts_per_tok), dtype=torch.long)
    topk_weights_cpu = torch.softmax(torch.randn(args.batch, args.num_experts_per_tok, dtype=torch.float32), dim=-1)
    routed = sorted(set(int(x) for x in topk_ids_cpu.flatten().tolist()))
    print(f"[run] hidden={tuple(hidden_cpu.shape)} routing[0]={topk_ids_cpu[0].tolist()} routed_experts={len(routed)}")

    print(f"[ref] dequant native MXFP4 layer {args.layer_idx} (routed {len(routed)} experts)...")
    w1d, w3d, w2d = load_routed_experts_fp32(model_dir, args.layer_idx, routed)

    try:
        from kt_kernel import KTMoEWrapper
    except ImportError as e:
        print(f"ERROR: import kt_kernel 失败: {e}", file=sys.stderr)
        return 2

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    wrapper = KTMoEWrapper(
        layer_idx=args.layer_idx,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=str(gguf_file),
        chunked_prefill_size=args.chunked_prefill_size,
        method="LLAMAFILE",
        numa_nodes=None,
    )
    wrapper.load_weights()

    print("[ref]  forward (pure pytorch fp32)...")
    ref_out = reference_moe_forward_dict(hidden_cpu, topk_ids_cpu, topk_weights_cpu, w1d, w3d, w2d)

    print(f"[cand] forward KTMoEWrapper(MXFP4) on {device} stream={stream_handle}...")
    cand_out = wrapper.forward(hidden_cpu.to(device), topk_ids_cpu.to(device),
                               topk_weights_cpu.to(device), stream_handle)
    if isinstance(cand_out, (list, tuple)):
        cand_out = cand_out[0]
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)

    a = cand_out.detach().cpu().float()
    b = ref_out.float()
    if a.shape != b.shape:
        print(f"FAIL: shape cand={tuple(a.shape)} ref={tuple(b.shape)}", file=sys.stderr)
        return 1

    cand_finite = bool(torch.isfinite(a).all().item())
    cos = float(cosine_sim(a, b).item())
    max_abs = float((a - b).abs().max().item())
    max_rel = max_abs / (float(b.abs().max().item()) + 1e-12)

    print("\n=== 数值对账 (MXFP4) ===")
    print(f"  cosine_sim   = {cos:.6f}     (期望 >= {args.cos_min:.4f})")
    print(f"  max_abs_err  = {max_abs:.4e}")
    print(f"  max_rel_err  = {max_rel:.4%}   (期望 <= {args.rel_tol:.2%})")
    print(f"  ref  L2/max  = {b.norm():.4e} / {b.abs().max():.4e}")
    print(f"  cand L2/max  = {a.norm():.4e} / {a.abs().max():.4e}")
    print(f"  finite(cand) = {cand_finite}")
    print(f"  ref [0,:6]   = {[round(float(x),5) for x in b[0,:6].tolist()]}")
    print(f"  cand[0,:6]   = {[round(float(x),5) for x in a[0,:6].tolist()]}")
    print(f"  token0 cand nonzero={(a[0].abs()>1e-7).sum().item()}/{a[0].numel()}\n")

    if (not cand_finite) or math.isnan(cos) or cos < args.cos_min or max_rel > args.rel_tol:
        print("RESULT: FAIL — MXFP4 CPU MoE 与参考不一致（kernel/转换器/nibble 序 之一有误）。", file=sys.stderr)
        return 1
    print("RESULT: PASS — MXFP4 kernel 数值正确（唯一损失源为激活 Q8 量化）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
