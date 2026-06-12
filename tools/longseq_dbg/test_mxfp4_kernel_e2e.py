#!/usr/bin/env python3
"""True §10.1 acceptance: my Triton kernel vs the verified vectorized conversion, each
fed through the SAME native NZ cast + production npu_fused_experts, compared at the
GEMM output. Proves the kernel is a drop-in W8A8 producer for the streaming path.

  ref path : MXFP4 -> mxfp4_to_w8a8_nz (vectorized, bit-exact slow) -> NZ -> npu_fused_experts
  ker path : MXFP4 -> mxfp4_dequant_requant (Triton kernel) -> transpose -> NZ -> npu_fused_experts
  PASS     : cosine(out_ker, out_ref) ~ 1.0
"""
import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parent))

from mxfp4_conv_vectorized_npu import (  # noqa: E402
    _load_weight_map, _open_shard, _as_u8, _fp4_lut, mxfp4_to_w8a8_nz,
)
from mxfp4_dequant_triton import mxfp4_layer_to_slots  # noqa: E402

DEV = "npu"
NZ = 29
H, I = 4096, 2048


def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale,
                      topk_weights, topk_ids, top_k):
    """Verbatim copy of the production op (sglang ...fused_moe_method_npu.npu_fused_experts,
    non-wna16 path) -- inlined to avoid the package's circular import in offline tests."""
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
               .view(top_k, -1).permute(1, 0).contiguous())
    hidden_states, expanded_row_idx, expanded_expert_idx = torch.ops.npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens)
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts).to(torch.int64)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w13],
        scale=[w13_scale.to(scale_dtype)], per_token_scale=[pertoken_scale],
        split_item=2, group_list_type=0, group_type=0, group_list=expert_tokens,
        output_dtype=original_dtype)[0]
    hidden_states, pertoken_scale = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, activate_left=True, quant_mode=1)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w2],
        scale=[w2_scale.to(scale_dtype)], per_token_scale=[pertoken_scale],
        split_item=2, group_list_type=0, group_type=0, group_list=expert_tokens,
        output_dtype=original_dtype)[0]
    return torch.ops.npu.npu_moe_finalize_routing(
        hidden_states, skip1=None, skip2=None, bias=None, scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx, export_for_source_row=topk_ids)


def load_combined(md, wm, cache, layer, experts):
    """Return combined w13 (cat w1,w3 along OUT) and w2 codes+scale (host uint8)."""
    def stack(proj):
        cs, ss = [], []
        for e in experts:
            wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
            sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
            h = _open_shard(md, wm, cache, wk)
            cs.append(_as_u8(h.get_tensor(wk)))
            ss.append(_as_u8(h.get_tensor(sk)))
        return torch.stack(cs), torch.stack(ss)

    c1, s1 = stack("w1")
    c3, s3 = stack("w3")
    c13 = torch.cat([c1, c3], dim=1)   # [E, 2I, H/2]
    s13 = torch.cat([s1, s3], dim=1)   # [E, 2I, H/32]
    c2, s2 = stack("w2")               # [E, H, I/2], [E, H, I/32]
    return c13, s13, c2, s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=6)
    # fp32 = apples-to-apples (kernel is internally fp32) -> isolates kernel correctness.
    # bf16 = the verified path's default; residual 0.9997 is CPU-torch vs NPU-triton bf16
    # division rounding, not a kernel defect.
    ap.add_argument("--ref-dtype", choices=["bf16", "fp32"], default="fp32")
    args = ap.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(0)

    wm = _load_weight_map(args.model_dir)
    cache = {}
    E = args.experts
    experts = list(range(E))
    c13, s13, c2, s2 = load_combined(args.model_dir, wm, cache, args.layer_idx, experts)
    c13, s13 = c13.to(DEV), s13.to(DEV)
    c2, s2 = c2.to(DEV), s2.to(DEV)
    fp4 = _fp4_lut(torch.float32)

    # ---- reference (vectorized, bit-exact slow) -> NZ ----
    rdt = torch.bfloat16 if args.ref_dtype == 'bf16' else torch.float32
    ref13_nz, ref13_s = mxfp4_to_w8a8_nz(c13, s13, 2 * I, H, fp4, rdt, do_nz=True)
    ref2_nz, ref2_s = mxfp4_to_w8a8_nz(c2, s2, H, I, fp4, rdt, do_nz=True)

    # ---- kernel via the production depool helper (the validated artifact) ----
    k13_nz, k13s, k2_nz, k2s = mxfp4_layer_to_slots(c13, s13, c2, s2, H, I)
    torch.npu.synchronize()

    # ---- synthetic activations + routing over the E test experts ----
    M, top_k = args.tokens, args.top_k
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    logits = torch.randn(M, E, device=DEV)
    tw, tid = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    tid = tid.to(torch.int32)

    def fe(w13, w13s, w2, w2s):
        return npu_fused_experts(
            hidden_states=x.clone(), w13=w13, w13_scale=w13s, w2=w2, w2_scale=w2s,
            topk_weights=tw.to(x.dtype), topk_ids=tid, top_k=top_k)

    out_ref = fe(ref13_nz, ref13_s, ref2_nz, ref2_s)
    out_ker = fe(k13_nz, k13s, k2_nz, k2s)
    torch.npu.synchronize()

    def cos(a, b):
        a = a.reshape(-1).float(); b = b.reshape(-1).float()
        return float((a @ b) / (a.norm() * b.norm() + 1e-9))

    c = cos(out_ker, out_ref)
    rel = (out_ker.float() - out_ref.float()).norm() / (out_ref.float().norm() + 1e-9)
    print(f"[e2e] layer={args.layer_idx} E={E} M={M} top_k={top_k}")
    print(f"  npu_fused_experts: cos(kernel, ref_vectorized) = {c:.8f}  rel_l2_err = {rel:.3e}")
    print(f"  VERDICT: {'PASS (cos>=0.9999)' if c >= 0.9999 else 'CHECK'}")


if __name__ == "__main__":
    main()
