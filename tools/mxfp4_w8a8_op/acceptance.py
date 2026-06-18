#!/usr/bin/env python3
"""End-to-end ACCEPTANCE harness for the MXFP4 -> W8A8 operator (Ascend NPU).

Runs one real DeepSeek-V4-Flash MoE layer through the production op `npu_fused_experts` two ways:
  reference : MXFP4 --(golden, golden.py)--> int8+scale --transpose--> NZ --> npu_fused_experts
  candidate : MXFP4 --(YOUR KERNEL)--------> int8+scale --transpose--> NZ --> npu_fused_experts
and reports the output cosine + per-layer timing.

ACCEPTANCE (see SPEC.md §Acceptance):
  - end-to-end cosine(candidate, reference) >= 0.9999
  - full-layer (w13+w2, E=256) conversion time <= ~150 ms (warmup first)

Plug your kernel into `candidate_convert_layer()` below (one call site). Self-contained:
only numpy / torch / torch_npu / safetensors.

Run:  ASCEND_RT_VISIBLE_DEVICES=<free card> python3 acceptance.py --experts 32
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import FP4_TABLE, mxfp4_to_w8a8_golden  # noqa: E402

DEV = "npu"
NZ = 29  # ACL_FORMAT_FRACTAL_NZ
H, I = 4096, 2048


# ----------------------------------------------------------------------------
#  Production op (verbatim copy of sglang npu_fused_experts non-wna16 path,
#  inlined to avoid package imports). int8 weights are FRACTAL_NZ [E, IN, OUT];
#  scale is bf16 per-output-channel [E, OUT].
# ----------------------------------------------------------------------------
def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale,
                      topk_weights, topk_ids, top_k):
    od = hidden_states.dtype
    sd = od if od == torch.bfloat16 else torch.float32
    nt = hidden_states.shape[0]
    ne = w13.shape[0]
    row_idx = (torch.arange(0, nt * top_k, dtype=torch.int32, device=topk_weights.device)
               .view(top_k, -1).permute(1, 0).contiguous())
    hidden_states, exp_row, exp_eid = torch.ops.npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=nt)
    et = torch.ops.npu.npu_moe_compute_expert_tokens(exp_eid, ne).to(torch.int64)
    hidden_states, pts = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w13], scale=[w13_scale.to(sd)], per_token_scale=[pts],
        split_item=2, group_list_type=0, group_type=0, group_list=et, output_dtype=od)[0]
    hidden_states, pts = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, activate_left=True, quant_mode=1)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w2], scale=[w2_scale.to(sd)], per_token_scale=[pts],
        split_item=2, group_list_type=0, group_type=0, group_list=et, output_dtype=od)[0]
    return torch.ops.npu.npu_moe_finalize_routing(
        hidden_states, skip1=None, skip2=None, bias=None, scales=topk_weights,
        expanded_src_to_dst_row=exp_row, export_for_source_row=topk_ids)


# ----------------------------------------------------------------------------
#  Real MXFP4 weight loading (per expert) -> combined w13/w2 codes+scale (host u8)
# ----------------------------------------------------------------------------
def _as_u8(t):
    return (t if t.dtype == torch.uint8 else t.view(torch.uint8)).contiguous()


def load_layer_mxfp4(md: Path, layer: int, experts):
    wm = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]
    cache = {}

    def shard(k):
        if wm[k] not in cache:
            cache[wm[k]] = safe_open(md / wm[k], framework="pt")
        return cache[wm[k]]

    def stack(proj):
        cs, ss = [], []
        for e in experts:
            wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
            sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
            h = shard(wk)
            cs.append(_as_u8(h.get_tensor(wk)))
            ss.append(_as_u8(h.get_tensor(sk)))
        return torch.stack(cs), torch.stack(ss)

    c1, s1 = stack("w1")
    c3, s3 = stack("w3")
    c13 = torch.cat([c1, c3], dim=1)   # [E, 2I, H/2]
    s13 = torch.cat([s1, s3], dim=1)   # [E, 2I, H/32]
    c2, s2 = stack("w2")               # [E, H, I/2], [E, H, I/32]
    return c13, s13, c2, s2


# ----------------------------------------------------------------------------
#  REFERENCE conversion (golden): MXFP4 -> int8+scale -> transpose -> NZ
# ----------------------------------------------------------------------------
def golden_proj_to_nz(codes_u8_dev, scale_u8_dev, IN):
    E, OUT = codes_u8_dev.shape[0], codes_u8_dev.shape[1]
    cu = codes_u8_dev.cpu().numpy()
    su = scale_u8_dev.cpu().numpy()
    q = torch.empty((E, OUT, IN), dtype=torch.int8)
    s = torch.empty((E, OUT), dtype=torch.bfloat16)
    for e in range(E):
        qe, se = mxfp4_to_w8a8_golden(cu[e], su[e])
        q[e], s[e] = qe, se
    q_nz = torch_npu.npu_format_cast(q.to(DEV).transpose(1, 2).contiguous(), NZ)  # [E,IN,OUT]
    return q_nz, s.to(DEV)


# ----------------------------------------------------------------------------
#  CANDIDATE conversion -- *** PLUG YOUR KERNEL HERE ***
#  Must return (q_nz [E,IN,OUT] FRACTAL_NZ int8, scale_bf16 [E,OUT]) on DEV.
#  Default: falls back to golden so the harness runs out of the box.
# ----------------------------------------------------------------------------
def candidate_proj_to_nz(codes_u8_dev, scale_u8_dev, IN):
    # TODO(agent): replace with your AscendC kernel:
    #   q_int8 [E,OUT,IN], oscale_bf16 [E,OUT] = your_kernel(codes, scale)
    #   q_nz = torch_npu.npu_format_cast(q_int8.transpose(1,2).contiguous(), 29)
    #   return q_nz, oscale_bf16
    return golden_proj_to_nz(codes_u8_dev, scale_u8_dev, IN)


def convert_layer(fn, c13, s13, c2, s2):
    w13_nz, s13b = fn(c13, s13, H)
    w2_nz, s2b = fn(c2, s2, I)
    return w13_nz, s13b, w2_nz, s2b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(0)

    md = Path(args.model_dir)
    experts = list(range(args.experts))
    c13, s13, c2, s2 = load_layer_mxfp4(md, args.layer, experts)
    c13, s13, c2, s2 = (x.to(DEV) for x in (c13, s13, c2, s2))
    E = args.experts

    ref = convert_layer(golden_proj_to_nz, c13, s13, c2, s2)
    cand = convert_layer(candidate_proj_to_nz, c13, s13, c2, s2)
    torch.npu.synchronize()

    M, top_k = args.tokens, args.top_k
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    tw, tid = torch.topk(torch.softmax(torch.randn(M, E, device=DEV), -1), top_k, -1)
    tid = tid.to(torch.int32)

    def fe(slots):
        w13, s13b, w2, s2b = slots
        return npu_fused_experts(x.clone(), w13, s13b, w2, s2b, tw.to(x.dtype), tid, top_k)

    out_ref, out_cand = fe(ref), fe(cand)
    torch.npu.synchronize()
    a, b = out_cand.reshape(-1).float(), out_ref.reshape(-1).float()
    cos = float((a @ b) / (a.norm() * b.norm() + 1e-9))
    print(f"[acceptance] layer={args.layer} E={E} M={M} top_k={top_k}")
    print(f"  end-to-end cos(candidate, reference) = {cos:.8f}   "
          f"{'PASS' if cos >= 0.9999 else 'FAIL'} (>=0.9999)")
    print("  (default candidate == golden -> cos should be 1.0; plug your kernel into "
          "candidate_proj_to_nz)")


if __name__ == "__main__":
    main()
