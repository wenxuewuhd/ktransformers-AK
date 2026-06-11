#!/usr/bin/env python3
"""Experiment (1): ACCURACY of the streaming MXFP4->W8A8 NPU path for DSv4-Flash MoE.

Three paths, all run through the SAME NPU expert operator (npu_fused_experts, the
prefill-coherent kernel inlined from sglang fused_moe_method_npu.py), with the SAME
random bf16 activations + routing (fixed seed). Only the WEIGHTS differ:

  cur : weights = native W8A8 safetensors (int8 per-output-channel) -> NPU op.
        This is the current production path.
  new : weights = MXFP4 GGUF -> dequant_native to bf16 -> re-quantize to W8A8
        (int8, per-OUTPUT-CHANNEL scale) -> NPU op.   <-- the new streaming path (a).
        Double loss: 4-bit values + scale granularity per-block-32 -> per-channel.
  ref : weights = native W8A8 dequant'd to bf16 -> pure bf16 expert compute on NPU.
        This is the "truth" of the CURRENT precision (what cur is quantizing).

Reported: cos(cur,ref), cos(new,ref), cos(new,cur), relerr each. Core conclusion =
how much new loses RELATIVE TO cur (MXFP4 path vs current W8A8 path).

NOTE: 'cur' (native W8A8) and 'new' (MXFP4) come from two DIFFERENT quantizations of
the same fp16 master, so they are not bit-identical even before our requant — this is
expected; cos(new,cur) measures the combined gap and cos(*,ref) the gap to bf16 math.

Pure operator-level, no server, reads checkpoints directly. Card via ASCEND_RT_VISIBLE_DEVICES.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve()
_TOOLS = _HERE.parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import json  # noqa: E402
from safetensors import safe_open  # noqa: E402
from verify_mxfp4_layer import dequant_native  # noqa: E402


# inlined from convert_mxfp4_layer_to_gguf (avoid its module-level gguf.MXFP4 enum dep)
def _load_weight_map(model_dir: Path):
    return json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def _detect_experts_prefix(weight_map, layer_idx):
    for layer_prefix in (f"model.layers.{layer_idx}.", f"layers.{layer_idx}."):
        for k in weight_map:
            if k.startswith(layer_prefix) and ".experts." in k and ".shared_experts" not in k \
                    and k.endswith(".w1.weight") and ".experts.0." in k:
                before, _ = k.split(".experts.0.", 1)
                return before + ".experts"
    raise ValueError(f"No native MXFP4 experts found for layer {layer_idx}")


def _open_shard(model_dir, weight_map, cache, key):
    shard = weight_map[key]
    if shard not in cache:
        cache[shard] = safe_open(model_dir / shard, framework="pt")
    return cache[shard]


def _as_u8(t):
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return t.contiguous().numpy()

DEV = "npu"
NZ = 29  # ACL_FORMAT_FRACTAL_NZ
H = 4096
I = 2048


# ----- the NPU expert operator (verbatim from kernel_decode_vs_prefill.py) -----
def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights,
                      topk_ids, top_k):
    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
               .view(top_k, -1).permute(1, 0).contiguous())
    hidden_states, expanded_row_idx, expanded_expert_idx = torch.ops.npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens)
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(expanded_expert_idx, num_experts).to(torch.int64)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w13], scale=[w13_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale], split_item=2, group_list_type=0, group_type=0,
        group_list=expert_tokens, output_dtype=original_dtype)[0]
    hidden_states, pertoken_scale = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, activate_left=True, quant_mode=1)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w2], scale=[w2_scale.to(scale_dtype)],
        per_token_scale=[pertoken_scale], split_item=2, group_list_type=0, group_type=0,
        group_list=expert_tokens, output_dtype=original_dtype)[0]
    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states, skip1=None, skip2=None, bias=None, scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx, export_for_source_row=topk_ids)
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def quant_per_outchannel_bf16(w_bf16):
    """w_bf16 [E,OUT,IN] (rows=output channels) -> int8 [E,OUT,IN], scale bf16 [E,OUT]."""
    amax = w_bf16.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = (w_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.bfloat16)


# ---------------- weight loaders (CPU numpy/torch, then move to NPU) ----------------
def load_w8a8_experts(w8a8_dir: Path, layer_idx: int, experts):
    """Native W8A8 safetensors -> (w13_q[E,2I,H] int8, w13_s[E,2I] bf16,
                                    w2_q[E,H,I] int8,  w2_s[E,H] bf16),
       and bf16 dequant masters (w13_bf[E,2I,H], w2_bf[E,H,I]) for the ref path."""
    wm = _load_weight_map(w8a8_dir)
    cache = {}
    w13_q, w13_s, w2_q, w2_s = [], [], [], []
    w13_bf, w2_bf = [], []
    for e in experts:
        pref = f"layers.{layer_idx}.ffn.experts.{e}"
        cols = {}
        for proj in ("w1", "w3", "w2"):
            wk = f"{pref}.{proj}.weight"
            sk = f"{pref}.{proj}.weight_scale"
            h = _open_shard(w8a8_dir, wm, cache, wk)
            wq = h.get_tensor(wk).to(torch.int8)        # [OUT,IN]
            ws = h.get_tensor(sk).to(torch.float32)     # [OUT,1]
            cols[proj] = (wq, ws)
        # stack gate(w1)+up(w3) along OUT -> [2I,H]
        g_q, g_s = cols["w1"]; u_q, u_s = cols["w3"]; d_q, d_s = cols["w2"]
        w13q = torch.cat([g_q, u_q], dim=0)             # [2I,H]
        w13s = torch.cat([g_s, u_s], dim=0).squeeze(-1)  # [2I]
        w13_q.append(w13q); w13_s.append(w13s)
        w2_q.append(d_q); w2_s.append(d_s.squeeze(-1))
        # bf16 master = int8 * per-channel scale
        w13_bf.append((w13q.float() * w13s.unsqueeze(-1)).to(torch.bfloat16))
        w2_bf.append((d_q.float() * d_s).to(torch.bfloat16))
    return (torch.stack(w13_q), torch.stack(w13_s).to(torch.bfloat16),
            torch.stack(w2_q), torch.stack(w2_s).to(torch.bfloat16),
            torch.stack(w13_bf), torch.stack(w2_bf))


def load_mxfp4_experts_bf16(native_dir: Path, layer_idx: int, experts):
    """Native MXFP4 safetensors -> bf16 masters w13_bf[E,2I,H], w2_bf[E,H,I] via dequant_native."""
    wm = _load_weight_map(native_dir)
    prefix = _detect_experts_prefix(wm, layer_idx)
    cache = {}
    w13_bf, w2_bf = [], []
    for e in experts:
        cols = {}
        for proj in ("w1", "w3", "w2"):
            wk = f"{prefix}.{e}.{proj}.weight"
            sk = f"{prefix}.{e}.{proj}.scale"
            h = _open_shard(native_dir, wm, cache, wk)
            deq = dequant_native(_as_u8(h.get_tensor(wk)), _as_u8(h.get_tensor(sk)))  # [OUT,IN] f32
            cols[proj] = torch.from_numpy(deq)
        w13 = torch.cat([cols["w1"], cols["w3"]], dim=0).to(torch.bfloat16)  # [2I,H]
        w13_bf.append(w13)
        w2_bf.append(cols["w2"].to(torch.bfloat16))                         # [H,I]
    return torch.stack(w13_bf), torch.stack(w2_bf)


def to_npu_kernel_weights(w13_q, w13_s, w2_q, w2_s):
    """int8 [E,OUT,IN] -> NPU NZ [E,IN,OUT] + scale on device."""
    w13_k = torch_npu.npu_format_cast(w13_q.transpose(1, 2).contiguous().to(DEV), NZ)  # [E,H,2I]
    w2_k = torch_npu.npu_format_cast(w2_q.transpose(1, 2).contiguous().to(DEV), NZ)    # [E,I,H]
    return w13_k, w13_s.to(DEV), w2_k, w2_s.to(DEV)


def bf16_ref_moe(x, w13_bf, w2_bf, topk_ids, tw, top_k):
    """Pure bf16(->f32 math) MoE reference (swiglu, activate_left, gate=first half)."""
    M = x.shape[0]
    out = torch.zeros(M, H, device=DEV, dtype=torch.float32)
    for t in range(M):
        for j in range(top_k):
            e = int(topk_ids[t, j]); wgt = float(tw[t, j])
            h = x[t].float() @ w13_bf[e].float().t()   # [2I]
            g, u = h[:I], h[I:]
            act = (g * torch.sigmoid(g)) * u
            y = act @ w2_bf[e].float().t()             # [H]
            out[t] += wgt * y
    return out


def cos(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def relerr(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w8a8-dir", type=Path, default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8"))
    ap.add_argument("--native-dir", type=Path, default=Path("/workspace/models/DeepSeekV4/DeepSeek-V4-Flash"))
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--num-experts", type=int, default=32, help="how many experts (0..N-1) to load")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.npu.set_device(0)  # ASCEND_RT_VISIBLE_DEVICES already remaps
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    E = args.num_experts
    experts = list(range(E))
    print(f"[load] layer={args.layer_idx} E={E} tokens={args.tokens} topk={args.top_k} seed={args.seed}")

    # cur: native W8A8 (+ its bf16 master for ref)
    w13q_c, w13s_c, w2q_c, w2s_c, w13bf_ref, w2bf_ref = load_w8a8_experts(args.w8a8_dir, args.layer_idx, experts)
    # new: MXFP4 -> bf16 -> requant per-output-channel W8A8
    w13bf_m, w2bf_m = load_mxfp4_experts_bf16(args.native_dir, args.layer_idx, experts)
    w13q_n, w13s_n = quant_per_outchannel_bf16(w13bf_m)
    w2q_n, w2s_n = quant_per_outchannel_bf16(w2bf_m)

    # sanity: how close are the two bf16 masters (W8A8-dequant vs MXFP4-dequant)?
    print(f"[sanity] cos(masters w13) = {cos(w13bf_ref, w13bf_m):.5f}  cos(masters w2) = {cos(w2bf_ref, w2bf_m):.5f}")

    # move weights to NPU kernel layout
    w13_kc, w13s_kc, w2_kc, w2s_kc = to_npu_kernel_weights(w13q_c, w13s_c, w2q_c, w2s_c)
    w13_kn, w13s_kn, w2_kn, w2s_kn = to_npu_kernel_weights(w13q_n, w13s_n, w2q_n, w2s_n)
    w13bf_ref = w13bf_ref.to(DEV); w2bf_ref = w2bf_ref.to(DEV)

    # activations + routing
    M = args.tokens
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=DEV)[:args.top_k] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, args.top_k, device=DEV, dtype=torch.bfloat16)
    tw = tw / tw.sum(1, keepdim=True)

    def run(w13_k, w13_s, w2_k, w2_s):
        return npu_fused_experts(x.clone(), w13_k, w13_s, w2_k, w2_s,
                                 tw.clone(), topk_ids.clone(), args.top_k).float()

    torch.npu.synchronize()
    o_cur = run(w13_kc, w13s_kc, w2_kc, w2s_kc); torch.npu.synchronize()
    o_new = run(w13_kn, w13s_kn, w2_kn, w2s_kn); torch.npu.synchronize()
    o_ref = bf16_ref_moe(x, w13bf_ref, w2bf_ref, topk_ids, tw, args.top_k); torch.npu.synchronize()

    print("\n=== ACCURACY (MXFP4->W8A8 streaming path) ===")
    print(f"  cos(cur, ref) = {cos(o_cur, o_ref):.5f}   relerr = {relerr(o_cur, o_ref):.4f}   <- current W8A8 vs bf16 truth")
    print(f"  cos(new, ref) = {cos(o_new, o_ref):.5f}   relerr = {relerr(o_new, o_ref):.4f}   <- MXFP4 path vs bf16 truth")
    print(f"  cos(new, cur) = {cos(o_new, o_cur):.5f}   relerr = {relerr(o_new, o_cur):.4f}   <- MXFP4 vs current W8A8")
    print(f"\n  CONCLUSION: new loses {cos(o_cur,o_ref)-cos(o_new,o_ref):+.5f} cos vs cur relative to bf16 truth.")
    print(f"  per-token cos(new,cur): " + " ".join(f"{cos(o_new[t],o_cur[t]):.3f}" for t in range(min(M, 12))))


if __name__ == "__main__":
    main()
