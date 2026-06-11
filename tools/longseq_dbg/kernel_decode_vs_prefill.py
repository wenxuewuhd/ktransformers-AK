#!/usr/bin/env python3
"""Pin the real-topK decode gibberish to a branch.

Feeds IDENTICAL (hidden_states, w13/scale, w2/scale, topk_weights/ids) to the two
NPU MoE kernels — prefill `npu_fused_experts` (known coherent) and decode
`npu_fused_experts_w8a8_decode` (single-card hybrid decode branch, suspect) — and
compares their outputs against a bf16-dequant reference.

If decode-kernel diverges from prefill-kernel/reference -> the decode kernel itself
is the bug (independent of any CPU/NPU hybrid combine). If both agree -> kernel is
fine and the bug is elsewhere (combine / logical->gpu indexing).

Pure operator-level, no server. W8A8: per-output-channel int8 weights + bf16 scale.
"""
import os, sys
import torch, torch_npu

torch.npu.set_device(0)
DEV = "npu"
NZ = 29  # ACL_FORMAT_FRACTAL_NZ

H = 4096
I = 2048
E = int(os.environ.get("E", "32"))     # resident expert count
TOPK = int(os.environ.get("TOPK", "6"))
M = int(os.environ.get("M", "8"))       # tokens
torch.manual_seed(0)

# Inlined verbatim from sglang fused_moe_method_npu.py to avoid circular import.
def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights,
                      topk_ids, top_k, **kwargs):
    use_wna16 = kwargs.get("use_wna16", False)
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

def npu_fused_experts_w8a8_decode(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights,
                                  topk_ids, top_k, **kwargs):
    num_tokens = hidden_states.shape[:-1].numel()
    first_expert_idx = 0
    last_expert_idx = w13.shape[0]
    global_num_experts = w13.shape[0]
    original_shape = hidden_states.shape
    group_list_type = 1
    sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = torch.ops.npu.npu_moe_init_routing_v2(
        hidden_states, topk_ids, active_num=num_tokens * top_k, expert_num=global_num_experts,
        expert_tokens_num_type=group_list_type, expert_tokens_num_flag=True,
        active_expert_range=[first_expert_idx, last_expert_idx], quant_mode=1)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[sorted_hidden_states], weight=[w13], scale=[w13_scale], per_token_scale=[pertoken_scale],
        group_list=expert_tokens, split_item=2, group_type=0, group_list_type=group_list_type,
        output_dtype=torch.bfloat16)[0]
    hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, quant_mode=1, activate_left=True)
    output = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w2], scale=[w2_scale], per_token_scale=[swiglu_out_scale],
        group_list=expert_tokens, split_item=2, group_type=0, group_list_type=group_list_type,
        output_dtype=torch.bfloat16)[0]
    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=output, sorted_indices=torch.abs(expanded_row_idx), probs=topk_weights)
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states

def quant_per_outchannel(w_bf16):
    # w_bf16: [E, OUT, IN]  (rows = output channels). per-output-channel int8.
    amax = w_bf16.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)   # [E,OUT,1]
    scale = (amax / 127.0)
    q = (w_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.bfloat16)   # q [E,OUT,IN] int8, scale [E,OUT] bf16

# --- build random experts (bf16 master), quantize to W8A8 ---
# w13 logical: [E, 2I, H] (out=2I, in=H);  w2 logical: [E, H, I] (out=H, in=I)
w13_bf = (torch.randn(E, 2 * I, H, device=DEV, dtype=torch.bfloat16) * 0.02)
w2_bf  = (torch.randn(E, H, I,   device=DEV, dtype=torch.bfloat16) * 0.02)
w13_q, w13_s = quant_per_outchannel(w13_bf)   # int8 [E,2I,H], scale [E,2I]
w2_q,  w2_s  = quant_per_outchannel(w2_bf)     # int8 [E,H,I],  scale [E,H]

# kernels want weight as [E, IN, OUT] for x[.,IN] @ w -> [.,OUT]; transpose then NZ-cast
w13_k = torch_npu.npu_format_cast(w13_q.transpose(1, 2).contiguous(), NZ)   # [E,H,2I] NZ
w2_k  = torch_npu.npu_format_cast(w2_q.transpose(1, 2).contiguous(), NZ)     # [E,I,H] NZ

# --- input + routing ---
x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
tw = torch.rand(M, TOPK, device=DEV, dtype=torch.bfloat16); tw = tw / tw.sum(1, keepdim=True)

def run(fn):
    return fn(hidden_states=x.clone(), w13=w13_k, w13_scale=w13_s, w2=w2_k, w2_scale=w2_s,
             topk_weights=tw.clone(), topk_ids=topk_ids.clone(), top_k=TOPK).float()

# --- bf16 reference (no quant, exact swiglu-MoE) ---
def ref():
    out = torch.zeros(M, H, device=DEV, dtype=torch.float32)
    for t in range(M):
        for j in range(TOPK):
            e = int(topk_ids[t, j]); wgt = float(tw[t, j])
            h = x[t].float() @ w13_bf[e].float().t()       # [2I]
            g, u = h[:I], h[I:]
            act = (g * torch.sigmoid(g)) * u                # swiglu (activate_left, gate=first half)
            y = act @ w2_bf[e].float().t()                  # [H]
            out[t] += wgt * y
    return out

torch.npu.synchronize()
o_pre = run(npu_fused_experts); torch.npu.synchronize()
o_dec = run(npu_fused_experts_w8a8_decode); torch.npu.synchronize()
o_ref = ref()

def cos(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))
def relerr(a, b):
    return float((a - b).norm() / (b.norm() + 1e-9))

print(f"shapes: x{tuple(x.shape)} w13_k{tuple(w13_k.shape)} w2_k{tuple(w2_k.shape)} E={E} TOPK={TOPK} M={M}")
print(f"cos(prefill, ref)  = {cos(o_pre, o_ref):.5f}   relerr={relerr(o_pre,o_ref):.4f}")
print(f"cos(decode , ref)  = {cos(o_dec, o_ref):.5f}   relerr={relerr(o_dec,o_ref):.4f}")
print(f"cos(decode , prefill) = {cos(o_dec, o_pre):.5f}   relerr={relerr(o_dec,o_pre):.4f}")
print(f"per-token cos(decode,prefill): " +
      " ".join(f"{cos(o_dec[t],o_pre[t]):.3f}" for t in range(M)))
