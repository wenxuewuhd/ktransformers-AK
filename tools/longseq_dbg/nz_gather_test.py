#!/usr/bin/env python3
"""Confirm the real-topK bug: per-slot NZ-format gather scrambles experts.

Replicates _apply_dynamic_residency's resident-weight build two ways and runs the
real NPU MoE kernel on each:
  A (current code): per-slot copy between NZ tensors  stagA[s].copy_(pool_nz[top[s]])
  B (correct):      gather in ND then NZ-cast once     stagB = format_cast(w_nd[top])
If A diverges from B (and B matches the ND reference), the per-slot NZ gather is the bug.
"""
import os, torch, torch_npu
torch.npu.set_device(0); DEV="npu"; NZ=29
H=4096; I=2048; E=16; K=8; TOPK=6; M=8
torch.manual_seed(0)

def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, top_k):
    original_dtype = hidden_states.dtype; num_tokens = hidden_states.shape[0]; num_experts = w13.shape[0]
    row_idx = (torch.arange(0, num_tokens*top_k, dtype=torch.int32, device=topk_weights.device)
               .view(top_k,-1).permute(1,0).contiguous())
    hs, er, ee = torch.ops.npu.npu_moe_init_routing(hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens)
    et = torch.ops.npu.npu_moe_compute_expert_tokens(ee, num_experts).to(torch.int64)
    hs, pts = torch.ops.npu.npu_dynamic_quant(hs)
    hs = torch.ops.npu.npu_grouped_matmul(x=[hs], weight=[w13], scale=[w13_scale.to(original_dtype)],
        per_token_scale=[pts], split_item=2, group_list_type=0, group_type=0, group_list=et, output_dtype=original_dtype)[0]
    hs, pts = torch.ops.npu.npu_dequant_swiglu_quant(hs, activate_left=True, quant_mode=1)
    hs = torch.ops.npu.npu_grouped_matmul(x=[hs], weight=[w2], scale=[w2_scale.to(original_dtype)],
        per_token_scale=[pts], split_item=2, group_list_type=0, group_type=0, group_list=et, output_dtype=original_dtype)[0]
    return torch.ops.npu.npu_moe_finalize_routing(hs, skip1=None, skip2=None, bias=None, scales=topk_weights,
        expanded_src_to_dst_row=er, export_for_source_row=topk_ids)

def q(w):  # per-output-channel int8, w [E,OUT,IN]
    s=(w.abs().amax(2,keepdim=True).clamp(min=1e-8)/127.0)
    return (w/s).round().clamp(-127,127).to(torch.int8), s.squeeze(-1).to(torch.bfloat16)

# full pool of E experts (logical order), ND int8 [E,IN,OUT] layout for kernel
w13_bf=torch.randn(E,2*I,H,device=DEV,dtype=torch.bfloat16)*0.02
w2_bf =torch.randn(E,H,I,device=DEV,dtype=torch.bfloat16)*0.02
w13_q,w13_s=q(w13_bf); w2_q,w2_s=q(w2_bf)
w13_nd=w13_q.transpose(1,2).contiguous()  # [E,H,2I] ND int8
w2_nd =w2_q.transpose(1,2).contiguous()    # [E,I,H]  ND int8
pool13_nz=torch_npu.npu_format_cast(w13_nd.clone(), NZ)   # the streaming pool (NZ)
pool2_nz =torch_npu.npu_format_cast(w2_nd.clone(),  NZ)

top=torch.tensor([3,1,7,0,11,5,9,2],device=DEV)[:K]   # non-identity permutation (resident set)
x=torch.randn(M,H,device=DEV,dtype=torch.bfloat16)
# all M tokens route only into resident slots 0..K-1
tid=torch.stack([torch.randperm(K,device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
tw=torch.rand(M,TOPK,device=DEV,dtype=torch.bfloat16); tw=tw/tw.sum(1,keepdim=True)
s13_top=w13_s[top]; s2_top=w2_s[top]

# Method A: per-slot NZ copy (CURRENT _apply_dynamic_residency)
stagA13=torch_npu.npu_format_cast(torch.empty(K,H,2*I,dtype=torch.int8,device=DEV),NZ)
stagA2 =torch_npu.npu_format_cast(torch.empty(K,I,H,dtype=torch.int8,device=DEV),NZ)
for s,e in enumerate(top.cpu().tolist()):
    stagA13[s].copy_(pool13_nz[e]); stagA2[s].copy_(pool2_nz[e])
outA=npu_fused_experts(x.clone(),stagA13,s13_top,stagA2,s2_top,tw,tid,TOPK).float()

# Method B: gather in ND, NZ-cast once (proposed fix)
stagB13=torch_npu.npu_format_cast(w13_nd[top].contiguous(),NZ)
stagB2 =torch_npu.npu_format_cast(w2_nd[top].contiguous(),NZ)
outB=npu_fused_experts(x.clone(),stagB13,s13_top,stagB2,s2_top,tw,tid,TOPK).float()

# Reference: bf16 ND gather, exact
def ref():
    o=torch.zeros(M,H,device=DEV,dtype=torch.float32)
    for t in range(M):
        for j in range(TOPK):
            e=int(top[int(tid[t,j])]); wgt=float(tw[t,j])
            h=x[t].float()@w13_bf[e].float().t(); g,u=h[:I],h[I:]
            o[t]+=wgt*((g*torch.sigmoid(g))*u)@w2_bf[e].float().t()
    return o
outR=ref()
cos=lambda a,b:float((a.reshape(-1)@b.reshape(-1))/(a.norm()*b.norm()+1e-9))
print(f"top(perm)={top.cpu().tolist()}")
print(f"cos(A_perslotNZ , ref) = {cos(outA,outR):.5f}")
print(f"cos(B_NDgather  , ref) = {cos(outB,outR):.5f}")
print(f"cos(A , B)             = {cos(outA,outB):.5f}")
