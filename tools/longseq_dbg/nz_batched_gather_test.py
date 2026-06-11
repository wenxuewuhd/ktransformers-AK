#!/usr/bin/env python3
"""Test whether a BATCHED gather of resident experts from an NZ pool is correct
(and thus can replace the 32-per-layer per-slot copies that cost ~152s/switch).

Compares, via the real NPU MoE kernel:
  A per-slot loop (current fix, known correct): for s,e: dst[s].copy_(pool_nz[e])
  B fancy-index:        dst = pool_nz[top]
  C index_select:       dst = torch.index_select(pool_nz, 0, top)
all against a bf16 ND reference. Whichever batched form matches A/ref is safe + fast.
"""
import torch, torch_npu, time
torch.npu.set_device(0); DEV="npu"; NZ=29
H=4096; I=2048; E=64; K=32; TOPK=6; M=8
torch.manual_seed(0)

def fe(x,w13,w13s,w2,w2s,tw,tid,k):
    nt=x.shape[0]; ne=w13.shape[0]
    ri=(torch.arange(0,nt*k,dtype=torch.int32,device=tw.device).view(k,-1).permute(1,0).contiguous())
    hs,er,ee=torch.ops.npu.npu_moe_init_routing(x,row_idx=ri,expert_idx=tid,active_num=nt)
    et=torch.ops.npu.npu_moe_compute_expert_tokens(ee,ne).to(torch.int64)
    hs,pts=torch.ops.npu.npu_dynamic_quant(hs)
    hs=torch.ops.npu.npu_grouped_matmul(x=[hs],weight=[w13],scale=[w13s.to(torch.bfloat16)],per_token_scale=[pts],split_item=2,group_list_type=0,group_type=0,group_list=et,output_dtype=torch.bfloat16)[0]
    hs,pts=torch.ops.npu.npu_dequant_swiglu_quant(hs,activate_left=True,quant_mode=1)
    hs=torch.ops.npu.npu_grouped_matmul(x=[hs],weight=[w2],scale=[w2s.to(torch.bfloat16)],per_token_scale=[pts],split_item=2,group_list_type=0,group_type=0,group_list=et,output_dtype=torch.bfloat16)[0]
    return torch.ops.npu.npu_moe_finalize_routing(hs,skip1=None,skip2=None,bias=None,scales=tw,expanded_src_to_dst_row=er,export_for_source_row=tid)

def q(w):
    s=(w.abs().amax(2,keepdim=True).clamp(min=1e-8)/127.0); return (w/s).round().clamp(-127,127).to(torch.int8),s.squeeze(-1).to(torch.bfloat16)
w13b=torch.randn(E,2*I,H,device=DEV,dtype=torch.bfloat16)*0.02; w2b=torch.randn(E,H,I,device=DEV,dtype=torch.bfloat16)*0.02
w13q,w13s=q(w13b); w2q,w2s=q(w2b)
pool13=torch_npu.npu_format_cast(w13q.transpose(1,2).contiguous(),NZ)  # [E,H,2I] NZ
pool2 =torch_npu.npu_format_cast(w2q.transpose(1,2).contiguous(),NZ)
top=torch.tensor([3,1,7,0,11,5,9,2,17,33,40,12,4,8,6,15,22,31,44,50,2,9,1,3,7,0,5,11,13,19,23,29][:K],device=DEV)
top=torch.unique(top)[:K]; K=top.numel()
x=torch.randn(M,H,device=DEV,dtype=torch.bfloat16)
tid=torch.stack([torch.randperm(K,device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
tw=torch.rand(M,TOPK,device=DEV,dtype=torch.bfloat16); tw=tw/tw.sum(1,keepdim=True)
s13t=w13s[top]; s2t=w2s[top]

def ref():
    o=torch.zeros(M,H,device=DEV,dtype=torch.float32)
    for t in range(M):
        for j in range(TOPK):
            e=int(top[int(tid[t,j])]); wt=float(tw[t,j])
            h=x[t].float()@w13b[e].float().t(); g,u=h[:I],h[I:]
            o[t]+=wt*((g*torch.sigmoid(g))*u)@w2b[e].float().t()
    return o
oR=ref()
def empty_nz(k,a,b): return torch_npu.npu_format_cast(torch.empty(k,a,b,dtype=torch.int8,device=DEV),NZ)

# A: per-slot loop
A13=empty_nz(K,H,2*I); A2=empty_nz(K,I,H)
t=time.time()
for s,e in enumerate(top.cpu().tolist()): A13[s].copy_(pool13[e]); A2[s].copy_(pool2[e])
torch.npu.synchronize(); tA=time.time()-t
oA=fe(x.clone(),A13,s13t,A2,s2t,tw,tid,TOPK).float()
# B: fancy-index
t=time.time(); B13=pool13[top].contiguous(); B2=pool2[top].contiguous(); torch.npu.synchronize(); tB=time.time()-t
oB=fe(x.clone(),B13,s13t,B2,s2t,tw,tid,TOPK).float()
# C: index_select
t=time.time(); C13=torch.index_select(pool13,0,top).contiguous(); C2=torch.index_select(pool2,0,top).contiguous(); torch.npu.synchronize(); tC=time.time()-t
oC=fe(x.clone(),C13,s13t,C2,s2t,tw,tid,TOPK).float()
cos=lambda a,b:float((a.reshape(-1)@b.reshape(-1))/(a.norm()*b.norm()+1e-9))
print(f"K={K}")
print(f"A per-slot  : cos(ref)={cos(oA,oR):.5f}  time={tA*1e3:.1f}ms")
print(f"B fancy-idx : cos(ref)={cos(oB,oR):.5f}  cos(A)={cos(oB,oA):.5f}  time={tB*1e3:.1f}ms")
print(f"C idx_select: cos(ref)={cos(oC,oR):.5f}  cos(A)={cos(oC,oA):.5f}  time={tC*1e3:.1f}ms")

# D: ND round-trip gather (format_cast NZ->ND, fancy-index in ND (full BW), ND->NZ)
torch.npu.synchronize(); t=time.time()
nd13=torch_npu.npu_format_cast(pool13,2); nd2=torch_npu.npu_format_cast(pool2,2)   # NZ->ND
D13=torch_npu.npu_format_cast(nd13[top].contiguous(),NZ); D2=torch_npu.npu_format_cast(nd2[top].contiguous(),NZ)
torch.npu.synchronize(); tD=time.time()-t
oD=fe(x.clone(),D13,s13t,D2,s2t,tw,tid,TOPK).float()
print(f"D ND-gather : cos(ref)={cos(oD,oR):.5f}  cos(A)={cos(oD,oA):.5f}  time={tD*1e3:.1f}ms  (vs A per-slot {tA*1e3:.0f}ms)")
