#!/usr/bin/env python3
"""子任务 2a:DDR pinned 权重池 + 单层流式 round-trip 验证(handoff §3.5 D 步骤 2)。

目标:验证"把一层全 E 个专家的 int8 权重摆进 pinned DDR → H2D 进 NPU → 跑生产
prefill 算子 npu_fused_experts → 数值正确"这条流式地基,不碰 forward 编排、不动 B、零重编。

生产 W8A8 NPU 专家路径(已核实):
  权重 layout(NPUW8A8Int8DynamicMoEMethod.process_weights_after_loading):
    w13: int8 [E, 2I, H] --transpose(1,2)--> [E, H, 2I] --npu_format_cast--> FRACTAL_NZ
    w2 : int8 [E, H, I ] --transpose(1,2)--> [E, I, H ] --npu_format_cast--> FRACTAL_NZ
    w13_scale: fp32 [E,2I,1]--squeeze-->[E,2I]--> bf16 ;  w2_scale 同理 [E,H]
  prefill 算子(npu_fused_experts):init_routing -> dynamic_quant -> grouped_matmul(w13,scale,
    per_token) -> dequant_swiglu_quant -> grouped_matmul(w2) -> finalize_routing。

核心待验:FRACTAL_NZ 的 int8 权重能否经 pinned DDR round-trip 后仍喂给 grouped_matmul?
  Path-1: 存 NZ 字节进 DDR,H2D 后直接用(最快)。
  Path-2: 存 ND 字节进 DDR,H2D 后在 NPU 上 npu_format_cast(ND->NZ)再用(稳妥兜底,+~5ms/层)。
"""
import argparse, time
import torch, torch_npu

# Inlined to avoid sglang quantization package circular import. Faithful copies of:
#   sglang .../hardware_backend/npu/utils.py:npu_format_cast (FRACTAL_NZ=29)
#   sglang .../npu/quantization/fused_moe_method_npu.py:npu_fused_experts (prefill chain)
_ACL_FORMAT_FRACTAL_NZ = 29
_ACL_FORMAT_ND = 2


def npu_format_cast(t):
    return torch_npu.npu_format_cast(t, _ACL_FORMAT_FRACTAL_NZ)


def npu_fused_experts(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, top_k):
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx = (torch.arange(0, num_tokens * top_k, dtype=torch.int32, device=topk_weights.device)
               .view(top_k, -1).permute(1, 0).contiguous())
    hidden_states, expanded_row_idx, expanded_expert_idx = torch.ops.npu.npu_moe_init_routing(
        hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens)
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(expanded_expert_idx, num_experts).to(torch.int64)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w13], scale=[w13_scale.to(scale_dtype)], per_token_scale=[pertoken_scale],
        split_item=2, group_list_type=0, group_type=0, group_list=expert_tokens, output_dtype=original_dtype)[0]
    hidden_states, pertoken_scale = torch.ops.npu.npu_dequant_swiglu_quant(
        hidden_states, activate_left=True, quant_mode=1)
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states], weight=[w2], scale=[w2_scale.to(scale_dtype)], per_token_scale=[pertoken_scale],
        split_item=2, group_list_type=0, group_type=0, group_list=expert_tokens, output_dtype=original_dtype)[0]
    return torch.ops.npu.npu_moe_finalize_routing(
        hidden_states, skip1=None, skip2=None, bias=None, scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx, export_for_source_row=topk_ids)


def build_layer_weights(E, H, I, device, seed=0):
    """造一层随机 int8 专家权重 + channel scale(ND 布局,模拟 checkpoint 已加载未 process)。"""
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randint(-127, 127, (E, 2 * I, H), generator=g, dtype=torch.int8)
    w2 = torch.randint(-127, 127, (E, H, I), generator=g, dtype=torch.int8)
    s13 = (torch.rand(E, 2 * I, 1, generator=g) * 0.02 + 1e-3)
    s2 = (torch.rand(E, H, 1, generator=g) * 0.02 + 1e-3)
    return (w13.to(device), w2.to(device), s13.to(device), s2.to(device))


def process_after_loading(w13, w2, s13, s2):
    """复刻 NPUW8A8Int8DynamicMoEMethod.process_weights_after_loading(在 NPU 上)。"""
    w13_nz = npu_format_cast(w13.transpose(1, 2).contiguous())  # [E,H,2I] -> NZ
    w2_nz = npu_format_cast(w2.transpose(1, 2).contiguous())    # [E,I,H]  -> NZ
    s13_bf16 = s13.squeeze(-1).to(torch.bfloat16)
    s2_bf16 = s2.squeeze(-1).to(torch.bfloat16)
    return w13_nz, w2_nz, s13_bf16, s2_bf16


def run_experts(w13, w2, s13, s2, hidden, topk_ids, topk_w, top_k):
    return npu_fused_experts(
        hidden_states=hidden, w13=w13, w13_scale=s13, w2=w2, w2_scale=s2,
        topk_weights=topk_w, topk_ids=topk_ids, top_k=top_k,
    )


def to_pinned(t):
    """D2H 把 NPU 张量字节搬到 pinned host。"""
    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    host.copy_(t)
    return host


def npu_fmt(t):
    try:
        return int(torch_npu.get_npu_format(t))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--I", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    torch.npu.set_device(0)
    dev = "npu"
    E, M, H, I, top_k = args.E, args.M, args.H, args.I, args.topk
    print(f"# 2a round-trip  E={E} M={M} H={H} I={I} top_k={top_k}")

    # 输入
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=dev) * 0.1
    g = torch.Generator().manual_seed(1)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(M)]).to(dev).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.bfloat16)

    # 1) 造权重 + process_weights_after_loading（= 生产 resident 形态），D2H 到 pinned DDR 池后释放 HBM。
    #    这样 HBM 上同时只驻 ~1 层权重（贴近流式实际：只双缓冲，不全驻）。
    w13, w2, s13, s2 = build_layer_weights(E, H, I, dev)
    w13_nz, w2_nz, s13b, s2b = process_after_loading(w13, w2, s13, s2)
    print(f"# w13_nz shape={tuple(w13_nz.shape)} dtype={w13_nz.dtype} npu_format={npu_fmt(w13_nz)} "
          f"(ND=2, FRACTAL_NZ=29)")
    del w13, w2; torch.npu.empty_cache()

    # reference（权重 resident 时的输出）+ DDR pinned 池（NZ 字节）。然后释放 HBM 权重。
    ref = run_experts(w13_nz, w2_nz, s13b, s2b, hidden, topk_ids, topk_w, top_k).clone()
    torch.npu.synchronize()
    nz_w13_host, nz_w2_host = to_pinned(w13_nz), to_pinned(w2_nz)
    nz_bytes = nz_w13_host.numel() + nz_w2_host.numel()
    print(f"# reference norm={ref.float().norm().item():.4f}")
    print(f"# pinned DDR 池: w13 {nz_w13_host.numel()/1e6:.0f}MB + w2 {nz_w2_host.numel()/1e6:.0f}MB "
          f"= {nz_bytes/1e9:.2f}GB (pinned={nz_w13_host.is_pinned()})")
    del w13_nz, w2_nz; torch.npu.empty_cache()

    # 2) 流式 round-trip：新 slot ← H2D(pinned NZ 字节) → 跑生产算子 → 对 reference。
    slot13 = torch.empty(nz_w13_host.shape, dtype=torch.int8, device=dev)
    slot2 = torch.empty(nz_w2_host.shape, dtype=torch.int8, device=dev)
    slot13 = npu_format_cast(slot13); slot2 = npu_format_cast(slot2)  # 让 slot 也是 NZ 容器
    slot13.copy_(nz_w13_host); slot2.copy_(nz_w2_host)
    print(f"# H2D 后 slot13 npu_format={npu_fmt(slot13)}(期望 29=NZ 字节直存成功)")
    out1 = run_experts(slot13, slot2, s13b, s2b, hidden, topk_ids, topk_w, top_k)
    torch.npu.synchronize()
    d1 = (out1.float() - ref.float()).abs().max().item()
    ok1 = torch.allclose(out1.float(), ref.float(), atol=1e-2, rtol=1e-2)
    print(f"# [Path-1 NZ字节直存→H2D→算子] max_abs_diff={d1:.3e} allclose={ok1}  <<< 流式地基")

    # 3) 计时：整层 NZ 权重 H2D + grouped_matmul compute(交叉验证 §3.4)。
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(3):
        slot13.copy_(nz_w13_host); slot2.copy_(nz_w2_host)
    torch.npu.synchronize(); h2d_ms = (time.perf_counter() - t0) / 3 * 1e3
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(3):
        run_experts(slot13, slot2, s13b, s2b, hidden, topk_ids, topk_w, top_k)
    torch.npu.synchronize(); cmp_ms = (time.perf_counter() - t0) / 3 * 1e3
    print(f"# [计时] 整层 H2D {nz_bytes/1e9:.2f}GB={h2d_ms:.1f}ms ({nz_bytes/1e9/(h2d_ms/1e3):.1f}GB/s) | "
          f"compute {cmp_ms:.1f}ms/层 | 节拍 max={max(h2d_ms,cmp_ms):.1f}ms copy-bound={h2d_ms>cmp_ms}")


if __name__ == "__main__":
    main()
