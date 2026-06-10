#!/usr/bin/env python3
"""子任务 2c-i:真实 checkpoint → pinned NZ DDR 池加载器 + CPU fp32 数值对照。

读 DeepSeek-V4-Flash-W8A8 某层全 256 专家的 int8 权重,按生产 NPU layout 摆进 pinned DDR,
H2D 跑生产算子 npu_fused_experts,与 CPU fp32 dequant 参考对数值(补 2a 余项)。

checkpoint 命名(已核实):layers.{L}.ffn.experts.{e}.{w1|w2|w3}.weight(+.weight_scale)
  w1=gate int8[I,H] scale[I,1] | w3=up int8[I,H] scale[I,1] | w2=down int8[H,I] scale[H,1]
NPU layout(NPUCompressedTensorsW8A8Int8DynamicMoE):
  w13 = concat([w1(gate), w3(up)], dim=0) int8[E,2I,H](line340: w1→idx0 上半, w3→idx1 下半)
  w2  = down int8[E,H,I];  scale 同理 concat / 直传,squeeze→bf16
  process_weights_after_loading: w13/w2 transpose(1,2) + npu_format_cast(FRACTAL_NZ)
"""
import argparse, json, os, struct, time, sys
import torch, torch_npu
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_2a_roundtrip import process_after_loading, run_experts, to_pinned, npu_fmt

CKPT = "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8"


def load_layer_experts(layer, E, H, I):
    """从 checkpoint 读一层 E 个专家 → (w13 int8[E,2I,H], w2 int8[E,H,I], s13 fp32[E,2I,1], s2[E,H,1])。
    按 shard 文件分组读,每文件只开一次。"""
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    need = {}  # file -> list of (key, (e, kind))
    for e in range(E):
        for w, kind in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            for suf in ("weight", "weight_scale"):
                k = f"layers.{layer}.ffn.experts.{e}.{w}.{suf}"
                need.setdefault(idx[k], []).append((k, (e, kind, suf)))
    w13 = torch.empty(E, 2 * I, H, dtype=torch.int8)
    w2 = torch.empty(E, H, I, dtype=torch.int8)
    s13 = torch.empty(E, 2 * I, 1, dtype=torch.float32)
    s2 = torch.empty(E, H, 1, dtype=torch.float32)
    for fn, items in need.items():
        with safe_open(os.path.join(CKPT, fn), framework="pt") as f:
            for k, (e, kind, suf) in items:
                t = f.get_tensor(k)
                if suf == "weight":
                    if kind == "gate":   w13[e, 0:I] = t            # 上半
                    elif kind == "up":   w13[e, I:2 * I] = t        # 下半
                    else:                w2[e] = t                  # down
                else:
                    if kind == "gate":   s13[e, 0:I] = t
                    elif kind == "up":   s13[e, I:2 * I] = t
                    else:                s2[e] = t
    return w13, w2, s13, s2


def cpu_reference(w13, w2, s13, s2, hidden, topk_ids, topk_w, I):
    """CPU fp32 dequant MoE 参考(小 M)。w13/w2 int8 ND(未 transpose),scale per-out-channel。"""
    M, H = hidden.shape
    hid = hidden.float().cpu()
    out = torch.zeros(M, H, dtype=torch.float32)
    w13f = w13.float() * s13            # [E,2I,H]
    w2f = w2.float() * s2              # [E,H,I]
    for t in range(M):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j]); wgt = float(topk_w[t, j])
            gate = hid[t] @ w13f[e, 0:I].T          # [I]
            up = hid[t] @ w13f[e, I:2 * I].T        # [I]
            act = torch.nn.functional.silu(gate) * up
            out[t] += wgt * (act @ w2f[e].T)        # [H]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--M", type=int, default=64, help="对数值用小 M(CPU ref 慢)")
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--I", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    torch.npu.set_device(0); dev = "npu"
    E, M, H, I, top_k, L = args.E, args.M, args.H, args.I, args.topk, args.layer
    print(f"# 2c-i ckpt loader  layer={L} E={E} M={M}")

    t0 = time.perf_counter()
    w13, w2, s13, s2 = load_layer_experts(L, E, H, I)
    print(f"# 读 checkpoint 一层: w13{tuple(w13.shape)} w2{tuple(w2.shape)} "
          f"({(w13.numel()+w2.numel())/1e9:.2f}GB) in {time.perf_counter()-t0:.1f}s")

    # NPU layout + pinned DDR 池
    w13_nz, w2_nz, s13b, s2b = process_after_loading(w13.to(dev), w2.to(dev), s13.to(dev), s2.to(dev))
    print(f"# process→NZ: w13_nz fmt={npu_fmt(w13_nz)}(29=NZ)")
    h13, h2 = to_pinned(w13_nz), to_pinned(w2_nz)
    del w13_nz, w2_nz; torch.npu.empty_cache()

    # 输入 + 流式 H2D + 生产算子
    g = torch.Generator().manual_seed(0)
    hidden = (torch.randn(M, H, generator=g, dtype=torch.float32) * 1.0).to(torch.bfloat16).to(dev)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(M)]).to(dev).to(torch.int32)
    topk_w = torch.rand(M, top_k, generator=g, dtype=torch.float32).to(torch.bfloat16).to(dev)
    slot13 = torch.empty(h13.shape, dtype=torch.int8, device=dev); slot13 = torch_npu.npu_format_cast(slot13, 29)
    slot2 = torch.empty(h2.shape, dtype=torch.int8, device=dev); slot2 = torch_npu.npu_format_cast(slot2, 29)
    slot13.copy_(h13); slot2.copy_(h2)
    npu_out = run_experts(slot13, slot2, s13b, s2b, hidden, topk_ids, topk_w, top_k).float().cpu()
    torch.npu.synchronize()

    # CPU fp32 参考
    ref = cpu_reference(w13, w2, s13, s2, hidden, topk_ids, topk_w, I)
    cos = torch.nn.functional.cosine_similarity(npu_out.flatten(), ref.flatten(), dim=0).item()
    rel = ((npu_out - ref).norm() / ref.norm().clamp(min=1e-6)).item()
    print(f"# [NPU 流式 vs CPU fp32 dequant 参考] cosine={cos:.5f} rel_err={rel:.4f} "
          f"(int8 dynamic-quant 容差;cosine>0.99 即正确)")
    print(f"# npu_out norm={npu_out.norm():.2f} ref norm={ref.norm():.2f}")


if __name__ == "__main__":
    main()
