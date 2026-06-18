#!/usr/bin/env python3
"""验证 kt_stream_prefill 的新代码(chunked _nz_pinned + _ensure_slot + 池结构)与
2c-i 的整层 full-cast 路径数值一致。不依赖 sglang npu 模块(用 stream_2a 的 inlined 算子)。

即:同一层真实权重,(A)full process_after_loading → run_experts 作参考,
(B)kt_stream_prefill 的 chunked _nz_pinned → slot.copy_ → run_experts,二者应 bitwise 一致。
"""
import argparse, os, sys
import torch, torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 把 sglang 包加进路径以导入 kt_stream_prefill(其顶层 import 很轻:json/os/torch)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "third_party", "sglang", "python"))
from stream_2a_roundtrip import process_after_loading, run_experts, npu_fmt
from sglang.srt.layers.moe import kt_stream_prefill as ksp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--I", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    torch.npu.set_device(0); dev = "npu"
    E, M, H, I, top_k, L = args.E, args.M, args.H, args.I, args.topk, args.layer
    print(f"# test kt_stream_prefill  layer={L} E={E} M={M} NZ_CHUNK={ksp._NZ_CHUNK}")

    w13, w2, s13, s2 = ksp._load_layer_experts(L, E, H, I)
    print(f"# loaded ckpt layer: w13{tuple(w13.shape)} w2{tuple(w2.shape)}")

    g = torch.Generator().manual_seed(0)
    hidden = (torch.randn(M, H, generator=g, dtype=torch.float32)).to(torch.bfloat16).to(dev)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(M)]).to(dev).to(torch.int32)
    topk_w = torch.rand(M, top_k, generator=g, dtype=torch.bfloat16).to(dev)

    # (A) 参考:整层 full cast(2c-i 路径)
    w13n, w2n, s13b, s2b = process_after_loading(w13.to(dev), w2.to(dev), s13.to(dev), s2.to(dev))
    ref = run_experts(w13n, w2n, s13b, s2b, hidden, topk_ids, topk_w, top_k).float().clone()
    del w13n, w2n; torch.npu.empty_cache()

    # (B) kt_stream_prefill:chunked _nz_pinned → slot → run_experts
    h13 = ksp._nz_pinned(w13, dev)
    h2 = ksp._nz_pinned(w2, dev)
    print(f"# _nz_pinned: h13{tuple(h13.shape)} pinned={h13.is_pinned()} h2{tuple(h2.shape)}")
    slot13, slot2 = ksp._ensure_slot(h13.shape, h2.shape, dev)
    print(f"# slot13 npu_format={npu_fmt(slot13)}(29=NZ)")
    slot13.copy_(h13); slot2.copy_(h2)
    out = run_experts(slot13, slot2, s13b, s2b, hidden, topk_ids, topk_w, top_k).float()
    torch.npu.synchronize()

    d = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"# [chunked stream vs full-cast 参考] max_abs_diff={d:.3e} cosine={cos:.6f} allclose={ok}")
    print("# PASS" if ok else "# FAIL")


if __name__ == "__main__":
    main()
