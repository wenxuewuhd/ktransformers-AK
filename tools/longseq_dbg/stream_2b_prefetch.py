#!/usr/bin/env python3
"""子任务 2b:双缓冲 H2D 预取器 + 多流带宽探测(handoff §3.5 D 步骤 2)。

建立在 2a 验证的流式地基上(NZ int8 权重经 pinned DDR round-trip,bitwise 正确)。

本脚本三部分:
  (1) 双缓冲多层流式:2 个 HBM weight slot + 独立 copy stream,算第 L 层时异步 H2D 预取
      L+1 进另一 slot。量 overlapped 总时 vs serial(无重叠)总时。
      预期:copy-bound 下 overlap 只省 compute(~2%);双缓冲主要保正确性(算/搬不撞 slot)。
  (2) 多 copy stream 聚合带宽:1/2/4 并发 H2D 流,看 910B3 是否多 DMA 引擎 → 提升有效
      PCIe 带宽(直接攻 308ms/层 瓶颈;§3.4 的"再快只能动 PCIe 侧")。
  (3) 正确性:沿用 2a 的 NZ round-trip,确认多层流水输出与权重常驻一致。
"""
import argparse, time, sys, os
import torch, torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_2a_roundtrip import (
    build_layer_weights, process_after_loading, run_experts, npu_format_cast, to_pinned, npu_fmt,
)


def make_pool(K, E, H, I, dev):
    """建 K 层的 pinned DDR 池(NZ int8 权重),scales 留 NPU(极小)。释放 HBM 权重。"""
    pool = []
    scales = []
    for L in range(K):
        w13, w2, s13, s2 = build_layer_weights(E, H, I, dev, seed=L)
        w13_nz, w2_nz, s13b, s2b = process_after_loading(w13, w2, s13, s2)
        del w13, w2
        pool.append((to_pinned(w13_nz), to_pinned(w2_nz)))
        scales.append((s13b, s2b))
        del w13_nz, w2_nz
        torch.npu.empty_cache()
    return pool, scales


def alloc_nz_slot(shape, dev):
    t = torch.empty(shape, dtype=torch.int8, device=dev)
    return npu_format_cast(t)


def double_buffer_stream(pool, scales, hidden, topk_ids, topk_w, top_k, dev):
    """2-slot 双缓冲流水。关键(2b 发现):H2D 必须走 **default stream**(全带宽 ~21GB/s);
    side stream 的 H2D 只有 ~10GB/s。故 compute 放 side stream,default 背靠背 H2D 全带宽。
    返回 (总ms, 末层输出)。"""
    K = len(pool)
    w13_shape, w2_shape = pool[0][0].shape, pool[0][1].shape
    slot13 = [alloc_nz_slot(w13_shape, dev), alloc_nz_slot(w13_shape, dev)]
    slot2 = [alloc_nz_slot(w2_shape, dev), alloc_nz_slot(w2_shape, dev)]
    compute_stream = torch.npu.Stream()
    default = torch.npu.current_stream()
    load_done = [None] * K
    compute_done = [None] * K

    torch.npu.synchronize()
    t0 = time.perf_counter()
    last = None
    for L in range(K):
        buf = L % 2
        # H2D 第 L 层进 slot[buf](default stream,全带宽)。slot[buf] 上次被 L-2 compute 用,须等其算完。
        if L >= 2:
            default.wait_event(compute_done[L - 2])
        slot13[buf].copy_(pool[L][0], non_blocking=True)
        slot2[buf].copy_(pool[L][1], non_blocking=True)
        load_done[L] = default.record_event()
        # compute 第 L 层(side stream),等其 H2D 完;与下一层 H2D 在 default 上并行
        compute_stream.wait_event(load_done[L])
        with torch.npu.stream(compute_stream):
            last = run_experts(slot13[buf], slot2[buf], scales[L][0], scales[L][1],
                               hidden, topk_ids, topk_w, top_k)
        compute_done[L] = compute_stream.record_event()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3, last


def serial_stream(pool, scales, hidden, topk_ids, topk_w, top_k, dev):
    """无重叠基线:逐层 H2D 然后 compute,串行。"""
    K = len(pool)
    slot13 = alloc_nz_slot(pool[0][0].shape, dev)
    slot2 = alloc_nz_slot(pool[0][1].shape, dev)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for L in range(K):
        slot13.copy_(pool[L][0]); slot2.copy_(pool[L][1])
        run_experts(slot13, slot2, scales[L][0], scales[L][1], hidden, topk_ids, topk_w, top_k)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3


def multistream_bw(pool, dev, nstreams):
    """N 个并发 copy stream 各搬一层权重,量聚合 H2D 带宽。"""
    n = min(nstreams, len(pool))
    slots = [(alloc_nz_slot(pool[i][0].shape, dev), alloc_nz_slot(pool[i][1].shape, dev)) for i in range(n)]
    streams = [torch.npu.Stream() for _ in range(n)]
    bytes_total = sum(pool[i][0].numel() + pool[i][1].numel() for i in range(n))
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for rep in range(2):
        for i in range(n):
            with torch.npu.stream(streams[i]):
                slots[i][0].copy_(pool[i][0], non_blocking=True)
                slots[i][1].copy_(pool[i][1], non_blocking=True)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / 2
    del slots; torch.npu.empty_cache()
    return bytes_total / 1e9 / dt  # GB/s aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6, help="流式层数")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--I", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    torch.npu.set_device(0)
    dev = "npu"
    E, M, H, I, top_k, K = args.E, args.M, args.H, args.I, args.topk, args.K
    print(f"# 2b prefetch  K={K}层 E={E} M={M} H={H} I={I} top_k={top_k}")

    pool, scales = make_pool(K, E, H, I, dev)
    per_layer_gb = (pool[0][0].numel() + pool[0][1].numel()) / 1e9
    print(f"# pinned DDR 池: {K} 层 × {per_layer_gb:.2f}GB = {K*per_layer_gb:.1f}GB")

    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=dev) * 0.1
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(M)]).to(dev).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.bfloat16)

    # (1) 双缓冲 vs serial
    db_ms, _ = double_buffer_stream(pool, scales, hidden, topk_ids, topk_w, top_k, dev)
    torch.npu.empty_cache()
    se_ms = serial_stream(pool, scales, hidden, topk_ids, topk_w, top_k, dev)
    torch.npu.empty_cache()
    print(f"# [双缓冲] {K}层 overlapped={db_ms:.1f}ms ({db_ms/K:.1f}ms/层) | "
          f"serial={se_ms:.1f}ms ({se_ms/K:.1f}ms/层) | overlap 省 {(se_ms-db_ms)/se_ms*100:.1f}%")
    print(f"# → 全 43 层外推 prefill MoE: overlapped ≈ {db_ms/K*43/1000:.1f}s (vs CPU ~1058s)")

    # (2) 多流带宽
    print("# [多 copy stream 聚合 H2D 带宽]")
    for ns in (1, 2, 4):
        if ns <= K:
            bw = multistream_bw(pool, dev, ns)
            print(f"#   {ns} stream: {bw:.1f} GB/s aggregate")


if __name__ == "__main__":
    main()
