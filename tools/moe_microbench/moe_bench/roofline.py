from __future__ import annotations

import time

from moe_bench.config import MoEConfig


def routed_weight_bytes(cfg: MoEConfig) -> tuple[int, int]:
    """grouped GEMM up / down weight bytes."""
    H, I, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_active_experts
    bpw = cfg.roofline_bytes_per_weight
    up = E * H * (2 * I) * bpw
    down = E * I * H * bpw
    return up, down


def shared_weight_bytes(cfg: MoEConfig) -> tuple[int, int]:
    """dense shared expert gate_up + down weight bytes."""
    H, S = cfg.hidden_size, cfg.shared_intermediate_size
    bpw = cfg.roofline_bytes_per_weight
    gate_up = H * (2 * S) * bpw
    down = S * H * bpw
    return gate_up, down


def lower_bound_us(weight_bytes: int, hbm_tb_s: float) -> float:
    return weight_bytes / (hbm_tb_s * 1e12) * 1e6


def utilizations(actual_us: float, weight_bytes: int, cfg: MoEConfig) -> dict:
    lb_peak = lower_bound_us(weight_bytes, cfg.roofline_hbm_peak_tb_s)
    lb_eff = lower_bound_us(weight_bytes, cfg.roofline_hbm_effective_tb_s)
    return {
        "lb_peak_us": lb_peak,
        "lb_effective_us": lb_eff,
        "util_vs_peak": actual_us / lb_peak if lb_peak else None,
        "util_vs_achievable": actual_us / lb_eff if lb_eff else None,
    }


def measure_hbm_effective_tb_s(device, size_mb: int = 256) -> float:
    """D2 校准: x.clone() 双向 memcpy 测可达 HBM 带宽.

    注意: 双向 memcpy (read+write)，pure-read 场景可能偏低 10–20%。
    """
    import torch

    dev = torch.device(device)
    n = size_mb * 1024 * 1024 // 2
    x = torch.empty(n, dtype=torch.bfloat16, device=dev)
    if dev.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()
        for _ in range(5):
            _ = x.clone()
        torch_npu.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = x.clone()
        torch_npu.npu.synchronize()
    else:
        for _ in range(5):
            _ = x.clone()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = x.clone()
    dt = (time.perf_counter() - t0) / 20
    bytes_moved = 2 * size_mb * 1024 * 1024
    return bytes_moved / dt / 1e12
