# NOTE: keep in sync with tools/attn_microbench/attn_bench/timing.py
# Changes here should be mirrored manually (no shared module by design).

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class TimingResult:
    device_mean_us: float
    device_p50_us: float
    device_p95_us: float
    device_p99_us: float
    device_max_us: float
    device_std_us: float
    host_mean_us: float
    samples_device: list[float]
    samples_host: list[float]

    def to_dict(self) -> dict:
        return {
            "device_mean_us": self.device_mean_us,
            "device_p50_us": self.device_p50_us,
            "device_p95_us": self.device_p95_us,
            "device_p99_us": self.device_p99_us,
            "device_max_us": self.device_max_us,
            "device_std_us": self.device_std_us,
            "host_mean_us": self.host_mean_us,
            "n": len(self.samples_device),
        }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _sync_device(device: torch.device) -> None:
    if device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_op(fn: Callable[[], None], warmup: int, repeat: int, device: torch.device | None = None) -> TimingResult:
    if device is None:
        device = torch.device("npu:0")

    for _ in range(warmup):
        fn()
    _sync_device(device)

    samples_device: list[float] = []
    samples_host: list[float] = []
    for _ in range(repeat):
        host_t0 = time.perf_counter()
        if device.type == "npu":
            import torch_npu

            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            _sync_device(device)
            host_t1 = time.perf_counter()
            samples_device.append(start.elapsed_time(end) * 1000.0)
        else:
            fn()
            _sync_device(device)
            host_t1 = time.perf_counter()
            samples_device.append((host_t1 - host_t0) * 1e6)
        samples_host.append((host_t1 - host_t0) * 1e6)

    return TimingResult(
        device_mean_us=statistics.mean(samples_device),
        device_p50_us=statistics.median(samples_device),
        device_p95_us=_percentile(samples_device, 95),
        device_p99_us=_percentile(samples_device, 99),
        device_max_us=max(samples_device),
        device_std_us=statistics.pstdev(samples_device) if len(samples_device) > 1 else 0.0,
        host_mean_us=statistics.mean(samples_host),
        samples_device=samples_device,
        samples_host=samples_host,
    )
