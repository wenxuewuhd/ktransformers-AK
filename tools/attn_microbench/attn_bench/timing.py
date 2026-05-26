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

    # Backward-compatible aliases
    @property
    def mean_us(self) -> float:
        return self.device_mean_us

    @property
    def p50_us(self) -> float:
        return self.device_p50_us

    @property
    def p99_us(self) -> float:
        return self.device_p99_us

    def to_dict(self) -> dict:
        return {
            "mean": self.device_mean_us,
            "device_mean_us": self.device_mean_us,
            "device_p50_us": self.device_p50_us,
            "device_p95_us": self.device_p95_us,
            "device_p99_us": self.device_p99_us,
            "device_max_us": self.device_max_us,
            "device_std_us": self.device_std_us,
            "host_mean_us": self.host_mean_us,
            "mean_us": self.device_mean_us,
            "p50_us": self.device_p50_us,
            "p99_us": self.device_p99_us,
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


def bench_op(fn: Callable[[], None], warmup: int, repeat: int) -> TimingResult:
    import torch_npu

    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()

    samples_device: list[float] = []
    samples_host: list[float] = []
    for _ in range(repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        host_t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch_npu.npu.synchronize()
        host_t1 = time.perf_counter()
        samples_device.append(start.elapsed_time(end) * 1000.0)
        samples_host.append((host_t1 - host_t0) * 1e6)

    return TimingResult(
        device_mean_us=statistics.mean(samples_device),
        device_p50_us=statistics.median(samples_device),
        device_p95_us=_percentile(samples_device, 95),
        device_p99_us=_percentile(samples_device, 99),
        device_max_us=max(samples_device),
        device_std_us=statistics.pstdev(samples_device)
        if len(samples_device) > 1
        else 0.0,
        host_mean_us=statistics.mean(samples_host),
        samples_device=samples_device,
        samples_host=samples_host,
    )
