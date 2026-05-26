from __future__ import annotations

import os
from pathlib import Path


def maybe_profile(out_dir: str | Path, name: str, fn, *, n_steps: int = 8, warmup: int = 2) -> Path:
    """N3: schedule + prof.step(); dump 到 OUT_DIR/profile/<name>/."""
    import torch_npu

    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    try:
        sched = torch_npu.profiler.schedule(wait=0, warmup=warmup, active=n_steps, repeat=1)
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=sched,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            record_shapes=True,
        ) as prof:
            for _ in range(warmup + n_steps):
                fn()
                prof.step()
            torch_npu.npu.synchronize()
    except Exception:
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
            record_shapes=True,
        ) as prof:
            for _ in range(warmup + n_steps):
                fn()
                prof.step()
            torch_npu.npu.synchronize()
    return target
