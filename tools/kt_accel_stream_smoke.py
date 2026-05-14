#!/usr/bin/env python3
"""P2.2 轻量验证：在不跑整网的前提下检查 ``kt_accel`` 在当前设备上可用。

用法::

  python tools/kt_accel_stream_smoke.py

期望：在 CUDA 或 NPU 上打印 ``ok`` 并以 0 退出；仅 CPU 时跳过并以 0 退出。
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    if torch.cuda.is_available() and getattr(torch.version, "cuda", None):
        d = torch.device("cuda", torch.cuda.current_device())
    elif hasattr(torch, "npu") and torch.npu.is_available():
        d = torch.device("npu", torch.npu.current_device())
    else:
        print("skip: no cuda/npu")
        return 0

    from sglang.srt.utils.kt_accel import (
        kt_current_stream,
        kt_device_synchronize,
        kt_new_event,
        kt_new_stream,
        kt_stream_context,
    )

    s = kt_new_stream(d)
    e = kt_new_event(d)
    x = torch.zeros(4, device=d)
    with kt_stream_context(s, d):
        x += 1
    e.record(s)
    kt_current_stream(d).wait_event(e)
    kt_device_synchronize(d)
    assert int(x.sum().item()) == 4, x
    print(f"ok device={d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
