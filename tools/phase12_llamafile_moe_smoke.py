#!/usr/bin/env python3
"""
Phase 1.2 冒烟：单层 GGUF + KTMoEWrapper(LLAMAFILE)，全 CPU expert 走一遍 forward。

前置：
  - 已编译并可用 `kt_kernel_ext`（`cd kt-kernel && pip install -e .`，见下「与 torch_npu 版本」）
  - 单层 GGUF，张量名为 `blk.{L}.ffn_{gate,up,down}_exps.weight`，`--layer-idx` 必须与 GGUF 内 blk 编号一致

与 torch_npu / 昇腾：
  - `torch_npu` 与 `torch` 主版本必须一致（例如均为 2.8.x）。`pip install -e kt-kernel` 若把 torch 升到 2.9.x，会导致
    `import torch` 时自动加载 `torch_npu` 失败（undefined symbol）。请先恢复 `torch==2.8.0`（及匹配的 torchvision），
    再执行 `pip install -e . --no-deps` 仅注册 kt-kernel 而不改 torch。
  - 临时绕过（不推荐长期使用）：`export TORCH_DEVICE_BACKEND_AUTOLOAD=0` 可跳过有问题的 NPU 后端自加载，
    但 NPU 功能不可用，仅适合纯 CPU 冒烟。

示例::

  cd /workspace/code/ktransformer/ktransformers-AK/kt-kernel && \\
    pip install -e . --no-deps

  # 若尚未安装 torch 等，再装一次（勿与 torch_npu 要求的版本冲突）
    --gguf /workspace/models/cache/dsv4_layer3.gguf \\
    --layer-idx 3

  # Phase 2-B：同一进程内两个不同 GGUF（验证 LlamafileMoEWrapper 按路径缓存 loader）
    --gguf /workspace/models/cache/dsv4_layer3.gguf --layer-idx 3 \\
    --second-gguf /workspace/models/cache/dsv4_layer40.gguf --second-layer-idx 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _smoke_one(
    gguf: Path,
    layer_idx: int,
    args: argparse.Namespace,
    tag: str,
) -> int:
    """Run one KTMoEWrapper forward; return 0 on success."""
    if not gguf.is_file():
        print(f"ERROR: [{tag}] GGUF 不存在: {gguf}", file=sys.stderr)
        return 2

    try:
        from kt_kernel import KTMoEWrapper  # noqa: WPS433
    except ImportError as e:
        print(
            "ERROR: 无法 import kt_kernel。\n"
            "  请先安装可编辑包（推荐）：\n"
            "    cd /workspace/code/ktransformer/ktransformers-AK/kt-kernel && "
            "/usr/local/python3.11.14/bin/python3 -m pip install -e .\n"
            "  或确保已将编译好的 `kt_kernel_ext*.so` 放在 `kt-kernel/python/` 目录下（与源码同目录供 _cpu_detect 查找）。\n"
            f"  原始错误: {e}",
            file=sys.stderr,
        )
        return 2

    device = torch.device(args.device)
    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)

    print(
        f"[{tag}] gguf={gguf} layer={layer_idx} device={device} "
        f"H={args.hidden_size} inter={args.moe_intermediate_size} topk={args.num_experts_per_tok}"
    )

    wrapper = KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=str(gguf),
        chunked_prefill_size=args.chunked_prefill_size,
        method="LLAMAFILE",
        numa_nodes=list(range(args.threadpool_count)),
    )
    wrapper.load_weights()

    torch.manual_seed(0)
    hidden = torch.randn(args.batch, args.hidden_size, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, args.num_experts, (args.batch, args.num_experts_per_tok), dtype=torch.long, device=device)
    tw = torch.randn(args.batch, args.num_experts_per_tok, dtype=torch.float32, device=device)
    topk_weights = torch.softmax(tw, dim=-1)

    stream = _stream_handle(device)
    out = wrapper.forward(hidden, topk_ids, topk_weights, stream)
    if not torch.isfinite(out).all():
        print(f"[{tag}] FAIL: non-finite outputs shape={tuple(out.shape)}", file=sys.stderr)
        return 1
    print(f"[{tag}] OK forward out shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    return 0


def _stream_handle(device: torch.device) -> int:
    if device.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.current_stream(device).cuda_stream)
    try:
        import torch_npu  # noqa: F401

        if device.type == "npu" and torch.npu.is_available():
            return int(torch.npu.current_stream(device).npu_stream)
    except Exception:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-experts-per-tok", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--cpuinfer-threads", type=int, default=24)
    ap.add_argument("--threadpool-count", type=int, default=8)
    ap.add_argument("--chunked-prefill-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu", help="hidden/topk 张量所在设备：cpu | cuda | npu")
    ap.add_argument(
        "--second-gguf",
        type=Path,
        default=None,
        help="Phase 2-B：第二个 GGUF 路径；与 --second-layer-idx 同时指定时在首层之后同进程再跑一层",
    )
    ap.add_argument("--second-layer-idx", type=int, default=None, help="与 --second-gguf 配对")
    args = ap.parse_args()

    s_gguf, s_layer = args.second_gguf, args.second_layer_idx
    if (s_gguf is None) ^ (s_layer is None):
        print("ERROR: --second-gguf 与 --second-layer-idx 必须同时指定或同时省略", file=sys.stderr)
        return 2

    gguf = args.gguf.expanduser().resolve()
    rc = _smoke_one(gguf, args.layer_idx, args, "p12")
    if rc != 0:
        return rc

    if s_gguf is not None:
        gguf2 = s_gguf.expanduser().resolve()
        rc2 = _smoke_one(gguf2, s_layer, args, "p12b")
        if rc2 != 0:
            return rc2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
