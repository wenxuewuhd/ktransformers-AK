#!/usr/bin/env python3
"""
多进程批量：将 DeepSeek-V4-Flash W8A8 的多个 MoE 层分别转为独立 GGUF（Phase 1.1）。

- 层间并行：使用 ProcessPoolExecutor，每个 worker 子进程调用
  `convert_w8a8_to_gguf_q8_0.py` 处理一层（避免在父进程里重复占内存）。
- 输出文件名：``{output_dir}/{name_prefix}{L}{name_suffix}.gguf``。
  - ``--quant q8_0``（默认）：name_suffix 默认 ``""``，与历史路径兼容
    （``dsv4_layer3.gguf``）。
  - ``--quant bf16``：name_suffix 默认 ``_bf16``，例如 ``dsv4_layer3_bf16.gguf``，
    便于与同层 Q8_0 共存。
- 全部完成后可选抽样用 GGUFReader 校验（兼容 NumPy 2）。

示例 — Q8_0 全量::

  /usr/local/python3.11.14/bin/python3 tools/batch_convert_w8a8_layers_mp.py \\
    --input /workspace/models/DeepSeek-V4-Flash-W8A8 \\
    --output-dir /workspace/models/cache \\
    --layer-start 0 --layer-end 42 \\
    --quant q8_0 \\
    --jobs 4 \\
    --verify-sample 3

示例 — BF16 全量（aarch64 上 Q8_0 仍 NaN 时走的路径，见 Handoff 附录 Z）::

  /usr/local/python3.11.14/bin/python3 tools/batch_convert_w8a8_layers_mp.py \\
    --input /workspace/models/DeepSeek-V4-Flash-W8A8 \\
    --output-dir /workspace/models/cache_bf16 \\
    --layer-start 0 --layer-end 42 \\
    --quant bf16 \\
    --jobs 2 \\
    --verify-sample 3

容量参考（W8A8 -> GGUF，per layer，DSv4-Flash 256 experts × 4096H × 2048I）：
  - Q8_0  : 6.85 GiB / layer ×43 = ~295 GiB
  - BF16  : 12.9 GiB / layer ×43 = ~555 GiB  ← 注意磁盘
注意：每层 BF16 ~12.9 GB；``--jobs`` 过大易打满内存与磁盘带宽，建议 2～4。
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _convert_script() -> Path:
    return _repo_root() / "tools" / "convert_w8a8_to_gguf_q8_0.py"


def _run_one_layer(
    py: str,
    model_dir: str,
    layer_idx: int,
    output_path: str,
    num_experts: int,
    expert_batch: int,
    hidden_size: int,
    moe_intermediate_size: int,
    quant: str,
) -> tuple[int, int, str]:
    """子进程入口：跑单层转换。返回 (layer_idx, returncode, tail_log)."""
    script = str(_convert_script())
    cmd = [
        py,
        script,
        "--input",
        model_dir,
        "--layer-idx",
        str(layer_idx),
        "--output",
        output_path,
        "--num-experts",
        str(num_experts),
        "--expert-batch",
        str(expert_batch),
        "--hidden-size",
        str(hidden_size),
        "--moe-intermediate-size",
        str(moe_intermediate_size),
        "--quant",
        quant,
    ]
    # convert_w8a8_to_gguf_q8_0.py 用 TORCH_DEVICE_BACKEND_AUTOLOAD=0 可绕开 torch_npu
    # 自加载（与 npu 卡无关），加速纯 numpy/torch CPU 转换；批量场景下默认打开。
    import os as _os
    env = _os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = ""
    if proc.stdout:
        tail += proc.stdout[-4000:]
    if proc.stderr:
        tail += "\n--- stderr ---\n" + proc.stderr[-4000:]
    return layer_idx, proc.returncode, tail


def _verify_sample_paths(paths: list[Path]) -> None:
    repo = _repo_root()
    gguf_py = repo / "third_party" / "llama.cpp" / "gguf-py"
    if not gguf_py.is_dir():
        raise RuntimeError(f"Missing {gguf_py}")
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # noqa: WPS433

    for p in paths:
        if not p.is_file():
            print(f"[verify-sample] SKIP missing: {p}")
            continue
        sz_gb = p.stat().st_size / (1024**3)
        reader = GGUFReader(str(p))
        names = [t.name for t in reader.tensors]
        print(f"[verify-sample] {p.name} ({sz_gb:.3f} GiB) tensors={len(names)}")
        for t in reader.tensors:
            tt = t.tensor_type
            tname = tt.name if hasattr(tt, "name") else str(tt)
            print(f"    {t.name} type={tname} shape={list(t.shape)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="HF 模型目录（含 index.json）")
    ap.add_argument("--output-dir", type=Path, required=True, help="输出目录，如 /workspace/models/cache")
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=42, help="含端点，与 layer-start 组成闭区间")
    ap.add_argument("--jobs", type=int, default=4, help="并行进程数（每层一个子进程）")
    ap.add_argument("--python", type=Path, default=Path(sys.executable), help="用于跑 convert 脚本的 Python")
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--expert-batch", type=int, default=32)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--name-prefix", type=str, default="dsv4_layer", help="输出文件名前缀")
    ap.add_argument(
        "--name-suffix",
        type=str,
        default=None,
        help=(
            "输出文件名后缀（不含 .gguf）。默认按 --quant 自动选：q8_0 -> ''，"
            "bf16 -> '_bf16'。手动传入可覆盖。"
        ),
    )
    ap.add_argument(
        "--quant",
        type=str,
        default="q8_0",
        choices=["q8_0", "bf16"],
        help=(
            "Output element type, passthrough to convert_w8a8_to_gguf_q8_0.py。"
            "q8_0 容量小 ~6.85 GiB/layer，但 aarch64 ARM_NEON 走 tinyBLAS_Q0_ARM 当前 NaN（见 Handoff 附录 Z.5）；"
            "bf16 容量 12.9 GiB/layer，aarch64 走 BF16×F32 tinyBLAS path（已修复 Z.2 + Z.4），数值健全。"
        ),
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 gguf 已存在且 >1GiB 则跳过（按 quant 容量判定的最小阈值）",
    )
    ap.add_argument("--verify-sample", type=int, default=3, help="结束后随机抽样验证的层数；0 关闭")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    args = ap.parse_args()

    if args.name_suffix is None:
        args.name_suffix = "" if args.quant == "q8_0" else f"_{args.quant}"
    # Skip threshold: BF16 ~12 GiB, Q8_0 ~6 GiB；用 1 GiB 已经足够区分 "完整 vs 半成品"
    min_skip_bytes = 1 << 30

    model_dir = args.input.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"ERROR: --input 不是目录: {model_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = _convert_script()
    if not script_path.is_file():
        print(f"ERROR: 找不到 {script_path}", file=sys.stderr)
        return 2

    layers = list(range(args.layer_start, args.layer_end + 1))
    if not layers:
        print("ERROR: --layer-start/--layer-end 形成空区间", file=sys.stderr)
        return 2
    py = str(args.python.expanduser())

    tasks: list[tuple] = []
    for lid in layers:
        outp = out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf"
        if args.skip_existing and outp.is_file() and outp.stat().st_size > min_skip_bytes:
            print(f"[batch] skip existing {outp.name}")
            continue
        tasks.append(
            (
                py,
                str(model_dir),
                lid,
                str(outp),
                args.num_experts,
                args.expert_batch,
                args.hidden_size,
                args.moe_intermediate_size,
                args.quant,
            )
        )

    if not tasks:
        print("[batch] 没有待转换任务（可能均被 skip-existing 跳过）")
    else:
        print(
            f"[batch] model={model_dir} layers={args.layer_start}..{args.layer_end} "
            f"pending={len(tasks)} jobs={args.jobs}"
        )
        failed: list[tuple[int, str]] = []
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futures = {ex.submit(_run_one_layer, *t): t[2] for t in tasks}
            for fut in as_completed(futures):
                lid = futures[fut]
                try:
                    layer_idx, rc, tail = fut.result()
                except Exception as e:
                    failed.append((lid, repr(e)))
                    print(f"[batch] layer {lid} worker exception: {e!r}")
                    continue
                if rc != 0:
                    failed.append((layer_idx, f"exit {rc}"))
                    print(f"[batch] layer {layer_idx} FAILED rc={rc}")
                    print(tail[-2000:] if tail else "(no output)")
                else:
                    print(f"[batch] layer {layer_idx} OK")

        if failed:
            print(f"[batch] 完成，失败 {len(failed)} 层: {failed[:10]}", file=sys.stderr)
            return 1

    if args.verify_sample > 0:
        done_layers = [lid for lid in layers]
        rnd = random.Random(args.seed)
        k = min(args.verify_sample, len(done_layers))
        sample = sorted(rnd.sample(done_layers, k)) if k > 0 else []
        paths = [
            out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf" for lid in sample
        ]
        print(f"[batch] verify-sample k={k} layers={sample}")
        _verify_sample_paths(paths)

    print("[batch] 全部结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
