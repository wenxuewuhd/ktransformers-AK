#!/usr/bin/env python3
"""
多进程批量：将 DeepSeek-V4-Flash 原生 MXFP4 的多个 MoE 层分别转为独立 GGUF。

每层专家独占一个 safetensors shard（见 handoff §3），所以单层转换只读一个文件，
层间用 ProcessPoolExecutor 并行。输出 ``{output_dir}/{prefix}{L}{suffix}.gguf``
（默认 ``dsv4_layer{L}_mxfp4.gguf``）。无损 nibble repack，详见
``convert_mxfp4_layer_to_gguf.py``。

容量：MXFP4 ~3.42 GiB/layer ×43 = ~147 GiB（Q8_0 是 ~295 GiB，正好一半）。

示例（全量 43 层）::

  /usr/local/python3.11.14/bin/python3 tools/batch_convert_mxfp4_layers_mp.py \\
    --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash \\
    --output-dir /workspace/models/cache \\
    --layer-start 0 --layer-end 42 --jobs 4 --skip-existing --verify-sample 3
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _convert_script() -> Path:
    return _repo_root() / "tools" / "convert_mxfp4_layer_to_gguf.py"


def _run_one_layer(py, model_dir, layer_idx, output_path, num_experts, hidden_size, moe_intermediate_size):
    cmd = [
        py, str(_convert_script()),
        "--input", model_dir,
        "--layer-idx", str(layer_idx),
        "--output", output_path,
        "--num-experts", str(num_experts),
        "--hidden-size", str(hidden_size),
        "--moe-intermediate-size", str(moe_intermediate_size),
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # pure numpy/torch CPU, skip torch_npu autoload
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = (proc.stdout or "")[-3000:]
    if proc.stderr:
        tail += "\n--- stderr ---\n" + proc.stderr[-3000:]
    return layer_idx, proc.returncode, tail


def _verify_sample_paths(paths: list[Path]) -> None:
    sys.path.insert(0, str(_repo_root() / "third_party" / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    for p in paths:
        if not p.is_file():
            print(f"[verify-sample] SKIP missing: {p}")
            continue
        reader = GGUFReader(str(p))
        print(f"[verify-sample] {p.name} ({p.stat().st_size/1e9:.3f} GB) tensors={len(reader.tensors)}")
        for t in reader.tensors:
            tt = t.tensor_type
            print(f"    {t.name} type={getattr(tt,'name',tt)} shape={list(t.shape)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="原生 MXFP4 模型目录（含 index.json）")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=42, help="含端点")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--name-prefix", type=str, default="dsv4_layer")
    ap.add_argument("--name-suffix", type=str, default="_mxfp4")
    ap.add_argument("--skip-existing", action="store_true", help="目标 >1GiB 则跳过")
    ap.add_argument("--verify-sample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_dir = args.input.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"ERROR: --input 不是目录: {model_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    min_skip = 1 << 30
    py = str(args.python.expanduser())

    layers = list(range(args.layer_start, args.layer_end + 1))
    tasks = []
    for lid in layers:
        outp = out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf"
        if args.skip_existing and outp.is_file() and outp.stat().st_size > min_skip:
            print(f"[batch] skip existing {outp.name}")
            continue
        tasks.append((py, str(model_dir), lid, str(outp), args.num_experts, args.hidden_size,
                      args.moe_intermediate_size))

    if not tasks:
        print("[batch] 无待转换任务（均 skip）")
    else:
        print(f"[batch] model={model_dir} layers={args.layer_start}..{args.layer_end} pending={len(tasks)} jobs={args.jobs}")
        failed = []
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
                    print(f"[batch] layer {layer_idx} FAILED rc={rc}\n{tail[-1500:]}")
                else:
                    print(f"[batch] layer {layer_idx} OK")
        if failed:
            print(f"[batch] 失败 {len(failed)} 层: {failed[:10]}", file=sys.stderr)
            return 1

    if args.verify_sample > 0:
        rnd = random.Random(args.seed)
        k = min(args.verify_sample, len(layers))
        sample = sorted(rnd.sample(layers, k)) if k > 0 else []
        paths = [out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf" for lid in sample]
        print(f"[batch] verify-sample k={k} layers={sample}")
        _verify_sample_paths(paths)

    print("[batch] 全部结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
