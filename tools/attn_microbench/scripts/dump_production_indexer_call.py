#!/usr/bin/env python3
"""Monkey-patch dump nsa_indexer.forward_npu_dsv4_fusion kwargs from one SGLang decode.

Does not modify nsa_indexer.py — uses PYTHONSTARTUP import hook in server subprocesses.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MICRO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MICRO_ROOT.parents[1]
HOOK = MICRO_ROOT / "scripts" / "dump_hook_startup.py"
DEFAULT_MODEL = "/workspace/models/DeepSeek-V4-Flash-W8A8"
DEFAULT_KT = "/workspace/models/cache/dsv4_layer{layer_idx}.gguf"


def _build_env(out_path: Path, port: int) -> dict:
    env = os.environ.copy()
    env["ATTN_DUMP_INDEXER_PATH"] = str(out_path.resolve())
    env["PYTHONSTARTUP"] = str(HOOK.resolve())
    env["REPO_ROOT"] = str(REPO_ROOT)
    py_paths = [
        str(REPO_ROOT / "third_party" / "sglang" / "python"),
        str(REPO_ROOT / "kt-kernel" / "python"),
    ]
    env["PYTHONPATH"] = ":".join(py_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", env.get("NPU_DEVICE_ID", "0"))
    env.setdefault("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
    env.setdefault("IS_DEEPSEEK_V4", "1")
    env.setdefault("USE_FUSED_COMPRESSOR", "1")
    env.setdefault("LI_KV_DTYPE_INT8", "1")
    env.setdefault("USE_PA_DECODE", "1")
    env.setdefault("USE_PA_PREFILL", "1")
    env.setdefault("SGLANG_SET_CPU_AFFINITY", "1")
    env.setdefault("TASK_QUEUE_ENABLE", "1")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"/usr/local/kml/lib:{ld}".rstrip(":")
    return env


def _wait_ready(port: int, timeout_s: int = 900) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)
    return False


def _one_request(port: int) -> None:
    payload = {
        "text": "Hello",
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        resp.read()


def main() -> int:
    p = argparse.ArgumentParser(description="Dump production indexer kwargs")
    p.add_argument("out", type=str, help="Output JSON path")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    p.add_argument("--kt-weight-path", default=os.environ.get("KT_GGUF_TEMPLATE", DEFAULT_KT))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "18091")))
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    py = os.environ.get(
        "PYTHON_BIN", "/usr/local/python3.11.14/bin/python3.11"
    )
    env = _build_env(out_path, args.port)
    cmd = [
        py,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--device",
        "npu",
        "--tensor-parallel-size",
        "1",
        "--page-size",
        "128",
        "--attention-backend",
        "ascend",
        "--quantization",
        "compressed-tensors",
        "--disable-shared-experts-fusion",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--disable-radix-cache",
        "--max-prefill-tokens",
        "4096",
        "--context-length",
        "8192",
        "--skip-server-warmup",
        "--kt-method",
        "LLAMAFILE",
        "--kt-num-gpu-experts",
        "0",
        "--kt-weight-path",
        args.kt_weight_path,
        "--kt-threadpool-count",
        "8",
        "--kt-cpuinfer",
        "24",
        "--max-running-requests",
        "1",
        "--chunked-prefill-size",
        "2048",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--disable-cuda-graph",
        "--log-level",
        "error",
    ]

    print(f"[dump] launching server on port {args.port} ...", flush=True)
    server_log = out_path.parent / "p1_3_server.log"
    with open(server_log, "w", encoding="utf-8") as slog:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            stdout=slog,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        if not _wait_ready(args.port, args.timeout):
            print(f"[dump][ERROR] server health timeout; see {server_log}", flush=True)
            return 2
        print("[dump] server ready, sending one decode request ...", flush=True)
        _one_request(args.port)
        for _ in range(60):
            if out_path.is_file() and out_path.stat().st_size > 0:
                print(f"[dump] wrote {out_path}", flush=True)
                return 0
            time.sleep(2)
        print("[dump][ERROR] hook did not write dump (patch may not have fired)", flush=True)
        return 3
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
