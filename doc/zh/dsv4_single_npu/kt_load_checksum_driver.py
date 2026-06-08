#!/usr/bin/env python3
"""P1 精度对齐复现 driver（配套 DeepSeek-V4-Flash_CPU权重加载加速_P0-P1.md §5）。

只构造若干层 LlamafileMoEWrapper 并调用 load_weights()，让 C++ 在
KT_LOAD_CHECKSUM=1 时打印 m_local_{gate,up,down} 的 FNV-1a 校验和。
分别用 KT_PARALLEL_LOAD=1（并行）与 =0（串行）各跑一次，逐 (layer,tp) 比对，
应逐字节一致。不涉及 NPU 前向 / sglang。

用法：
  export PYTHONPATH=<repo>/kt-kernel:<repo>/kt-kernel/python
  export PYTHONDONTWRITEBYTECODE=1
  KT_LOAD_CHECKSUM=1 KT_PARALLEL_LOAD=1 python3 kt_load_checksum_driver.py 0,1,2,5,17,30,42 2>par.err
  KT_LOAD_CHECKSUM=1 KT_PARALLEL_LOAD=0 python3 kt_load_checksum_driver.py 0,1,2,5,17,30,42 2>ser.err
  diff <(grep KT_LOAD_CHECKSUM par.err|sort) <(grep KT_LOAD_CHECKSUM ser.err|sort)  # 应为空

注：KT_LOAD_CHECKSUM 是临时插桩（未入库）；复现时需先把 §5.1 的 checksum 段加回
C++ load_weights 并重编 .so。GGUF 路径/模型超参按 DeepSeek-V4-Flash-W8A8 写死，按需改。
"""
import os
import sys

os.environ.setdefault("KT_LOAD_CHECKSUM", "1")
os.environ.setdefault("KT_TIME_LOAD", "0")

import torch  # noqa: E402,F401  (kt_kernel 需要 torch 已 import)
from kt_kernel.utils.llamafile import LlamafileMoEWrapper  # noqa: E402

TEMPLATE = "/workspace/models/cache/dsv4_layer{layer_idx}.gguf"
LAYERS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2"])]

print(f"[driver] KT_PARALLEL_LOAD={os.environ.get('KT_PARALLEL_LOAD','<unset=parallel>')} layers={LAYERS}", flush=True)
for L in LAYERS:
    wp = TEMPLATE.format(layer_idx=L)
    w = LlamafileMoEWrapper(
        layer_idx=L,
        num_experts=256,
        num_experts_per_tok=6,
        hidden_size=4096,
        moe_intermediate_size=2048,
        gpu_experts_mask=None,
        cpuinfer_threads=24,
        threadpool_count=8,
        weight_path=wp,
        chunked_prefill_size=2048,
        method="LLAMAFILE",
    )
    w.load_weights()
    print(f"[driver] layer {L} load_weights() done", flush=True)
print("[driver] DONE", flush=True)
