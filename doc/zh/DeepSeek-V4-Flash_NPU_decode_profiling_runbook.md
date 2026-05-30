# DeepSeek-V4-Flash NPU Decode Profiling Runbook

> 容器重启 / 代码恢复后按此文档拉起服务并复现 seq32k profiling。  
> 原始 msprof 数据：`tools/npu_results_dbg/seq32k_decode32/`、`seq200_decode/`。

---

## 0. 前置

```bash
export REPO=/workspace/code/ktransformer/ktransformers-AK
cd "$REPO"
apt-get update && apt-get install -y libhwloc15   # 新容器一次
bash tools/p27_e2e_preflight.sh                   # 必须 PASS
npu-smi info                                      # 选空闲卡
```

补丁检查（恢复后应 >0）：

```bash
grep -c SGLANG_NPU_PROFILE_ENABLE third_party/sglang/python/sglang/srt/environ.py
grep -c "if q.dim() == 4" third_party/sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
grep -c kt_llamafile_sgemm kt-kernel/operators/llamafile/moe.hpp
```

---

## 1. 日常推理（Graph，无 profiling）

```bash
SGLANG_NPU_PROFILE_ENABLE=0 \
PORT=8001 ASCEND_RT_VISIBLE_DEVICES=7 \
./tools/p27_launch_ds4flash_npu_num_expert_0.sh
```

另开终端：

```bash
PORT=8001 bash tools/p27_curl_f2_prompts.sh
```

---

## 2. seq32k decode #32 profiling（推荐）

**终端 1 — 起服务**

```bash
SGLANG_NPU_PROFILE_ENABLE=1 \
SGLANG_NPU_PROFILE_DECODE_TOKEN=32 \
SGLANG_NPU_PROFILE_DIR=./tools/npu_results_dbg/seq32k_decode32 \
SGLANG_NPU_PROFILE_LEVEL=0 \
SGLANG_NPU_PROFILE_ANALYSE=0 \
CHUNKED_PREFILL_SIZE=32768 \
PORT=8001 ASCEND_RT_VISIBLE_DEVICES=2 \
./tools/p27_launch_ds4flash_npu_longcontext.sh
```

**终端 2 — 发 workload**

```bash
PORT=8001 PROMPT_LEN=32768 MAX_NEW=64 bash tools/p27_curl_long_prompt_sweep.sh
```

约束：

- `MAX_NEW >= SGLANG_NPU_PROFILE_DECODE_TOKEN`（否则 profile 不触发）
- `CHUNKED_PREFILL_SIZE=32768` 使 32k prefill 单 chunk（避免 compressor 跨 chunk bug）
- profiling 会强制 eager（`SGLANG_NPU_PROFILE_DISABLE_GRAPH=1`），输出可能退化，仅用于 trace

**离线解析**（不要在进程内 `ANALYSE=1`）：

```bash
PYBIN=/usr/local/python3.11.14/bin/python3.11
"$PYBIN" - <<'PY'
from torch_npu.profiler.profiler import analyse
analyse("tools/npu_results_dbg/seq32k_decode32/<host>_<pid>_..._ascend_pt")
PY
```

产物：`ASCEND_PROFILER_OUTPUT/{kernel_details.csv,operator_details.csv,trace_view.json}`。

---

## 3. token200 profiling（短 prompt）

```bash
SGLANG_NPU_PROFILE_ENABLE=1 \
SGLANG_NPU_PROFILE_DECODE_TOKEN=200 \
SGLANG_NPU_PROFILE_DIR=./tools/npu_results_dbg/seq200_decode \
PORT=8001 ASCEND_RT_VISIBLE_DEVICES=7 \
./tools/p27_launch_ds4flash_npu_num_expert_0.sh
```

终端 2：用 `tools/p27_long_context_decode_test.sh` 或多次 decode 攒 KV 到 200（见 Handoff）。

---

## 4. 关键结论（seq32k decode #32）

| 指标 | 值 |
|------|-----|
| Stage 墙钟 | ~318 ms |
| NPU Computing | ~28.5 ms (8.9%) |
| SparseAttnSharedkv | ~1.36 ms（43 层，几乎不随 32k KV 增长） |
| MatMul/GEMM (NPU busy) | ~12.5 ms（**不含** CPU MoE group matmul） |
| MoE routed experts | CPU KT GGUF（`--kt-num-gpu-experts 0`） |

Graph + msprof 生产路径：`tools/p27_msprof_graph_baseline_exec.sh`。

---

## 5. 常见故障

| 现象 | 处理 |
|------|------|
| `Argument list too long` | 用 `p27_curl_long_prompt_sweep.sh`（JSON 写文件） |
| `loc.numel()=1024 vs cache.shape[0]=513` | 加大 `CHUNKED_PREFILL_SIZE` 到 32768 |
| `head num(0)` / SparseAttn tiling failed | 确认 `ascend_backend.py` squeeze 补丁 |
| `[KT] kt_kernel not available` | `bash tools/p27_e2e_preflight.sh`；launch 须 source `p27_ensure_kt_kernel.sh` |
| profiler 后服务挂 / SIGBUS | 保持 `SGLANG_NPU_PROFILE_KEEP_EAGER_AFTER=1` |

---

*2026-05-29 恢复版；完整分析报告需从 msprof 数据重新生成。*
