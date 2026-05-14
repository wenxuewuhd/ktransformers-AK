# Phase 0 / Phase 1 变更记录与复现手册

> **目的**：在动手 Phase 2（多 GGUF loader 等）之前，把截至 **Phase 1.2 完成** 为止的**所有工程向改动**、**系统依赖**与**逐步验证方法**固化到一处，便于新环境/新同事 **按步骤复现**。  
> **说明**：本文以**源码与脚本**为主；**权重文件**（HF W8A8、生成的 GGUF）体积巨大，需自备路径，不随仓库分发。

---

## 0. 两套 SGLang：本仓库 `third_party/sglang` 与「环境里的 sglang」

很多容器 / 机器上**另外**装有一份 SGLang（例如独立 checkout **`/sgl-workspace/sglang`**，或通过 **`pip install sglang`** 装进 `site-packages`）。这与本仓库内的 **`third_party/sglang/`** 往往是**不同代码树**；Phase 2 的补丁（`kt_ep_wrapper`、`kt_accel`、`deepseek_v4` 等）只存在于后者。

**复现与跑 P2.7 前请固定 PYTHONPATH**（将本仓库的 `python` 子目录放在最前）：

```bash
export REPO=/workspace/code/ktransformer/ktransformers-AK
export PYTHONPATH="$REPO/third_party/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
```

**自检**（应打印出带 `ktransformers-AK/third_party/sglang` 的路径，而不是 `/sgl-workspace/...` 或纯 site-packages）：

```bash
python3 -c "import sglang; print(sglang.__file__)"
```

本手册里凡写 `python -m sglang.launch_server`、`tools/run_p22_smoke_checks.sh` 等，均**默认**已按上式设置 `PYTHONPATH`；若你直接打 `sglang` 命令却看不到 Phase 2 行为，多半就是加载了**错误那份** SGLang。

---

## 1. Git 锚点（建议复现时先对齐）

在仓库根目录执行：

```bash
cd /workspace/code/ktransformer/ktransformers-AK
git log -3 --oneline
# 典型输出包含：
#   062b5bc feat(ds4-moe): Ascend/K920 kt-kernel 与 W8A8→GGUF 至 Phase1.2
#   b97b298 phase 0 done!
#   e95c561 gitignore: ...
```

若你基于 `origin/main` 做 diff，工程向改动集中在上述提交及子模块指针更新中。**请勿依赖**仓库内 `model/w8a8/*.safetensors` 等大文件作为复现前提（可能仅本地存在）；转换与冒烟只需指向你机器上的 **DeepSeek-V4-Flash-W8A8** 目录即可。

---

## 2. Phase 0：编译期适配（aarch64 + CANN + llamafile）

### 2.1 改动文件清单（代码）

| 路径 | 类型 | 摘要 |
|------|------|------|
| `kt-kernel/CMakeLists.txt` | 修改 | `KTRANSFORMERS_USE_ASCEND_NPU`、ARM feature 选项、动态 `-march`、查找 CANN 头与 `libascendcl.so` |
| `kt-kernel/cpu_backend/vendors/vendor.h` | 修改 | `#elif USE_ASCEND_NPU` → `ascend_npu.h` |
| `kt-kernel/cpu_backend/vendors/ascend_npu.h` | **新建** | `cudaStream_t`/`cudaLaunchHostFunc` → ACL `aclrtStream`/`aclrtLaunchCallback` 映射 |
| `kt-kernel/cpu_backend/cpuinfer.h` | 修改 | NPU 分支；`submit_with_cuda_stream` / `sync_with_cuda_stream` 在 CUDA 或 NPU 下启用 |
| `kt-kernel/setup.py` | 修改 | `CPUINFER_USE_ASCEND_NPU`、CANN 探测、`LLAMA_ARM_*`、aarch64 默认 `KML=OFF` |
| `kt-kernel/install.sh` | 修改 | aarch64 上探测 ARM feature / CANN、`CPUINFER_ENABLE_KML=OFF` 等 export |
| `third_party/llamafile/iqk_mul_mat_arm82.cpp` | 修改 | 取消注释 `#define iqk_mul_mat iqk_mul_mat_arm82` 等，修复 `_arm82` 符号缺失导致的 `dlopen` 失败 |
| `kt-kernel/operators/moe_kernel/mat_kernel/kml_kernel/**` | 恢复（可选） | 历史提交曾删除；`git checkout <parent> -- ...` 恢复后 **默认不参与编译**（SVE 汇编与 K920 不兼容），供 Phase 4 参考 |

### 2.2 系统依赖（非 Git，复现时按需）

| 依赖 | 用途 | 典型安装 |
|------|------|-----------|
| `libhwloc-dev` | `WorkerPool` NUMA | `apt-get install libhwloc-dev`（内网环境可能需 `--allow-unauthenticated` 等） |
| `numactl` | 查看 NUMA | `apt-get install numactl` |
| CANN 8.5 | `libascendcl.so` | 环境已有 `/usr/local/Ascend/ascend-toolkit/latest` |
| KML 2.5（可选） | Phase 4 | `/usr/local/kml/`，`ldconfig` 包含其 `lib` |

### 2.3 如何验证 Phase 0

**配置 + 编译**（二选一）：

```bash
cd /workspace/code/ktransformer/ktransformers-AK/kt-kernel
./install.sh build
```

或手动 CMake（与 `install.sh` 逻辑等价的核心开关）：

```bash
mkdir -p /tmp/kt_kernel_build && cd /tmp/kt_kernel_build
cmake /workspace/code/ktransformer/ktransformers-AK/kt-kernel \
  -DKTRANSFORMERS_USE_ASCEND_NPU=ON \
  -DLLAMA_NATIVE=OFF \
  -DLLAMA_ARM_DOTPROD=ON -DLLAMA_ARM_FP16=ON \
  -DLLAMA_ARM_SVE=OFF -DLLAMA_ARM_BF16=OFF -DLLAMA_ARM_I8MM=OFF \
  -DKTRANSFORMERS_CPU_USE_KML=OFF -DKTRANSFORMERS_CPU_MOE_KERNEL=OFF \
  -DPYTHON_EXECUTABLE=/usr/local/python3.11.14/bin/python3 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel "$(nproc)"
```

**Python 冒烟**（将 `sys.path` 指向 build 目录或 `pip install -e . --no-deps` 后已安装扩展）：

```bash
/usr/local/python3.11.14/bin/python3 - <<'PY'
import sys
sys.path.insert(0, "/tmp/kt_kernel_build")  # 或 site-packages
import kt_kernel_ext as ext
assert hasattr(ext, "moe") and hasattr(ext.moe, "MOE")
ci = ext.CPUInfer(8)
assert hasattr(ci, "submit_with_cuda_stream")
print("Phase0 smoke OK")
PY
```

**链接检查**：

```bash
ldd /tmp/kt_kernel_build/kt_kernel_ext*.so | grep -E 'ascendcl|hwloc|numa'
```

**期望**：`import` 无 `undefined symbol: iqk_mul_mat_moe_arm82`；`ldd` 可见 `libascendcl.so`（若启用 NPU 选项）。

---

## 3. Phase 1.1：W8A8 → GGUF Q8_0（单层 + 批量）

### 3.1 新增/修改文件

| 路径 | 摘要 |
|------|------|
| `tools/convert_w8a8_to_gguf_q8_0.py` | 单层：从 HF `index.json` 读 `layers.{L}.ffn.experts.*` 或 `model.layers.*`，反量化 W8A8，`gguf` 写 `blk.{L}.ffn_{gate,up,down}_exps.weight`（Q8_0） |
| `tools/batch_convert_w8a8_layers_mp.py` | **层间多进程**：`ProcessPoolExecutor` + 子进程调单层脚本；`--skip-existing`、`--verify-sample` |
| `third_party/llama.cpp/gguf-py/gguf/gguf_reader.py` | **NumPy 2**：`ndarray.newbyteorder` 移除后用 `_np_apply_byteorder` + `view(dtype.newbyteorder(...))` |

### 3.2 权重与输出路径约定

- **输入（HF）**：例如 `/workspace/models/DeepSeek-V4-Flash-W8A8`（含 `model.safetensors.index.json`）。
- **输出（GGUF）**：建议 **`/workspace/models/cache/dsv4_layer{L}.gguf`**（与权重同盘，避免占满 `/workspace` 根分区）。

### 3.3 如何验证 Phase 1.1

**单层转换 + GGUFReader**：

```bash
/usr/local/python3.11.14/bin/python3 \
  /workspace/code/ktransformer/ktransformers-AK/tools/convert_w8a8_to_gguf_q8_0.py \
  --input /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --layer-idx 3 \
  --output /workspace/models/cache/dsv4_layer3.gguf \
  --verify-reader
```

**43 层批量 + 抽样校验**：

```bash
/usr/local/python3.11.14/bin/python3 \
  /workspace/code/ktransformer/ktransformers-AK/tools/batch_convert_w8a8_layers_mp.py \
  --input /workspace/models/DeepSeek-V4-Flash-W8A8 \
  --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 \
  --jobs 4 \
  --skip-existing \
  --verify-sample 3
```

**期望**：每层约 **6.3～6.9 GiB** 单文件；`verify-sample` 打印 3 个张量名、`type=Q8_0`、shape 为 `[256,2048,4096]` 等（与 `GGUFLoader` 的 `reversed` 展示一致）。

---

## 4. Phase 1.2：Llamafile MoE 冒烟

### 4.1 新增文件

| 路径 | 摘要 |
|------|------|
| `tools/phase12_llamafile_moe_smoke.py` | `KTMoEWrapper` + `LLAMAFILE`，全 CPU expert，`forward` 一次，检查 `isfinite` |

### 4.2 Python 环境（与 Ascend 对齐）

- **`torch` 必须与 `torch_npu` 一致**（例如 **2.8.x**）。安装 kt-kernel 时使用：  
  `cd kt-kernel && pip install -e . --no-deps`  
  避免再次把 `torch` 升到 2.9.x。
- `kt-kernel/pyproject.toml` / `requirements.txt` 已将 `torch` 约束为 **`>=2.8.0,<2.9.0`**，`triton` 仅在 **x86_64** 作为依赖（aarch64 NPU 减少无谓大包）。

### 4.3 如何验证 Phase 1.2

```bash
/usr/local/python3.11.14/bin/python3 \
  /workspace/code/ktransformer/ktransformers-AK/tools/phase12_llamafile_moe_smoke.py \
  --gguf /workspace/models/cache/dsv4_layer3.gguf \
  --layer-idx 3
```

**期望**：日志含 `LlamafileMoEWrapper` / `TP MOE layer 3`，最后一行 **`[p12] OK forward out shape=(2, 4096)`**（默认 `batch=2`）。

**多抽几层**（`--gguf` 与 `--layer-idx` 一致）：

```bash
for L in 0 40; do
  /usr/local/python3.11.14/bin/python3 \
    /workspace/code/ktransformer/ktransformers-AK/tools/phase12_llamafile_moe_smoke.py \
    --gguf /workspace/models/cache/dsv4_layer${L}.gguf \
    --layer-idx ${L} || exit 1
done
```

---

## 5. 文档与其它仓库内引用

| 路径 | 摘要 |
|------|------|
| `doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md` | 总方案、Phase 2 预告、`/workspace/models/cache`、工具索引等 |
| `doc/zh/DeepSeek-V4-Flash-K920-Single-NPU-Spec.md` | K920 / 单 NPU 规格笔记（若存在） |
| `doc/zh/DeepSeek-V4-Flash-K920-Single-NPU-Handoff.md` | 交接片段（若存在） |

---

## 6. 一键复现检查表（建议顺序）

1. **SGLang 路径**：见 **§0**，确认 `import sglang` 指向本仓库 `third_party/sglang`（勿误用 `/sgl-workspace/sglang` 或 pip 全局包）。
2. **系统**：`numactl --hardware`、`npu-smi info`（可选）、`ldconfig -p | grep hwloc`。
3. **Phase 0**：`./install.sh build` 或 CMake 块 → Python `import kt_kernel_ext` → `ldd`。
4. **Phase 1.1**：单层 `convert_...` + `verify-reader`；再批量 `batch_...` + `verify-sample`。
5. **Phase 1.2**：`pip install -e kt-kernel --no-deps`（torch 已对齐前提下）→ `phase12_...py`。
6. **Phase 2-B**（多 GGUF 同进程）：见 §8.2，`phase12_...py` 带 `--second-gguf` / `--second-layer-idx`。
7. **P2.2**（KT EP 设备无关）：见 §8.4；`bash tools/run_p22_smoke_checks.sh`。
8. **NPU 回归**（可选）：`import torch; import torch_npu`、原 8 卡启动脚本抽测。

---

## 7. 已知坑（复现时必读）

| 现象 | 原因 | 处理 |
|------|------|------|
| `torch==2.8.0+cpu` 在华为源找不到 | 镜像源无 `+cpu` 本地版本号 | 使用 `torch==2.8.0` 或 PyTorch 官方 `--index-url https://download.pytorch.org/whl/cpu` |
| SSL 证书错误拉官方 wheel | 公司代理 MITM | `--trusted-host download.pytorch.org` 或离线 `.whl` + `pip install ./xxx.whl` |
| `pip install -e .` 升级 torch 破坏 torch_npu | `pyproject` 曾锁 2.9.x | 已放宽；安装 kt-kernel 用 **`--no-deps`**，torch 以镜像为准 |
| `import torch` 报 torch_npu undefined symbol | torch 与 torch_npu 主版本不一致 | 恢复 **torch 2.8.x** 与现有 **torch_npu** 匹配 |
| `--verify-reader` 崩在 NumPy 2 | `GGUFReader` 旧 API | 已修 `gguf_reader.py` 中 `_np_apply_byteorder` |
| `import kt_kernel_ext` 缺 `iqk_mul_mat_moe_arm82` | llamafile 宏未定义 | 已修 `iqk_mul_mat_arm82.cpp` |
| **改 Phase 2 后行为与文档不一致** | 实际跑的是 `/sgl-workspace/sglang` 或 pip 里的包，未加载 `third_party/sglang` | 见 `Phase0_Phase1_变更记录与复现手册.md` §0：`PYTHONPATH` + `print(sglang.__file__)` 自检 |

---

## 8. Phase 2-B：多 GGUF 同进程（已实现）

### 8.1 代码改动

| 路径 | 摘要 |
|------|------|
| `kt-kernel/python/utils/llamafile.py` | `_gguf_loader_instance` 改为 `_gguf_loaders_by_path: dict[str, GGUFLoader]`，键为 `os.path.realpath(weight_path)`；**相同路径复用** mmap，**不同路径各建** `GGUFLoader` |
| `tools/phase12_llamafile_moe_smoke.py` | 抽取 `_smoke_one`；新增 `--second-gguf` / `--second-layer-idx`（须成对），用于**同一进程**先后构造两层 `KTMoEWrapper` 并各跑一次 `forward` |

### 8.2 如何验证

在已有两份 per-layer GGUF（例如 `dsv4_layer3.gguf` 与 `dsv4_layer40.gguf`）的前提下：

```bash
/usr/local/python3.11.14/bin/python3 \
  /workspace/code/ktransformer/ktransformers-AK/tools/phase12_llamafile_moe_smoke.py \
  --gguf /workspace/models/cache/dsv4_layer3.gguf --layer-idx 3 \
  --second-gguf /workspace/models/cache/dsv4_layer40.gguf --second-layer-idx 40
```

**期望**：先出现 `[p12] OK forward ...`，再出现 `[p12b] OK forward ...`。若仍为旧版「全局单例」，第二层会错误地从第一份 GGUF 读 `blk.40` 导致缺失或 shape 错误。

### 8.3 P2.3a：`--kt-weight-path` 按层展开（与 Phase 2-B 配套）

`create_kt_config_from_server_args` 现对 `server_args.kt_weight_path` 调用 `resolve_kt_weight_path_for_layer`：

- 推荐：`.../dsv4_layer{layer_idx}.gguf`
- 与 handoff 一致：`.../dsv4_layer{}.gguf`（**恰好一个**匿名 `{}` 时替换为层号）
- 无占位符：整路径原样返回（全模型共用一份权重时）

实现：`third_party/sglang/.../kt_ep_wrapper.py`；`--kt-weight-path` 的 argparse 说明已更新。

### 8.4 P2.2：`kt_ep_wrapper` 设备无关 Stream / 同步（CUDA + NPU）

新增 `third_party/sglang/python/sglang/srt/utils/kt_accel.py`：

- `kt_new_stream` / `kt_new_event` / `kt_stream_context`：按 `tensor.device.type` 在 `torch.cuda` 与 `torch.npu` 间分发。
- `kt_current_stream` / `kt_current_stream_handle`：供 `KTMoEWrapper.submit_forward` / `sync_forward` 传入底层 stream 句柄（NPU 为 `npu_stream`）。
- `kt_device_synchronize`：替代散落的 `torch.cuda.synchronize(device)`。
- `kt_maybe_cuda_host_register`：仅 **CUDA** 上对 KT 共享内存 CPU buffer 做 `cudaHostRegister`；NPU 上为 no-op。

`kt_ep_wrapper.py` 内原 `torch.cuda.*`（含 `_prepare_weight_*` 流水线、`KTEPWrapper.apply` 双 stream）已全部改为上述 helper。

**验证**：`PYTHONPATH=.../third_party/sglang/python` 下 `from sglang.srt.layers.moe import kt_ep_wrapper` 无语法/导入错误；真机 NPU 需在 `sglang serve` 路径上再跑一轮（见 handoff P2.7）。

### 8.5 P2.3：DeepSeek-V4 与 KT 接线

- **MoE 层门控**：`create_kt_config_from_server_args` 对 **非 MoE 层**（与 KT mask 相同的 `first_k_dense_replace` / `moe_layer_freq` / `num_hash_layers` 规则，并尊重 `SGLANG_DSV4_MODE=2604`）直接返回 `None`，避免给 dense 层挂 `KTEPWrapperMethod` 却去加载 `dsv4_layer{L}.gguf`。
- **NPU 辅助流**：`DeepseekV4Model` 在 `_is_npu` 时创建 `torch.npu.Stream()` 列表（与 CUDA 下 5 条 `torch.cuda.Stream()` 对齐）；`DeepseekV2MoE.forward_normal_dual_stream` 及 SBO 里与 `alt_stream` 相关的 `wait_stream` / `stream()` 改为 `kt_accel`，以便在 NPU 上启用 `alt_streams` 时不误调 `torch.cuda.*`。

V4 的 MoE 主体仍是 `DeepseekV2MoE` → `FusedMoE`；KT 混合在 `FusedMoE` 内通过 `KTEPWrapperMethod` 完成，P2.3 补齐的是 **层索引语义** 与 **NPU 侧辅助流**。

### 8.6 P2.2 单独验证（不必等 P2.7）

- **`tools/kt_accel_stream_smoke.py`**：只测 `kt_new_stream` / `kt_stream_context` / `wait_event` / `synchronize`，**不依赖** `kt_kernel` 与权重。
- **`import kt_ep_wrapper`**：`PYTHONPATH=.../third_party/sglang/python` 下 `from sglang.srt.layers.moe import kt_ep_wrapper`，确认大文件可加载。
- **`tools/phase12_llamafile_moe_smoke.py`**：`KTMoEWrapper` + `LLAMAFILE`，覆盖 kt-kernel 与 GGUF，**不经过** `kt_ep_wrapper`。
- **一键**：`bash tools/run_p22_smoke_checks.sh`（顺序执行上述 1→2→3；无默认 GGUF 时第 3 步跳过，可用 `P12_GGUF` / `P12_LAYER` 覆盖）。
- **更完整**：到 **P2.7** `sglang serve` 才会覆盖 `KTEPWrapper.apply` + 整网调度；那是「最快端到端」验证，但不是唯一能测 P2.2 的路径。

### 8.7 后续（未在本小节落地）

整模型 `sglang serve`、`deepseek_v4` 全量接线等仍见 `doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md`。

---

## 9. Phase 2-C：单卡 NPU 端到端拉起（P2.7，已打通到 `/generate 200`）

> 截至 2026-05-13，单卡 NPU + KT(LLAMAFILE) + DeepSeek-V4-Flash W8A8 服务**端到端**已经可以起、能接 `/generate` 并返回 200 OK；目前**剩下的是数值问题**——生成内容退化成 padding/感叹号（见 `Handoff §6.11`）。本节记录从「卡死在 import / 各种 NPU op 报错」走到「跑通 wiring」过程中**所有落地代码改动**与**复现方法**。

### 9.1 关键决策：把 `third_party/sglang` 切到 `iforgetmyname/sglang@dsv4_release` baseline

**起因**：原 `kvcache-ai/sglang` fork 的 `deepseek_v4` 路径强依赖 `tilelang` / `deep_gemm`（CUDA-only），并在 RoPE / Compressor / `freqs_cis` 上用 complex64 indexing（NPU `aclnnIndex` 不支持）。我们一开始走「graceful degradation」（每个不兼容点加 try/except 走 torch fallback），但补丁面越来越大，与「除 MoE offload 外其它算子与基线完全同步」的目标背离。

**操作**（已在仓库历史里完成；如复现需重做）：

```bash
cd /workspace/code/ktransformer/ktransformers-AK
git submodule deinit -f third_party/sglang
mv third_party/sglang third_party/sglang.kvcache-ai-archive  # 旧 fork 留作参考
# 更新 .gitmodules
#   url    = https://github.com/iforgetmyname/sglang.git
#   branch = dsv4_release
git submodule update --init --remote third_party/sglang
# 或直接：将 /sgl-workspace/sglang（已是该分支头）拷过去
rsync -a --delete --exclude='.git' /sgl-workspace/sglang/ third_party/sglang/
```

切完之后，**baseline 自带 KT EP / NPU 算子支持**，无需在 `deepseek_v4.py` / `kt_ep_wrapper.py` / RoPE / Compressor / HC pre/post 之类的地方再大改。只剩两类必补：
1. `kt_accel.py`（CUDA↔NPU stream/event 抽象，baseline 没有这个 helper）。
2. `kt_ep_wrapper.py` 内的少量 NPU-friendly 改动（详见 §9.3）。

### 9.2 新增/修改文件清单（截至 2026-05-13）

| 路径 | 改动类型 | 摘要 |
|------|---------|------|
| `.gitmodules` | 修改 | `third_party/sglang` 指向 `iforgetmyname/sglang@dsv4_release` |
| `third_party/sglang.kvcache-ai-archive/` | 归档 | 原 `kvcache-ai/sglang` fork，**只读参考**，与运行路径无关 |
| `third_party/sglang/python/sglang/srt/utils/kt_accel.py` | **新建（从 archive backport）** | CUDA↔NPU stream/event/同步抽象（与 §8.4 同一个 helper） |
| `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py` | 修改 | （a）`KTMoEWrapper` 构造参数适配本机 `kt_kernel 2026.x` wheel：传 `gpu_experts_mask: BoolTensor` 与 `numa_nodes=None`，而非 `num_gpu_experts: int`；（b）`load_weights/submit/sync` 内 stream 调用全部换成 `kt_accel.kt_current_stream_handle / kt_device_synchronize`；（c）权重路径走 `resolve_kt_weight_path_for_layer` 以支持 `{layer_idx}` / 单个 `{}` 占位符；（d）`mask_cpu_expert_ids` 重写为 `mask_cpu_expert_routing`：用 `torch.where` 把 CPU expert id 与 weight 同时改写为 `(0, 0.0)`，规避 NPU `npu_moe_init_routing/compute_expert_tokens` 对负 id 不支持，以及 `torchair` 对 `aten.index_put.default` 的不支持 |
| `third_party/sglang/python/sglang/srt/hardware_backend/npu/allocator_npu.py` | 修改 | import 期一次性探测 `triton.runtime.driver.driver.active`；探测失败（NPU 现状）就让 `alloc_extend` 跳过 `< 200 pages` 的 `@triton.jit` 快路径，统一走同文件已存在的 `alloc_extend_naive`（纯 torch，分配 KV 索引，与 forward 数值无关） |
| `third_party/sglang/python/sglang/srt/mem_cache/common.py` | 修改 | 同样的 driver 探测；新增 `_write_req_to_token_pool_torch` / `_write_req_to_token_pool_only_alloc_size_torch` 两个 torch 等价实现；`write_multi_cache_indices`（baseline 漏判 `support_triton('ascend')`、6 处无脑调 triton）、`write_cache_indices`、`get_last_loc` 三处统一在 driver 不可用时落 torch 分支 |
| `tools/p27_launch_ds4flash_npu.sh` | 修改 | （a）`PYTHON_BIN` 探测：依次试 `python3` / `python3.11` / `/usr/local/python3.11.14/bin/python3.11` 等，找第一个能 `import numpy/torch/torch_npu/sglang` 的；脚本里所有 `python3 -c` / `-m sglang.launch_server` 改成 `${PYTHON_BIN}`，避免 `source set_env.sh` 后 PATH 被改、`python3` 跑到 `/usr/bin/python3`（没装 numpy）。（b）参数与基线 8 卡 `launch_ds4flash_sglang.sh` 取「单卡子集」对齐：补 `--cuda-graph-bs 1` / `--disable-radix-cache` / `--max-prefill-tokens 65535` / `--context-length 65536` / `--watchdog-timeout 18000` / `--skip-server-warmup`。（c）引入 `EXTRA_FLAGS` 变量便于临时叠 `--disable-cuda-graph` 等调试 flag |
| `tools/p27_curl_generate.sh` | 沿用 | 对已起服务发一发 `/generate`，无改动 |

> **绝大多数原 fork 的 fallback 补丁（`tilelang` / `deep_gemm` / `complex64` / `_real_freqs_for_npu` / `_fused_rope_torch_fallback` 等）随 baseline 切换被自动撤回**，仅保留 KT/NPU 同步与 Triton 兜底两小块下游 patch。详细背景见 `Handoff §6.10`。

### 9.3 `kt_ep_wrapper.py` 三处下游 patch（细节）

下面三段是相对 baseline 的 minimal diff，**与上游 `deepseek_v4` 路径无关**，将来若 baseline 自身吃掉这些，直接撤回即可。

```python
# (1) per-layer 模板：与 Phase 2-B 的 P2.3a 同源
def resolve_kt_weight_path_for_layer(template: str, layer_idx: int) -> str:
    if "{layer_idx}" in template:
        return template.format(layer_idx=layer_idx)
    if template.count("{}") == 1:
        return template.format(layer_idx)
    return template

# (2) 适配本机 kt_kernel wheel：构造 BoolTensor mask + numa_nodes=None
gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool, device="cpu")
gpu_experts_mask[:num_gpu_experts] = True
wrapper = KTMoEWrapper(
    ...,
    gpu_experts_mask=gpu_experts_mask,  # 不是 num_gpu_experts: int
    numa_nodes=None,
    weight_path=resolve_kt_weight_path_for_layer(template, layer_idx),
)

# (3) NPU 友好的 CPU expert 屏蔽（替换 mask_cpu_expert_ids）
@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_routing(topk_ids, topk_weights, num_gpu_experts):
    """把 routing 中归属 CPU expert 的行同时改写为 (id=0, weight=0.0)。
    - NPU 的 npu_moe_init_routing/compute_expert_tokens 不接受负 id（曾用 -1，触发 aclnnMatmul aicore exception）。
    - torchair ge_converter 不支持 aten.index_put.default（原 `topk_ids[mask] = -1` 写法）。
    - weight=0 保证 GPU 这一路对 CPU expert 的贡献为 0；CPU 那一路由 KTMoEWrapper 独立累加。"""
    is_gpu = topk_ids < num_gpu_experts
    safe_ids = torch.where(is_gpu, topk_ids, torch.zeros_like(topk_ids))
    safe_weights = torch.where(is_gpu, topk_weights, torch.zeros_like(topk_weights))
    return safe_ids, safe_weights
```

### 9.4 Triton-on-NPU 现状与「全局兜底」逻辑

镜像内 `triton 3.7.0` 与 `triton-ascend 3.2.0` 是**版本错配**：

| 包 | 表现 |
|---|---|
| `triton 3.7.0` | upstream 已删除 `triton.backends.compiler.AttrsDescriptor` |
| `triton-ascend 3.2.0` | `triton/backends/ascend/compiler.py` 仍 `from triton.backends.compiler import AttrsDescriptor` —— import 立崩 |
| `triton.backends` 注册结果 | 仅 `amd` / `nvidia`，两者 `is_active()=False` |
| `import triton` / `from triton.runtime.driver import driver` | **不报错**（driver 是 `DriverConfig` 懒代理） |
| `driver.active` 第一次被访问 | `RuntimeError: 0 active drivers ([])` |
| 任何 `@triton.jit` kernel 调用 | 内部第一步就是 `driver.active.get_current_device()` → 同样报错 |

**所以这台机器上 Triton-on-NPU 是不可用的**。Baseline 自己也带这个隐患（`mem_cache/common.py` 里 6 处 triton 调用没有读 `support_triton('ascend')==False`），但 8 卡基线机器装的是匹配的 wheel，所以掩盖了。

我们的做法：在 **`allocator_npu.py`** 与 **`mem_cache/common.py`** import 期各做一次 driver 探测，缓存到 `_TRITON_DRIVER_AVAILABLE`，所有走 triton 的分支前面加一个判断；不可用时落到同文件已有的 torch 等价路径（`alloc_extend_naive` / `_write_req_to_token_pool_torch` / `get_last_loc_torch`）。这些路径**纯做整数下标算术**，对 forward 数值零影响。

降级开关：`SGLANG_NPU_ALLOC_FORCE_NAIVE=1` 显式强制 fallback（已 work 的状态）；将来 `triton-ascend` 修好（或者升到匹配 wheel）后探测会自动转 True，无需再改代码。

### 9.5 如何复现 P2.7 端到端

```bash
# 1) 确认 PYTHONPATH 与 PYTHON_BIN：脚本会自检，但建议事先在 shell 里也指向
export REPO=/workspace/code/ktransformer/ktransformers-AK
export PYTHONPATH="$REPO/third_party/sglang/python${PYTHONPATH:+:$PYTHONPATH}"

# 2) 启动单卡服务（脚本内部自动 source CANN set_env、自动锁 PYTHON_BIN）
ASCEND_RT_VISIBLE_DEVICES=1 bash $REPO/tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/p27.log
# 大约 7–8 分钟（KT 加载 43 层 + npu graph capture）后会看到：
#   [allocator_npu] Triton driver unavailable (RuntimeError: 0 active drivers ([])); falling back to alloc_extend_naive ...
#   ...
#   [2026-05-13 11:55:22] The server is fired up and ready to roll!

# 3) 另开 shell 发请求（HOST/PORT 与脚本一致）
bash $REPO/tools/p27_curl_generate.sh
```

**预期**：HTTP 200，`finish_reason.length=32`，`e2e_latency` 约 2.3s。**但 `text` 内容当前是退化的 padding/感叹号**（详见 §9.6）。

### 9.6 已知遗留：生成内容是 "  !  !  !  !  ..."

```
text:    "  !  !  !  !  ! ! ! ! ! ! ! ! ! "
output_ids: [223, 223, 3, 223, 223, 3, ...]   # 全是空格(223) 与 "!"(3)
```

服务从 import → load_weights → npu graph capture → 接 prefill → 完成 decode 全链都跑通，**数值环节出错**。这与 Triton fallback 无关（fallback 只动 KV 索引），属于下一阶段工作；具体诊断顺序与对照实验见 `Handoff §6.11`。

### 9.7 P2.7 验收（更新）

- [x] `sglang serve` 起来后接受 HTTP 请求，返回 200 OK
- [x] HBM 占用观测：`max_total_num_tokens=4276224, avail mem=7.92 GB`（mem_fraction_static=0.85 下 NPU 剩 ~8 GB）
- [x] 服务在「卡死 import / aclnnMatmul aicore exception / 0 active drivers」三类典型 NPU 错误后都已能稳定启动
- [ ] **生成内容语义有效**（与 8 卡基线同 prompt 同 seed 输出"接近"，容许 Q8_0 量化误差）—— **未达成，下一阶段**
- [ ] tokens/sec baseline 数据采集（先把数值修对再录）

---

**文档维护**：Phase 2 其它子项完成后，可继续在本文 §8/§9 下追加小节。
