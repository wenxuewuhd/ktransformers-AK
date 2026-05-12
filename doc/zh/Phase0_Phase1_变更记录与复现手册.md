# Phase 0 / Phase 1 变更记录与复现手册

> **目的**：在动手 Phase 2（多 GGUF loader 等）之前，把截至 **Phase 1.2 完成** 为止的**所有工程向改动**、**系统依赖**与**逐步验证方法**固化到一处，便于新环境/新同事 **按步骤复现**。  
> **说明**：本文以**源码与脚本**为主；**权重文件**（HF W8A8、生成的 GGUF）体积巨大，需自备路径，不随仓库分发。

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

1. **系统**：`numactl --hardware`、`npu-smi info`（可选）、`ldconfig -p | grep hwloc`。
2. **Phase 0**：`./install.sh build` 或 CMake 块 → Python `import kt_kernel_ext` → `ldd`。
3. **Phase 1.1**：单层 `convert_...` + `verify-reader`；再批量 `batch_...` + `verify-sample`。
4. **Phase 1.2**：`pip install -e kt-kernel --no-deps`（torch 已对齐前提下）→ `phase12_...py`。
5. **Phase 2-B**（多 GGUF 同进程）：见 §8，`phase12_...py` 带 `--second-gguf` / `--second-layer-idx`。
6. **NPU 回归**（可选）：`import torch; import torch_npu`、原 8 卡启动脚本抽测。

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
| **整网多 layer（旧行为）** | 曾用**全局唯一** `GGUFLoader`，第二个 `weight_path` 被忽略 | **已修**：见 §8，按 `os.path.realpath(weight_path)` **每路径一个 loader** |

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

### 8.4 后续（未在本小节落地）

整模型 `sglang serve`、NPU 侧 `kt_ep_wrapper` 等与总方案仍见 `doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md`。

---

**文档维护**：Phase 2 其它子项完成后，可继续在本文 §8 下追加小节。
