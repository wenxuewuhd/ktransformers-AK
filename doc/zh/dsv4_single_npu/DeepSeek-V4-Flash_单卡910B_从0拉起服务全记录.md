# DeepSeek-V4-Flash 单卡 910B 从 0 拉起服务 —— 全过程实录

> 本文档记录在**本容器/本机**上,从一个「patch 已打好、但未编译、未转权重」的工作树出发,
> 一步步把 DeepSeek-V4-Flash(W8A8)单卡 Ascend 910B + KT CPU MoE 推理服务拉起来的**全部实际操作**,
> 包括踩到的坑与修复。命令均为可直接复制执行的真实命令。
>
> - 仓库根:`/workspace/code/ktransformers-AK`
> - 模型(HF W8A8):`/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8`
> - GGUF 输出目录:`/workspace/models/cache`
> - Python:`/usr/local/python3.11.14/bin/python3.11`
> - 目标卡:910B card 0
>
> 说明:`tools/kt_dsv4_npu_patches/readme.MD` 是「从开源裸仓用 patch 复现」的指南。本工作树 patch **已打好**,
> 因此本文档**跳过打 patch**,只覆盖「编译 → 转权重 → 拉起 → 使用」。
>
> **⚡ 2026-06-09 更新(本文之后的进展,以总纲为准)**:① graph 路径已闭合(取代本文的 eager 回退);
> ② graph decode 已提速 **3.6 → 6.12 tok/s(~1.7×)**——`tools/p27_launch_ds4flash_npu.sh` 的
> `--kt-cpuinfer` 默认 **24→96**(CPU MoE 是内存带宽瓶颈,旧默认只用了 24/192 核;`KT_CPUINFER` 可覆盖,
> **勿 ≥128**)。纯配置改动 commit `68f8556`,精度无损。详见
> [DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md](DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md) §6.6
> + [graph_decode_profiling_report.md](graph_decode_profiling_report.md)。下面正文是「从 0 拉起」的历史实录,
> 拉起命令仍有效(脚本默认值已更新)。

---

## 0. 开工前环境体检

| 项 | 状态 | 备注 |
|---|---|---|
| sglang / llama.cpp 子模块 | ✅ 已就绪 | `third_party/sglang/python/sglang`、`third_party/llama.cpp/gguf-py` 均在 |
| 910B 单卡 | ✅ card 0 空闲 | `npu-smi info` 显示 0 号卡 HBM 几乎全空(65536MB) |
| kt-kernel C++ 扩展 `kt_kernel_ext*.so` | ❌ 未编译 | `kt-kernel/python/` 下无 `.so` |
| GGUF 权重 `dsv4_layer*.gguf` | ❌ 0 个 | `/workspace/models/cache` 还不存在 |
| 模型路径 | ⚠️ 非默认 | 在 `…/DeepSeekV4/…`,启动时须 `MODEL_PATH` 覆盖 |
| 磁盘 | ✅ | `/workspace/models` 余 ~20T,放 Q8_0 ~295 GiB 足够 |

模型规格(`config.json`):43 层全 MoE(`first_k_dense_replace=0`),`n_routed_experts=256`,`num_experts_per_tok=6`。

体检命令:

```bash
# 子模块
ls third_party/sglang/python/sglang/__init__.py
ls third_party/llama.cpp/gguf-py/gguf/__init__.py
# NPU
npu-smi info | head -20
# 是否已编译
find kt-kernel -name "kt_kernel_ext*.so"
# 是否已转权重
ls /workspace/models/cache/dsv4_layer*.gguf 2>/dev/null | wc -l
# 模型规格
grep -i "num_hidden_layers\|first_k_dense\|n_routed_experts\|num_experts_per_tok" \
  /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8/config.json
```

---

## 1. 编译 kt-kernel(带 Ascend NPU 后端)

### 1.1 命令

```bash
cd /workspace/code/ktransformers-AK/kt-kernel
CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
```

### 1.2 坑 ①:hwloc 缺失导致 CMake configure 失败

**报错**:

```
-- Checking for one of the modules 'hwloc'
CMake Error at /usr/share/cmake-3.22/Modules/FindPkgConfig.cmake:890 (message):
  None of the required 'hwloc' found
Call Stack (most recent call first):
  CMakeLists.txt:612 (pkg_search_module)
```

**原因**:`kt-kernel/CMakeLists.txt:612` 把 hwloc 设为 `pkg_search_module(HWLOC REQUIRED ...)`,
而系统未安装 hwloc(库/头文件/`hwloc.pc` 全无)。

**修复**:安装开发包(含运行期 `libhwloc.so.15`)。

```bash
apt-get install -y libhwloc-dev
# 验证
pkg-config --modversion hwloc   # → 2.7.0
```

> 安装版本:`libhwloc-dev / libhwloc15 = 2.7.0-2ubuntu1`(arm64)。
> 容器重启后 hwloc 可能丢失,需重装;运行期 `import kt_kernel` 也依赖 `libhwloc.so.15`。

### 1.3 坑 ②:llama.cpp 子模块版本不对 → llamafile 编译找不到头文件

装完 hwloc 重新编译,llamafile 阶段报错:

```
third_party/llamafile/iqk_mul_mat_arm.inc:27:10: fatal error:
  llama.cpp/ggml-impl.h: No such file or directory
   27 | #include "llama.cpp/ggml-impl.h"
gmake[1]: *** [CMakeFiles/Makefile2:189: CMakeFiles/llamafile.dir/all] Error 2
```

**原因**:`git status` 一开始就显示 `M third_party/llama.cpp`、`M third_party/sglang`
(子模块工作树指针被改过,是**未提交的 WIP bump**,且比仓库记录的还新)。

- llama.cpp:当前 HEAD `94a220cd6`(很新的 llama.cpp,ggml 头文件已搬到 `ggml/src/`),
  仓库记录应为 `ac315ccc`。
- sglang:当前 HEAD `c9edb75e0`,仓库记录应为 `68a0bce65`。

vendored 的 `third_party/llamafile` 仍按**老布局**写 `#include "llama.cpp/ggml-impl.h"`;
kt-kernel CMake 把 `third_party/` 加进 include 路径(`kt-kernel/CMakeLists.txt:421`
`include_directories(.../third_party)`),所以该 include 解析成
`third_party/llama.cpp/ggml-impl.h` —— 这在 **b3173 时代头在根目录**才成立,新版本已不在根目录,故失败。

**修复**:把 llama.cpp 切回公开基线 **tag b3173 = `a94e6ff`**。
(readme §7 里记录的 `ac315ccc` = b3173 + 一个**纯 Python**(gguf-py NumPy2 读取)补丁,
对 C++ 编译无影响,故编译只需 b3173 本体。)

```bash
cd /workspace/code/ktransformers-AK/third_party/llama.cpp
# 该子模块 remote 已是公开 GitHub(ggerganov/llama.cpp);内网 fork 的 ac315ccc 在 github 上没有,
# 但公开 tag b3173 有:
git fetch --depth 1 origin tag b3173
git checkout b3173                  # → a94e6ff8774b...
ls ggml-impl.h ggml-quants.h ggml-common.h   # 应都在根目录
```

> sglang 子模块(`c9edb75e0`)是 launch 期才用的 Python 依赖,**编译用不到**,
> 暂不改动,留用户的 WIP;拉起服务时再验证其可 import 且含 KT EP wrapper。

### 1.4 清缓存 + 重新编译

切换 llama.cpp 版本后,旧 `build/` 里的 CMakeCache 仍指向新版子模块,必须清掉:

```bash
cd /workspace/code/ktransformers-AK/kt-kernel
rm -rf build
CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
```

产物:`kt-kernel/python/kt_kernel_ext*.so`。

### 1.5 编译结果

```
kt-kernel/python/kt_kernel_ext.cpython-311-aarch64-linux-gnu.so   (1.8M)
```

---

## 2. W8A8 → 43 层 GGUF(Q8_0)

转换**不依赖 kt-kernel 编译**(只用 gguf-py + torch + safetensors),可与编译并行跑。

### 2.1 基础命令

```bash
mkdir -p /workspace/models/cache
cd /workspace/code/ktransformers-AK
/usr/local/python3.11.14/bin/python3.11 tools/batch_convert_w8a8_layers_mp.py \
  --input  /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 \
  --quant q8_0 --jobs <N> --verify-sample 3
```
输出 `dsv4_layer0.gguf … dsv4_layer42.gguf`,每层 Q8_0 ~6.85 GiB,合计 ~295 GiB。

### 2.2 多线程/多进程加速调优(本机 192 核 / 1.5TB RAM)

`batch_convert_*` 用 `ProcessPoolExecutor`,`--jobs` = 同时转换的层数(每层一个子进程,
子进程内 numpy/Q8_0 再自动多线程)。**单进程的 numpy 工作自然只扩到 ~10 线程**,
所以提速主要靠**加 `--jobs`(进程数)**,而非单进程线程。

实测(本机)聚合 CPU 占用:

| `--jobs` | 聚合 CPU | 内存 | 备注 |
|---|---|---|---|
| 4(脚本默认) | ~40 / 192 核 | ~18 GB | 严重闲置 |
| 16 | ~78 / 192 核 | ~68 GB | 仍有大量余量 |
| **32(选用)** | **~129 / 192 核** | ~121 GB | load≈220,磁盘 I/O 开始成为另一瓶颈;较优 |

> - 内存极充裕(1.4TB),不是限制;按每进程 ~4.5GB 估,jobs=32 约 144GB。
> - load avg 远高于聚合 CPU(220 vs 129)说明部分时间在等磁盘(读 safetensors / 写 6.85GiB GGUF),
>   再往上加 `--jobs` 收益递减,且 43 层 / 32 只剩 11 层尾波。故选 **jobs=32**。
> - 若机器更小,按「核数 / 10」估 `--jobs` 起步,内存够就往上加到接近核数饱和。

本次实际用:

```bash
nohup /usr/local/python3.11.14/bin/python3.11 tools/batch_convert_w8a8_layers_mp.py \
  --input  /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --quant q8_0 --jobs 32 --verify-sample 3 \
  > /tmp/kt_convert.log 2>&1 &
```

---

## 3. 运行期 import 自检(拉起前先做,省得几小时后才报错)

```bash
cd /workspace/code/ktransformers-AK
[ -e kt-kernel/kt_kernel ] || ln -sfn python kt-kernel/kt_kernel    # ensure_kt_kernel 的接线
export PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel"
PY=/usr/local/python3.11.14/bin/python3.11
$PY -c "import kt_kernel, kt_kernel.kt_kernel_ext as e; print('kt_kernel OK')"
$PY -c "import torch, torch_npu; print('torch_npu OK')"
$PY -c "import importlib.util as u; print('kt_ep_wrapper:', u.find_spec('sglang.srt.layers.moe.kt_ep_wrapper') is not None)"
```

`torch_npu`、`sglang`(含 `kt_ep_wrapper`)一次通过。但 `import kt_kernel` 触发了坑 ③。

### 坑 ③:`undefined symbol: iqk_mul_mat_moe_arm82`(aarch64 符号 rename 缺失)

```
ImportError: Failed to load kt_kernel extension (variant: avx2).
Original error: .../kt_kernel_ext.cpython-311-aarch64-linux-gnu.so:
  undefined symbol: iqk_mul_mat_moe_arm82
```

> 报错里的 `variant: avx2` 是 `_cpu_detect.py` 在 aarch64 上的**兜底标签**(该检测逻辑面向 x86,
> aarch64 走 `return "avx2"` 兜底),非真正问题;真正问题是 arm82 符号没定义。

**定位**:`nm -D .so | grep arm82` 显示 `U iqk_mul_mat_arm82`、`U iqk_mul_mat_moe_arm82`(都未定义)。

`third_party/llamafile/` 的 arm82 路径靠「**包含 `.inc` + `#define` 把无后缀符号 rename 成带后缀**」工作:
- `tinyblas_cpu_sgemm_arm82.cpp`:`#define llamafile_sgemm llamafile_sgemm_arm82` ✅ 生效
- `tinyblas_cpu_mixmul_arm82.cpp`:`#define llamafile_mixmul llamafile_mixmul_arm82` ✅ 生效
- `iqk_mul_mat_arm82.cpp`:两行 rename **被注释掉了** ❌ —— 它是唯一 `#include "iqk_mul_mat_arm.inc"`
  (该 inc 在 187/209 行定义 `iqk_mul_mat`/`iqk_mul_mat_moe`)的 TU,注释后只吐无后缀符号,
  而 `sgemm.cpp` 的 aarch64 dispatch 只引用带后缀的 `iqk_mul_mat_moe_arm82`,故 undefined。
  文件自己的注释甚至写明了这个报错。

**修复**:取消 `third_party/llamafile/iqk_mul_mat_arm82.cpp` 里两行注释:

```cpp
#ifdef __aarch64__
#define iqk_mul_mat iqk_mul_mat_arm82
#define iqk_mul_mat_moe iqk_mul_mat_moe_arm82
#include "iqk_mul_mat_arm.inc"
#endif
```

然后**增量重编译**(只重编这个 .cpp + 重链接):

```bash
cd /workspace/code/ktransformers-AK/kt-kernel
CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
# 验证:nm -D python/kt_kernel_ext*.so | grep arm82  → 应变成 T(已定义)
```

> 这是工作树里一处**未完成的 WIP 编辑**(注释写好了、修复没启用)。
> 与 hwloc(坑①)、llama.cpp 子模块版本(坑②)一样,都是「从 0 复现」必须先趟平的。

重编译后 `import kt_kernel` 通过;`nm -D -u .so | grep arm82` 显示无 undefined。

### 坑 ④:转换收尾 `--verify-sample` 报 NumPy 2.0 `newbyteorder`

43 层**写入全部成功**,但 batch 末尾用 `GGUFReader` 抽样校验时崩:

```
AttributeError: `newbyteorder` was removed from the ndarray class in NumPy 2.0.
  Use `arr.view(arr.dtype.newbyteorder(order))` instead.   (gguf_reader.py:141)
```

这就是 readme §7 那个「默认不打、读 GGUF 报错时再打」的可选补丁。**只影响读取/校验,不影响已写出的权重**。

**修复**:应用仓库自带补丁:

```bash
cd /workspace/code/ktransformers-AK/third_party/llama.cpp
git apply ../../tools/kt_dsv4_npu_patches/llama_cpp/0001-fix-gguf-NumPy-2-GGUFReader.patch
# 验证可读:
PYTHONPATH=$PWD/gguf-py /usr/local/python3.11.14/bin/python3.11 \
  -c "import gguf; r=gguf.GGUFReader('/workspace/models/cache/dsv4_layer0.gguf'); print('tensors', len(r.tensors))"
```

> 切到 b3173(坑②)后子模块就是「未打此补丁」的纯基线,所以必然命中;打上即为 readme 记录的 `ac315ccc`。

### 2.3 转换完成

- 43/43 层,`/workspace/models/cache/dsv4_layer{0..42}.gguf`,每层 ≥6.4 GiB,合计 ~275 GiB。
- 预检:`bash tools/p27_e2e_preflight.sh` → **PASS**(43 文件齐 + kt_kernel_ext 路径对)。

---

## 4. 拉起服务(单卡 910B)

### 4.1 命令

> **前提:`third_party/sglang` 必须在打好补丁的 `dsv4_release@a347a9ad5`(见坑⑤)。**
> 切对后启动命令很干净,不需要任何 config backup / 量化绕过。

```bash
cd /workspace/code/ktransformers-AK
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=0 \
bash tools/p27_launch_ds4flash_npu.sh
```

- `MODEL_PATH`:模型不在脚本默认路径,必须覆盖。
- `NPU_DEVICE_ID=0`:绑 910B 0 号卡。

### 坑 ⑤:`Quantization method (fp8) does not match (compressed-tensors)`

首次拉起在解析参数阶段就崩:

```
SGLANG_APPLY_CONFIG_BACKUP=auto: checkpoint has num_hidden_layers=43, dispatching to 'small'.
SGLANG_APPLY_CONFIG_BACKUP=small: using packaged config_backup_small.json instead of the checkpoint's config.json ...
ValueError: Quantization method specified in the model config (fp8) does not match
  the quantization method specified in the `quantization` argument (compressed-tensors).
```

**真正根因(关键)**:`third_party/sglang` 子模块**切到了错误的 fork/commit** ——
remote 指向 `github.com/kvcache-ai/sglang`(上游),HEAD 在 `c9edb75e0`
(“Fix/v4flash gpu prefill fallback mxfp4 #41”),**根本没打 KT 的 5 个补丁**,
也没切到 `dsv4_release` 分支。这版上游引入了严格 `DeepSeekV4Config`(`@dataclass`,
不吃 `**kwargs`)+ `SGLANG_APPLY_CONFIG_BACKUP` 机制(默认 `auto`,用打包的 fp8
`config_backup_small.json` 替换真实 config),与本 W8A8 模型完全不匹配。

> 走过的弯路(留作教训):一开始误以为只是 backup 机制干扰,试过
> `SGLANG_APPLY_CONFIG_BACKUP=none` 改读真实 config,结果又撞 `DeepSeekV4Config.__init__()
> got an unexpected keyword argument 'n_activated_experts'`(严格 dataclass 拒收真实 config
> 里的原生命名字段),且该类连模型要读的 `head_dim`/`swiglu_limit` 都没声明 —— 说明这版上游
> 的 V4 路径本身就不完整。**这些都是“错 commit”的症状,不是真问题。**

**正确做法:把 sglang 子模块切回打好补丁的 `dsv4_release` 分支**。本机已有正确代码副本
`/workspace/code/tmp/sglang`(分支 `dsv4_release`,HEAD `a347a9ad5`,即 base `298193eb3`
+ readme §6 的 5 个 KT patch:Triton 回退 / KT EP wrapper / graph host callback /
decode profiler / hot expert placement)。

```bash
cd /workspace/code/ktransformers-AK/third_party/sglang
git remote -v          # 错误状态:origin=github.com/kvcache-ai/sglang,HEAD=c9edb75e0(无 KT 补丁)
# 从本地正确副本拉取并切分支(目标 commit 内网 remote 拉不到,用本地 tmp 最稳):
git fetch /workspace/code/tmp/sglang dsv4_release
git checkout -B dsv4_release a347a9ad585207455bb7a2c14d94dcfdfda2a918
git rev-parse HEAD     # → a347a9ad5...
```

切换后核对(正确版**不含** backup 机制、kt_ep_wrapper 在):

```bash
cd /workspace/code/ktransformers-AK
grep -rl SGLANG_APPLY_CONFIG_BACKUP third_party/sglang/python/  # 应为空
export PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel"
/usr/local/python3.11.14/bin/python3.11 -c "import importlib.util as u; \
  print('kt_ep_wrapper:', u.find_spec('sglang.srt.layers.moe.kt_ep_wrapper') is not None)"
```

> 切对 sglang 后,**4.1 的启动命令不需要任何 `SGLANG_APPLY_CONFIG_BACKUP`**,
> 直接 `MODEL_PATH=... NPU_DEVICE_ID=0 bash tools/p27_launch_ds4flash_npu.sh` 即可。
>
> 操作教训:**不要用 `pkill -f "sglang.launch_server"` 杀进程** —— 你的命令行字符串本身含该模式,
> pkill 会把执行命令的 shell 一起杀掉(表现为 exit 1、无输出、命令像没运行)。按 PID 杀,或用
> 不匹配自身的更具体模式。

### 坑 ⑥:NPU graph 捕获崩 `aclrtMemcpy 107030`(capture 不允许同步 memcpy)

> **✅ 已修复(2026-06-08)** —— graph 路径已闭合,现在**默认 graph-on 即可端到端跑通**,decode
> `npu graph: True` ~3.5–3.9 tok/s(取代下文 eager ~1.6 回退)。真因是**两层**(都不在当时的嫌疑栈里):
> ① `kt_ep_wrapper.py::mask_cpu_expert_routing` 里 `gpu_experts_mask.to(device)` 在 capture 期做同步 H2D;
> ② 该函数被 `@torch.compile`→NPU torchair 编成绑定 stream 的独立子图,与外层图跨 stream 冲突
> (`Unsupport run graph with different stream`)。详细根因/改动/实测见
> [Plan-and-Progress §6.3](DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md)。**下文保留当时的崩溃实录与
> eager 回退**作历史;最新「可用启动命令」见本文 §4.2(已改 graph-on)。

切对 sglang 后,模型**完整加载成功**(~9 min,46 个 shard + 43 层 GGUF),但在最后的
NPU graph 捕获阶段崩:

> 注(后续):此处 ~9 min 为加载加速前的耗时。P0+P1(zero-copy + 并行重排)后,43 层 MoE
> GGUF 加载从 ~7.9 min 降到 ~47s,整体加载段降到 ~100s。详见
> `dsv4_single_npu/DeepSeek-V4-Flash_CPU权重加载加速_P0-P1.md`。

```
init_device_graphs → npu_graph_runner → cuda_graph_runner.py:680
Exception: Capture cuda graph failed: aclrtMemcpy, error code is 107030
EE9999: Not allow to synchronize captured-stream, stream_id=42.
  rtMemcpy ... the current capture mode does not support this operation
```

即 graph capture 期间发生了 host↔device 同步 memcpy(hybrid CPU MoE 与 NPU graph 的交互路径)。
launch 脚本默认 graph on(走 kt-kernel ACL callback worker + kt_ep_wrapper host callback),
此处该路径里仍有同步拷贝触发了 capture 限制。

**当前处置:先用 eager 模式(关 graph)把服务跑通**(launch 脚本自带回退):

```bash
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=0 \
EXTRA_FLAGS="--disable-cuda-graph" \
bash tools/p27_launch_ds4flash_npu.sh
```

> graph on 是生产/性能路径(基线 ~3.6–3.7 tok/s),eager 会慢不少但功能完整。
> graph 捕获里的同步 memcpy 来源待进一步定位(KT submit/sync 路径 or 某个 NPU 算子),
> 属于后续优化项,不阻塞「服务可用」。

### 坑 ⑦(真正的乱码根因):CPU MoE async submit 未 flush → 输出全零,需 `KT_FORCE_SYNC_SUBMIT=1`

eager 服务能启动、能出 token,但**输出乱码**(“我，我，我…”退化重复)。用仓库自带离线对账工具
`tools/p27_cpu_moe_reference_check.py`(把 KTMoEWrapper 输出 vs 纯 PyTorch dequant 参考比 cosine)定位:

```bash
export PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel"
# 默认(async submit):
python3 tools/p27_cpu_moe_reference_check.py --w8a8 <模型> --method LLAMAFILE \
  --gguf /workspace/models/cache/dsv4_layer3.gguf --layer-idx 3
#   → RESULT FAIL,cosine=0.0,cand 全零(nonzero=0/4096)

# 加 KT_FORCE_SYNC_SUBMIT=1:
KT_FORCE_SYNC_SUBMIT=1 python3 tools/p27_cpu_moe_reference_check.py ... --gguf dsv4_layer3.gguf --layer-idx 3
#   → RESULT PASS,cosine=0.999887,max_rel_err=1.63%(Q8_0 量化预期误差)
# BF16 同理:cosine=0.999997
```

**结论**:
- **CPU MoE 内核本身完全正确**,Q8_0(int8)和 BF16 离线对账都 cosine≈1.0。
- 乱码真因:CPU MoE 的 **async submit→sync→merge 在当前路径没 flush**,输出 buffer 留零 →
  MoE 结果缺失 → logits 退化 → 重复乱码。`KT_FORCE_SYNC_SUBMIT=1` 强制同步即修复。
- **Handoff 文档关于「aarch64 Q8_0 NaN」「MOE_INT8/KML 在 K920 不可用」的结论已过时**;
  实测 int8(Q8_0)CPU offload 跑得通,现有 275 GiB Q8_0 GGUF 可用,**无需转 BF16(555 GiB)**。

这也解释了之前两次失败的同一根源(KT 的同步拷贝):
- **graph on**:crash 在 capture 期的 `aclrtMemcpy`(坑⑥)—— 正是 KT submit 的 sync copy 撞 capture 限制;
- **eager 默认**:async 没 flush → 全零乱码(本坑)。

### 4.2 最终可用启动命令(graph-on,生产性能路径)

> 坑⑥/⑥b 修复后(2026-06-08),**默认 graph-on 即可端到端跑通**,无需任何 `KT_FORCE_SYNC_SUBMIT`
> / `--disable-cuda-graph`。先 `npu-smi info` 选空闲卡。

```bash
cd /workspace/code/ktransformers-AK
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=<空闲卡> \
bash tools/p27_launch_ds4flash_npu.sh
# 等加载(真实权重 GGUF 读取较慢)→ Capture npu graph end → The server is fired up
```

**eager 回退(仅对照/排障用)**:

```bash
MODEL_PATH=… NPU_DEVICE_ID=<空闲卡> \
KT_FORCE_SYNC_SUBMIT=1 EXTRA_FLAGS="--disable-cuda-graph" \
bash tools/p27_launch_ds4flash_npu.sh
```

**dbg 期绕过 CPU MoE 慢加载(`KT_DUMMY_CPU_WEIGHTS`)**:调 graph/capture 时反复重启,真实权重
GGUF 读取是主要时间开销。加 `KT_DUMMY_CPU_WEIGHTS=1` 会**跳过磁盘读取**、按张量元数据 fabricate
同字节布局的零 buffer(C++ MOE/load_weights_task 路径不变,capture 与 forward 忠实执行),拉起快很多。

```bash
KT_DUMMY_CPU_WEIGHTS=1 NPU_DEVICE_ID=<空闲卡> bash tools/p27_launch_ds4flash_npu.sh
```
> ⚠️ dummy 权重输出**无意义**,仅用于「capture / 图重放能否跑通」这类结构性调试,**严禁用于精度验收**。
> 验收必须去掉该开关,用真实权重 + `tools/p27_curl_f2_prompts.sh` 看连贯 + `p27_cpu_moe_reference_check.py` 对账。
> 实现见 `kt-kernel/python/utils/{loader,llamafile}.py`;细节见 [Plan-and-Progress §6.5](DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md)。

### 4.3 端到端验证(✅ 通过)

```bash
# 健康
curl -sf http://127.0.0.1:8000/health        # 200
# 原生 /generate
curl -sS -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是","sampling_params":{"max_new_tokens":40,"temperature":0}}'
#   → "北京，而北京是中国的首都…"            ✅ 连贯正确
# OpenAI 兼容
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"用一句话解释什么是机器学习。"}],"max_tokens":60,"temperature":0}'
#   → "机器学习是一种让计算机通过分析大量数据中的模式来自动改进性能，而无需显式编程的方法。"  ✅
```

**性能**:graph-on(坑⑥/⑥b 修复后)decode `npu graph: True` **3.46–3.89 tok/s**(capture 6.79s);
eager 回退约 **1.6 tok/s**。graph-on 下 F2 整网冒烟 `tools/p27_curl_f2_prompts.sh` 四 prompt 均连贯。

---

## 5. 全坑汇总(从 0 复现必趟)

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| ① | CMake configure 失败,找不到 hwloc | 系统未装 hwloc | `apt-get install -y libhwloc-dev` |
| ② | llamafile 编译 `llama.cpp/ggml-impl.h: No such file` | llama.cpp 子模块在新版(`94a220cd6`),头文件布局变了 | `git -C third_party/llama.cpp fetch --depth 1 origin tag b3173 && git checkout b3173` |
| ③ | `import kt_kernel` → `undefined symbol: iqk_mul_mat_moe_arm82` | `iqk_mul_mat_arm82.cpp` 两行 rename `#define` 被注释(WIP 未完成) | 取消注释 + 重编译 |
| ④ | 转换 `--verify-sample` 报 `newbyteorder` removed | gguf-py NumPy 2.0 不兼容 | `git apply tools/kt_dsv4_npu_patches/llama_cpp/0001-*.patch` |
| ⑤ | 启动崩 `quant fp8 != compressed-tensors` / `n_activated_experts` | **`third_party/sglang` 子模块切错 fork**(`kvcache-ai@c9edb75e0`,无 KT 补丁) | 切到 `dsv4_release@a347a9ad5`(从 `/workspace/code/tmp/sglang` fetch) |
| ⑥ | graph 捕获崩 `aclrtMemcpy 107030` | `mask_cpu_expert_routing` 内 `gpu_experts_mask.to(device)` 在 capture 期做同步 H2D | **✅ 已修(06-08)**:`process_weights_after_loading` 内 capture 前把 mask 预搬 device(§4.2 graph-on) |
| ⑥b | graph 重放崩 `Unsupport run graph with different stream` | `mask_cpu_expert_routing` 被 `@torch.compile`→torchair 绑定 stream 子图,跨 stream 冲突 | **✅ 已修(06-08)**:去掉该函数 `@torch.compile` 改 eager |
| ⑦ | 服务出 token 但乱码(重复,仅 eager 路径) | CPU MoE async submit 没 flush → 输出全零 | eager 下 `KT_FORCE_SYNC_SUBMIT=1`;graph-on 走 host-callback 不涉此坑 |

> 环境约束:Kunpeng 920(aarch64,无 SVE/i8mm)+ Atlas 910B,CANN 8.5.0,Python 3.11.14。
> Handoff 旧文档关于「Q8_0 NaN / MOE_INT8 不可用」的结论**已过时**:实测 Q8_0(int8)CPU offload
> 离线对账 cosine=0.9999,可用;现有 275 GiB Q8_0 GGUF 即可,无需 BF16(555 GiB)。

## 6. 一句话复现(本机现状已全部就绪)

```bash
cd /workspace/code/ktransformers-AK
# graph-on(生产路径,坑⑥/⑥b 已修)：先 npu-smi info 选空闲卡
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=<空闲卡> \
bash tools/p27_launch_ds4flash_npu.sh
# 等加载 → Capture npu graph end → The server is fired up → curl http://127.0.0.1:8000/health → 200
# decode npu graph: True ~3.5–3.9 tok/s；F2 冒烟 bash tools/p27_curl_f2_prompts.sh

# 备：eager 回退  NPU_DEVICE_ID=<卡> KT_FORCE_SYNC_SUBMIT=1 EXTRA_FLAGS="--disable-cuda-graph" bash tools/p27_launch_ds4flash_npu.sh
# 备：dbg 跳过慢加载  KT_DUMMY_CPU_WEIGHTS=1 NPU_DEVICE_ID=<卡> bash tools/p27_launch_ds4flash_npu.sh  （输出无意义，仅调图）
```
