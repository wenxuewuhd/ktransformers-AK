# DeepSeek-V4-Flash 单卡 910B 从 0 拉起服务 —— 全过程实录

> 本文档记录在**干净 container** 上,从源码出发把 DeepSeek-V4-Flash 单卡 Ascend 910B + KT CPU MoE
> 推理服务拉起来的**全部实际操作**(编译 → 转权重 → 拉起 → 使用 + 踩坑)。命令可直接复制执行。
>
> **现行生产路径(2026-06-11)**:**NPU 侧 W8A8 + CPU offload 侧原生 MXFP4 GGUF + graph-on**。
> decode **~13–16 tok/s**(清净窗口),DRAM **~137 GiB**。下面正文按 MXFP4 主线写;
> **旧的 int8(Q8_0)CPU 路径转换与使用见[附录 A](#附录-a旧路径int8q8_0-cpu-权重)。**
>
> - 仓库根:`/workspace/code/ktransformers-AK`
> - NPU 权重(W8A8):`/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8`
> - **CPU 转换源(原生 MXFP4)**:`/workspace/models/DeepSeekV4/DeepSeek-V4-Flash`
> - GGUF 输出:`/workspace/models/cache`
> - Python:`/usr/local/python3.11.14/bin/python3.11`
>
> 总纲(架构/红线/性能账)见 [DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md](DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md);
> 从开源裸仓打 patch 复现见 `tools/kt_dsv4_npu_patches/readme.MD`;
> MXFP4 转换+校验细节见 [mxfp4_gguf_conversion.md](mxfp4_gguf_conversion.md)。

---

## 0. 开工前环境体检

| 项 | 状态 | 备注 |
|---|---|---|
| `/workspace` 持久化 | ✅ | 代码、子模块(含 patch 态)、`.so`、GGUF 权重都在;新 container 通常只缺 hwloc |
| hwloc(libhwloc-dev/15) | ⚠️ 非持久 | **每容器重装**;`import kt_kernel` 运行期依赖 `libhwloc.so.15` |
| sglang / llama.cpp 子模块 | ✅ | sglang `dsv4_release@456687a0f`;llama.cpp b3173 + patch 0001+0002 |
| `kt_kernel_ext*.so` | 视情况 | 持久化;改 C++/换 patch 后需重编 |
| **原生 MXFP4 模型** | 需在位 | `/workspace/models/DeepSeekV4/DeepSeek-V4-Flash`(46 shard,每层专家独占一 shard) |
| MXFP4 GGUF | 视情况 | `/workspace/models/cache/dsv4_layer{0..42}_mxfp4.gguf`(每层 3.42 GiB,合计 138 GiB) |

```bash
apt-get install -y libhwloc-dev libhwloc15                    # ① 每容器必做
find kt-kernel -name "kt_kernel_ext*.so"                      # 是否已编译
ls /workspace/models/cache/dsv4_layer*_mxfp4.gguf | wc -l     # 是否已转(期望 43)
npu-smi info | head -20                                       # 选空闲卡
grep -i "expert_dtype\|num_hidden_layers\|num_experts_per_tok" \
  /workspace/models/DeepSeekV4/DeepSeek-V4-Flash/config.json  # expert_dtype:fp4, 43 层, top-6
```

---

## 1. 编译 kt-kernel(带 Ascend NPU 后端)

```bash
cd /workspace/code/ktransformers-AK/kt-kernel
CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
# 产物 kt-kernel/python/kt_kernel_ext*.so;import 无 undefined symbol
```

> ggml 源里 `GGML_TYPE_MXFP4 not handled in switch` 警告**良性**(非 MoE 路径 op 不需 mxfp4 分支)。
> 换 patch / 改 moe.hpp 后**必须重编**。

### 坑 ①:hwloc 缺失 → CMake configure 失败
`CMakeLists.txt` 把 hwloc 设 `REQUIRED`,系统未装则 `None of the required 'hwloc' found`。
修:`apt-get install -y libhwloc-dev libhwloc15`(`pkg-config --modversion hwloc` → 2.7.0)。容器重启会丢。

### 坑 ②:llama.cpp 子模块版本错 → llamafile 找不到头文件
新版 llama.cpp 把 ggml 头搬进 `ggml/src/`,vendored llamafile 仍按老布局写 `#include "llama.cpp/ggml-impl.h"`。
修:钉公开 tag **b3173**(`a94e6ff`,头在根目录)。本仓已固化此指针。

### 坑 ③:`undefined symbol: iqk_mul_mat_moe_arm82`
`third_party/llamafile/iqk_mul_mat_arm82.cpp` 两行 rename `#define` 曾被注释。本仓已取消注释并 commit,
干净 clone 直接生效;若 `nm -D .so | grep arm82` 见 `U`(未定义),取消注释重编即可。

### 坑(MXFP4 必需):llama.cpp 需打 patch 0001 + 0002
vendored ggml 的 MXFP4 类型在 **patch 0002**(不 commit 进子模块,红线 R7)。干净 b3173 上:

```bash
cd /workspace/code/ktransformers-AK/third_party/llama.cpp
git apply -p1 ../../tools/kt_dsv4_npu_patches/llama_cpp/0001-fix-gguf-NumPy-2-GGUFReader.patch  # NumPy2 读取
git apply -p1 ../../tools/kt_dsv4_npu_patches/llama_cpp/0002-add-ggml-type-mxfp4.patch          # MXFP4 类型(硬依赖)
git -C . status --short   # 应见 ggml.h/ggml-common.h/ggml-quants.{c,h}/ggml.c + gguf-py 改动
```
> 两 patch 不相交、顺序无关。打完**重编 .so**。本工作树通常已打好(随 `/workspace` 持久),换机/裸仓才需。

---

## 2. 原生 MXFP4 → 43 层 GGUF(现行主路径)

转换是**无损 bit-repack**(不是再量化):GGUF 反量化 == 官方 checkpoint 反量化**逐元素 bit-exact**。
唯一雷区是 **nibble 序**(官方 consecutive vs GGUF half-block,转换器逐 32-group 重排,见总纲坑⑬)。

### 2.1 单层快验(开工/换层先做)

```bash
cd /workspace/code/ktransformers-AK
PY=/usr/local/python3.11.14/bin/python3.11
$PY tools/convert_mxfp4_layer_to_gguf.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash --layer-idx 16 --output /tmp/l16_mxfp4.gguf
$PY tools/verify_mxfp4_layer.py --gguf /tmp/l16_mxfp4.gguf --layer-idx 16   # GGUF dequant == 原生 bit-exact
```

> 某 shard 是否下载完整:文件 >135B(非 LFS 指针)且 `8 + header_len + max(data_offsets) == 文件大小`。

### 2.2 全量转换 + 三级校验

```bash
mkdir -p /workspace/models/cache
nohup $PY tools/batch_convert_mxfp4_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash \
  --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --jobs 16 --verify-sample 3 \
  > /tmp/kt_mxfp4_convert.log 2>&1 &
# 输出 dsv4_layer{0..42}_mxfp4.gguf,每层 3.42 GiB,合计 ~138 GiB

# 收尾全集校验(尺寸 + sha256 指纹 + 抽样逐元素)
$PY tools/verify_mxfp4_gguf_set.py --dir /workspace/models/cache \
  --sha256-manifest tools/mxfp4_gguf_sha256.txt
```

> ⚠️ 并发转换曾把某层写截断成 576B → 全集校验会 catch;**收尾务必逐层 audit 文件大小**(都应 3422552640B)。

### 坑 ④:转换 `--verify-sample` 报 NumPy 2.0 `newbyteorder`
gguf-py NumPy 2.0 不兼容(`gguf_reader.py:141`)。修:打 patch `0001`(只影响读取/校验,不影响已写权重)。
切到 b3173 后子模块是未打此补丁的纯基线,必然命中;打上即可。

---

## 3. 运行期 import 自检(拉起前先做)

```bash
cd /workspace/code/ktransformers-AK
[ -e kt-kernel/kt_kernel ] || ln -sfn python kt-kernel/kt_kernel
export PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel"
PY=/usr/local/python3.11.14/bin/python3.11
$PY -c "import kt_kernel, kt_kernel.kt_kernel_ext as e; print('kt_kernel OK')"
$PY -c "import torch, torch_npu; print('torch_npu OK')"
$PY -c "from kt_kernel.utils.loader import GGMLQuantizationType as G; print('MXFP4=', int(G.MXFP4))"  # 应 39
$PY -c "import importlib.util as u; print('kt_ep_wrapper:', u.find_spec('sglang.srt.layers.moe.kt_ep_wrapper') is not None)"
```

---

## 4. 拉起服务(单卡 910B,MXFP4,graph-on)

### 4.1 命令

> **长跑服务在自己的终端前台拉**(`| tee log`,不加 `&`)——remote/后台拉的服务父进程上下文会被回收,
> 表现为成功服务几个请求后突然 `[ERROR] TBE Subprocess ... main process disappeared!`(坑⑭)。
> 先 `npu-smi info` 选空闲卡 + `ss -ltn | grep 8020` 确认端口空。

```bash
cd /workspace/code/ktransformers-AK
NPU_DEVICE_ID=<空闲卡> PORT=8020 \
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  KT_CPUINFER=128 KT_DECODE_TIMING=1 \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/kt_mxfp4_serve.log
```

- `KT_GGUF_TEMPLATE` **必须指 `_mxfp4` 模板**(否则脚本默认走 Q8_0 的 `dsv4_layer{layer_idx}.gguf`)。
- `MODEL_PATH` 指 **W8A8**(NPU 侧 attention/shared/router/常驻专家)。两份权重缺一不可。
- `KT_CPUINFER` 默认 128(每 NUMA 16 线程,留 8 核给 NPU host;**勿 192 满核会 thrash**)。
- graph-on 是默认(坑⑥/⑥b 已修,见总纲 §6.2),勿传 `--disable-cuda-graph`。

### 坑 ⑤:`Quantization method (fp8) does not match (compressed-tensors)`
sglang 子模块切错 fork(无 KT 补丁)。修:切 `dsv4_release@456687a0f`(含 graph 修复 + KT EP wrapper)。
核对:`grep -rl SGLANG_APPLY_CONFIG_BACKUP third_party/sglang/python/` 应为空。本仓已固化。

### 坑 ⑥/⑥b/⑦:graph capture / 重放 / eager 乱码
均已闭合(2026-06-08),现**默认 graph-on 即端到端跑通**。根因与改动见
[总纲 §6.2](DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md)。eager 回退(仅排障)需 `KT_FORCE_SYNC_SUBMIT=1`(坑⑦,附录 A)。

### 4.2 端到端验证(✅ 通过)

```bash
until curl -sf http://127.0.0.1:8020/health >/dev/null; do sleep 5; done    # 加载 ~2–3.5min 热 cache
# 四 prompt 连贯(单发顺序,别并发)
PORT=8020 bash tools/p27_curl_f2_prompts.sh
# 单发看吞吐
curl -sS -m 300 -X POST http://127.0.0.1:8020/generate -H 'Content-Type: application/json' \
  -d '{"text":"请详细解释什么是 Transformer 架构。","sampling_params":{"max_new_tokens":200,"temperature":0}}'
grep -E "KT_DECODE_TIMING|gen throughput" /tmp/kt_mxfp4_serve.log | tail
```

**实测(2026-06-11,中等争抢窗口 load~84–134)**:四 prompt 连贯("Transformer 架构是一种革命性的模型...
自注意力机制(Self-Attention)和位置编码");**decode 13.4–14.7 tok/s,cpu_moe_wall min 17.2 / median 22.6ms**
(p90 120ms 是邻居尖峰)。清净独占窗口可达 ~16 tok/s。

> ⚠️ **`--max-running-requests 1`,别并发多发**(并发撞争抢窗口会触发 NPU runtime 失稳崩,坑⑭ 同源)。
> 收服务:跑服务的终端 `Ctrl-C`(优雅释放 HBM);**绝不 `pkill -f sglang.launch_server`**(杀别 session、自杀 shell、留孤儿占 HBM)。

---

## 5. 全坑速查(从 0 复现必趟)

| # | 现象 | 修复 |
|---|------|------|
| ① | CMake 找不到 hwloc | `apt-get install -y libhwloc-dev libhwloc15`(每容器) |
| ② | llamafile `ggml-impl.h: No such file` | 钉 llama.cpp b3173(`a94e6ff`) |
| ③ | `undefined symbol: iqk_mul_mat_moe_arm82` | 取消 `iqk_mul_mat_arm82.cpp` 两行注释 + 重编(已 commit) |
| ④ | `--verify-sample` 报 `newbyteorder` | patch `0001`(NumPy2) |
| MXFP4 | CPU MoE 吃 MXFP4 需类型注册 | patch `0002`(`GGML_TYPE_MXFP4=39` + NEON kernel,硬依赖) |
| ⑤ | `quant fp8 != compressed-tensors` | sglang 切 `dsv4_release@456687a0f` |
| ⑥/⑥b/⑦ | graph capture/重放崩 / eager 乱码 | 已修(总纲 §6.2);默认 graph-on |
| ⑬ | MXFP4 输出乱码/对账偏 | nibble 序逐 32-group 重排;`verify_mxfp4_layer.py` bit-exact 闸门 |
| ⑭ | 服务跑一会儿 `main process disappeared` | 长跑服务在自己终端前台拉 |

> 环境:Kunpeng 920(aarch64,无 SVE/i8mm)+ Atlas 910B,CANN 8.5.0,Python 3.11.14。

---

## 6. 一句话复现

```bash
cd /workspace/code/ktransformers-AK
apt-get install -y libhwloc-dev libhwloc15                # 每容器
# 自己终端前台拉(MXFP4,graph-on);先 npu-smi info 选空闲卡
NPU_DEVICE_ID=<空闲卡> PORT=8020 \
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  KT_CPUINFER=128 MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh
# 等加载 → health 200 → PORT=8020 bash tools/p27_curl_f2_prompts.sh
```

---

## 附录 A:旧路径——int8(Q8_0)CPU 权重

> Q8_0 是 MXFP4 之前的 CPU offload 路径(int8,1.0625 B/元素,275 GiB)。**现行生产已换 MXFP4,
> 后续 CPU 迭代不再基于 Q8_0。** 保留供:① 无原生 MXFP4 权重时回退;② 对照基线。

**W8A8 → Q8_0 GGUF**(dequant→requant 的**再量化**,非无损 repack):

```bash
/usr/local/python3.11.14/bin/python3.11 tools/batch_convert_w8a8_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --quant q8_0 --jobs 32 --verify-sample 3
# 输出 dsv4_layer{0..42}.gguf(无 _mxfp4 后缀),每层 ~6.85 GiB,合计 ~275 GiB
# --jobs 32 较优(聚合 ~129/192 核,磁盘 I/O 成瓶颈);也支持 --quant bf16(数值基线)
```

**用 Q8_0 拉起**(不传 `KT_GGUF_TEMPLATE` 即走 Q8_0 默认模板):

```bash
NPU_DEVICE_ID=<空闲卡> PORT=8000 KT_CPUINFER=128 \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh
# 默认 KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}.gguf'
```

**坑 ⑦(eager 乱码)**:eager 路径 CPU MoE async submit 未 flush → 输出全零("我,我,我…"退化)。
graph-on 走 host-callback 不涉此坑;仅 eager 回退需 `KT_FORCE_SYNC_SUBMIT=1`:

```bash
NPU_DEVICE_ID=<卡> KT_FORCE_SYNC_SUBMIT=1 EXTRA_FLAGS="--disable-cuda-graph" \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 bash tools/p27_launch_ds4flash_npu.sh
```

**Q8_0 离线对账**:

```bash
PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel" /usr/local/python3.11.14/bin/python3.11 \
  tools/p27_cpu_moe_reference_check.py --w8a8 /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --gguf /workspace/models/cache/dsv4_layer3.gguf --layer-idx 3 --method LLAMAFILE
# Q8_0 cosine 0.9999;BF16 0.999997
```

> 历史结论修正:Spec/Handoff(05-12)的「Q8_0 aarch64 NaN / MOE_INT8 必须 BF16」已过时——
> Q8_0 实测可用(坑⑧:无 i8mm 时回退 `ggml_vec_dot_q8_0_q8_0`)。Q8_0 kernel 也做了行内预取优化
> (2.38×,`kt_vec_dot_q8_0_q8_0`,`KT_Q8_REF=1` 回退,ggml 零改动),惠及其他 W8A8 模型。
