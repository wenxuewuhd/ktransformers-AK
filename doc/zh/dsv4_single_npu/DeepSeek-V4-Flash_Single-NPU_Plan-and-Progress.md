# DeepSeek-V4-Flash 单卡 910B + K920 CPU MoE —— 总体方案与当前进展（整合版）

> **文档定位**：本文是 DeepSeek-V4-Flash「单卡 Ascend 910B + Kunpeng-920 CPU MoE offload」方案的**唯一现行总纲**。
> 当各来源冲突时**以本文为准**。维护分支：`dsv4_one_card_dev`。
> **最后更新：2026-06-11**（CPU MoE 换原生 **MXFP4**，搬运字节减半，decode ~8.5→~13–16 tok/s，
> DRAM 275→137 GiB；Session D 已合入主干 commit `91b9c92`）。
>
> | 来源 | 时间 | 角色 | 现状 |
> |---|---|---|---|
> | `DeepSeek-V4-Flash-K920-Single-NPU-Spec.md` | 05-12 | 盲写规格书 | 架构/接口有效;**量化结论已过时** |
> | `DeepSeek-V4-Flash-K920-Single-NPU-Handoff.md` | 05-12 | Phase 计划与红线 | 红线有效;**Q8_0/KML 结论已过时** |
> | `DeepSeek-V4-Flash_单卡910B_从0拉起服务全记录.md` | — | 从 0 拉起实操 | 现行（已刷新为 MXFP4 主线） |
> | `mxfp4_cpu_moe_handoff.md` / `mxfp4_gguf_conversion.md` | 06-10 | **Session D：CPU MXFP4** | **CPU 侧最新事实基准** |

---

## 0. 一句话现状

单卡 910B + K920 的 DeepSeek-V4-Flash 推理**已端到端拉起、输出连贯、NPU graph 性能路径闭合**。
**当前生产配置（2026-06-11）**：

> **NPU 侧 W8A8**（attention MLA+NSA+Indexer / shared expert / router / 前 32 个常驻 routed expert）
> **＋ CPU offload 侧原生 MXFP4 GGUF**（其余 224 专家/层，~137 GiB）**＋ graph-on**。

这是「两份权重」混合方案（对标 R1/V3 Ascend 教程的 Q4+W8A8 合并思路）：NPU 吃 W8A8 safetensors，
CPU 吃由**官方原生 MXFP4 checkpoint** 无损 repack 出来的 GGUF。

**演进账**（decode，真实权重，graph-on）：

| 里程碑 | decode | CPU MoE/token | 备注 |
|---|---|---|---|
| eager 基线 | ~1.6 tok/s | — | 功能对照 |
| graph + `kt-cpuinfer 24` | ~3.6 | ~215 ms | 06-08 graph 闭合 |
| graph + `kt-cpuinfer 96→128` | ~6.1→8.5 | ~115→55 ms | 06-09 带宽瓶颈定位 |
| **graph + CPU MXFP4 + 行内预取 kernel** | **~13–16 tok/s** | **~17–27 ms** | **06-11 现行**（清净窗口 ~16，中等争抢 ~13–14，独占实测） |

> 瓶颈定性（不变）：CPU MoE 深度 **memory-bound**（`AI=0.94 MAC/byte ≪ 平衡点 21`），decode 时间
> 主要花在把专家权重从 DDR 搬一遍。MXFP4 把搬运字节**精确减半**（Q8_0 1.0625 → MXFP4 0.53125 B/元素），
> 是这一版提速的主因；叠加 kernel 行内软预取（2.4×，根治 TSV110 硬件预取器跟不上 GEMV 低密度 load 流）。
> 详见 [mxfp4_cpu_moe_handoff.md](mxfp4_cpu_moe_handoff.md) + [graph_decode_bandwidth_findings.md](graph_decode_bandwidth_findings.md)。

---

## 1. 环境与硬件规格

### 1.1 硬件

| 部件 | 配置 |
|---|---|
| **CPU** | Kunpeng-920 5250,4 socket × 48 core = **192 物理核**,8 NUMA(每 NUMA 24 核 ~192 GB),**1.5 TB DRAM** |
| **CPU ISA** | ARMv8.2-A + `asimddp`(NEON SDOT)+ `asimdhp`/`fphp`(FP16);**无 SVE / 无 BF16 / 无 I8MM / 无 SME** |
| **DDR** | 每 NUMA 实插 **3/4 通道**（24 DIMM/32 槽）DDR4-**3200** → 真 spec **614 GB/s**；清净独占聚合 ~442 GB/s |
| **NPU** | 8 × Atlas 910B（每张 64 GB HBM）;**项目只用 1 张** |
| **CANN** | 8.5.0,`/usr/local/Ascend/ascend-toolkit/latest` |

> **ISA 红线(R1)**：任何 SVE / BF16 / I8MM 指令（`+sve` march、SVE 汇编、`__bf16`、`smmla`/`usdot`/`ptrue`）
> 在 K920 上 **SIGILL**。march 固定 `-march=armv8.2-a+fp16+dotprod`。MXFP4/Q8_0 kernel 只用 NEON
> `vqtbl1q_s8`（查表）+ `vdotq_s32`（SDOT）。

### 1.2 软件栈

| 组件 | 版本/路径 |
|---|---|
| OS / Python | Ubuntu 22.04 aarch64 / `/usr/local/python3.11.14/bin/python3.11` |
| PyTorch / torch_npu | `2.8.0+cpu` / `2.8.0.post2` |
| SGLang | 子模块 `third_party/sglang/`，分支 `dsv4_release` @ **`456687a0f`**（含 graph 修复，§6.3） |
| llama.cpp | 子模块 `third_party/llama.cpp/`，公开 tag **b3173**（`a94e6ff`）+ **patch 0001（NumPy2）+ 0002（MXFP4 类型）** |
| hwloc | `libhwloc-dev / libhwloc15 2.7.0`（kt-kernel 硬依赖;**每容器需重装**，见 §4.1） |

> **子模块 patch 机制（关键）**：vendored llama.cpp 的 ggml 改动**不 commit 进子模块**，全部在
> `tools/kt_dsv4_npu_patches/llama_cpp/000{1,2}-*.patch`。**`0002` 是 CPU MXFP4 路径的硬依赖**
> （注册 `GGML_TYPE_MXFP4=39` + NEON kernel + gguf-py 枚举）。两 patch 文件不相交、顺序无关。

### 1.3 模型规格与两份权重

`num_hidden_layers=43`（全 MoE，`first_k_dense_replace=0`）、`hidden_size=4096`、`n_routed_experts=256`、
`num_experts_per_tok=6`（top-k，**不是 8**）、`n_shared_experts=1`、`moe_intermediate_size=2048`、
`head_dim=512`、`num_attention_heads=64`、`num_key_value_heads=1`；Attention=MLA+NSA+Lightning Indexer
（`index_topk=512`）；`num_nextn_predict_layers=1`（MTP，**本项目禁用**，见 [mtp 实测](#)，不划算）。

**两份权重（缺一不可）**：

| 用途 | 路径 | 格式 | 谁用 |
|---|---|---|---|
| **NPU 侧** | `/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8` | W8A8 safetensors（int8 + fp32 per-channel scale） | attention / shared / router / 前 32 常驻专家;启动 `MODEL_PATH` 指它 |
| **CPU 转换源** | `/workspace/models/DeepSeekV4/DeepSeek-V4-Flash` | **原生 MXFP4**（`expert_dtype:"fp4"`，E2M1 nibble + ue8m0 scale） | 转成 GGUF 喂 CPU offload 专家 |

> 原生 MXFP4 专家张量：`layers.{L}.ffn.experts.{i}.w1/w3/w2.weight`（`I8`，gate/up `[2048,2048]`、
> down `[4096,1024]`，K nibble-packed 成 K/2）+ `.scale`（`F8_E8M0`，`[2048,128]` / `[4096,64]`，K/32 分组）。
> 46 shard，**每层专家独占一个 shard**（如 shard 00018 = layer 16）。`router gate` / `shared_experts` 留 NPU（红线 R8）。
> **Q8_0/BF16（旧 CPU 路径）转换与使用见[附录 A](#附录-aint8q8_0-cpu-权重旧路径附录)。**

---

## 2. 系统架构与数据流

```
单卡：Atlas 910B (64 GB HBM) + K920 (1.5 TB DRAM, 192 核, 8 NUMA)

input → [NPU: embedding / RoPE / MLA+NSA+Indexer attention]
      → [NPU: MoE router gate → topk_ids, topk_weights(k=6)]
      → ┌──────────────────────────┬──────────────────────────────┐
        │ NPU experts (N=32 默认)   │ CPU experts (224 默认)          │
        │ W8A8 safetensors         │ kt-kernel LLAMAFILE GGUF        │
        │ + shared experts (常驻)   │ **MXFP4（现行）** / Q8_0（附录） │
        └──────────────────────────┴──────────────────────────────┘
      → merge → linear + residual → 下一层
```

### 2.1 NPU 端
Attention MLA+NSA+Lightning Indexer+Compressor（SGLang `--attention-backend ascend`）；NPU MoE
`fused_experts_npu`（W8A8）承载前 N routed + shared + router topk；KV cache 在 HBM。

### 2.2 CPU 端（kt-kernel）
- backend：LLAMAFILE（`kt-kernel/operators/llamafile/moe.hpp` → `LLAMA_MOE_TP`）。**对权重量化类型泛化**
  （buffer 尺寸 / 激活量化 / NUMA TP / 加载加速 / graph callback 全经 ggml `type_traits`，换 MXFP4 这条线不改）。
- 线程池：8 NUMA worker pool，默认 `--kt-cpuinfer 128 --kt-threadpool-count 8` → **每 NUMA 16 线程**
  （留 8 核/NUMA 给 NPU host + scheduler）。**`128` 是甜点；192 满核会 thrash 崩**（无余量）。
- **MXFP4 GEMV kernel**（`ggml_vec_dot_mxfp4_q8_0`，patch 0002）：`vqtbl1q_s8` 查 E2M1 表 → `vdotq_s32`
  SDOT → e8m0 scale；**行内 `__builtin_prefetch(+512B)` + 双 float32x4 FMA 累加链**（2.4×）。激活在线量化到 Q8_0
  （`vec_dot_type=Q8_0`，唯一数值损失源，cosine 0.9999）。
- Expert layout（同 Q8_0 块方向）：gate/up 沿 hidden(4096) 分块、down 沿 intermediate(2048) 分块。

### 2.3 NPU↔CPU 桥（graph callback，任务2）
`kt-kernel/cpu_backend/ascend_callback_worker.{cpp,h}`：后台线程 `aclrtSubscribeReport`+循环
`aclrtProcessReport`，把 CPU MoE submit/flush 接进 NPU graph host callback。红线 R2/R3：ACL
`aclrtLaunchCallback` 不会自动触发，必须专用 poller 线程，否则卡 sync、NPU 空闲。

### 2.4 SGLang 集成
模型 `third_party/sglang/.../models/deepseek_v4.py`；KT wrapper `…/layers/moe/kt_ep_wrapper.py`
（per-layer `KTMoEWrapper`、`mask_cpu_expert_routing`、prefill/decode 分化、graph 走 host callback）；
设备抽象 `…/utils/kt_accel.py`；Triton 兜底 `SGLANG_NPU_ALLOC_FORCE_NAIVE=1`（不影响数值）。

---

## 3. 量化与权重方案 ⚠️（现行：CPU MXFP4）

### 3.1 现行事实

CPU offload 专家用**官方原生 MXFP4**（无损 repack 成 GGUF）。NPU 侧维持 W8A8。

| CPU 格式 | 字节/元素 | 单专家 | 最恶劣每层(top-6 全 CPU) | 43 层 DRAM | 现状 |
|---|---|---|---|---|---|
| **MXFP4** | **0.53125**（17B/32） | **13.4 MB** | **80 MB** | **~137 GiB** | **现行生产** |
| Q8_0 | 1.0625（34B/32） | 26.7 MB | 160 MB | ~275 GiB | 旧路径，附录 A |
| BF16 | 2.0 | ~50 MB | — | ~555 GiB | 数值基线，附录 A |

> MXFP4 是**官方发布的量化**（训练侧已对齐），转 GGUF 全程 **bit 级无损 repack**——不是再量化，
> 比"W8A8 int8 再砍到 4bit"的双重量化干净得多。CPU 专家 MXFP4 + NPU 专家 W8A8 混用无碍
> （各专家独立近似同一母权重）。离线对账 cosine **0.999939** / max_rel 1.12%（唯一损失=激活 Q8）。

### 3.2 原生 MXFP4 → GGUF 转换（现行主路径）

**详细转换 + 三级校验见 [mxfp4_gguf_conversion.md](mxfp4_gguf_conversion.md)。** 要点：

- **nibble 序是核心雷区（坑⑩同类）**：官方 ckpt 是 **consecutive**（byte i = K位 2i/2i+1），上游 GGUF
  是 **half-block**（qs[j] = K位 j/j+16）。转换器**逐 32-group 重排 nibble**（不是 byte copy！），
  e8m0 scale 字节原样直存。转换器与 kernel 必须同一约定，对账是裁判。
- 单层：`tools/convert_mxfp4_layer_to_gguf.py`；全量：`tools/batch_convert_mxfp4_layers_mp.py`。
- 校验三件套：`verify_mxfp4_layer.py`（GGUF dequant **逐元素 bit-exact** == 原生 dequant）、
  `p27_cpu_moe_reference_check_mxfp4.py`（kernel cosine 0.9999）、`verify_mxfp4_gguf_set.py`（全集
  尺寸+sha256+抽样三级，对 `tools/mxfp4_gguf_sha256.txt`）。命令见 §4.3 / §9。
- 产物：`/workspace/models/cache/dsv4_layer{0..42}_mxfp4.gguf`，每层 **3.42 GiB**，合计 **~138 GiB**。
  ⚠️ 并发转换曾把某层写截断（576B），**收尾务必逐层 audit 文件大小**（全集校验会 catch）。

### 3.3 KML / MOE_INT8 —— **不做**
K920 无 SVE/i8mm，KML `usdot`/`ptrue` 不可编译；MXFP4 已满足精度且字节最省，不投入。

---

## 4. 复现 / 拉起流程（从干净 container 起）

### 4.1 每次新 container 必做（非持久项）

`/workspace` 持久化代码、子模块内容（含 patch apply 态）、`.so`、GGUF 权重。**唯一非持久**是 apt 的 hwloc：

```bash
apt-get install -y libhwloc-dev libhwloc15      # 运行期 import kt_kernel 依赖 libhwloc.so.15;cmake 重编也需
```

> 若 import 报 "kt_kernel is not installed" 先查 `libhwloc.so.15`。其余（sglang/llama.cpp patch 态、
> `.so`、MXFP4 GGUF）随 `/workspace` 持久,通常无需重做。

### 4.2 从裸仓复现（仅首次/换机；本机已就绪可跳过）

```bash
# (1) 子模块：llama.cpp 钉 b3173 + 打 patch 0001（NumPy2）+ 0002（MXFP4 类型，CPU MXFP4 硬依赖）
cd third_party/llama.cpp                          # 干净 b3173 (a94e6ff)
git apply -p1 ../../tools/kt_dsv4_npu_patches/llama_cpp/0001-fix-gguf-NumPy-2-GGUFReader.patch
git apply -p1 ../../tools/kt_dsv4_npu_patches/llama_cpp/0002-add-ggml-type-mxfp4.patch
#     sglang 钉 dsv4_release@456687a0f（含 graph 修复）

# (2) 编译 kt-kernel（带 Ascend NPU 后端）
cd /workspace/code/ktransformers-AK/kt-kernel
CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
#     预期 march=armv8.2-a+fp16+dotprod;import kt_kernel_ext 无 undefined symbol
```

> ggml 源里出现 `GGML_TYPE_MXFP4 not handled in switch` 警告是**良性**（非 MoE 路径的 op 不需要 mxfp4 分支）。

### 4.3 原生 MXFP4 → 43 层 GGUF（现行主路径）

```bash
mkdir -p /workspace/models/cache
nohup /usr/local/python3.11.14/bin/python3.11 tools/batch_convert_mxfp4_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash \
  --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --jobs 16 --verify-sample 3 \
  > /tmp/kt_mxfp4_convert.log 2>&1 &
# 输出 dsv4_layer{0..42}_mxfp4.gguf（每层 3.42 GiB，合计 ~138 GiB）

# 收尾全集校验（尺寸 + sha256 + 抽样逐元素）
python3 tools/verify_mxfp4_gguf_set.py --dir /workspace/models/cache \
  --sha256-manifest tools/mxfp4_gguf_sha256.txt
```

> 单层快验（开工/换层时）：
> `python3 tools/convert_mxfp4_layer_to_gguf.py --input <MXFP4模型> --layer-idx 16 --output /tmp/l16.gguf` →
> `python3 tools/verify_mxfp4_layer.py --gguf /tmp/l16.gguf --layer-idx 16`（bit-exact）。

### 4.4 拉起服务（MXFP4，graph-on）

先 `npu-smi info` 选空闲卡（避开别容器/别 session）。**长跑服务建议在自己的终端前台拉**
（remote/后台拉的服务父进程上下文会被回收 → `main process disappeared`）。

```bash
cd /workspace/code/ktransformers-AK
NPU_DEVICE_ID=<空闲卡> PORT=8020 \
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  KT_CPUINFER=128 KT_DECODE_TIMING=1 SKIP_WARMUP=0 \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh 2>&1 | tee /tmp/kt_mxfp4_serve.log
```

- `KT_GGUF_TEMPLATE` **必须指 `_mxfp4` 模板**（否则脚本默认走 Q8_0 的 `dsv4_layer{layer_idx}.gguf`）。
- `MODEL_PATH` 指 **W8A8**（NPU 侧）。两者缺一不可。
- `KT_CPUINFER` 默认 128（`{layer_idx}` 花括号写法见脚本注释，勿被 `${var:-}` 吃掉）。
- **`SKIP_WARMUP=0` serving 建议开**：去掉脚本默认的 `--skip-server-warmup`，开机暖一次 NPU graph/cache →
  开机第一发不再冷（§7 实测 req1 2.1–2.5→1.8s，两臂区间不重叠，预热 pass 仅 ~5s）。默认 `SKIP_WARMUP=1` 保持基线。
- graph-on 是默认（勿传 `--disable-cuda-graph`）。eager 回退见[附录 A](#附录-aint8q8_0-cpu-权重旧路径附录) / §6.3。
- **`KT_NSA_COMPRESSOR_MODE`（跨 CANN 版本兼容开关，`split` 默认 | `single`）**：NSA compressor
  有两套互斥 ABI —— **CANN 8.5.0** 用私有 19 参 split-state（两个分开的 kv/score buffer），
  **CANN 9.0.0+** 用公开 18 参 single-state（一个交织 `[kv|score]` state_cache）。纯环境变量选择、
  **不自动探测**。**本镜像(8.5.0)不设即默认 `split` = 原行为**;若在 **9.0.0** 机器上必须
  `export KT_NSA_COMPRESSOR_MODE=single`(否则崩在 compressor 参数不匹配)。实现见
  `hardware_backend/npu/nsa_compressor_mode.py`,细节见 `A3_W8A8_数值对齐调查.md`。

### 4.5 验证（加载 ~2–3.5 min 热 cache）

```bash
until curl -sf http://127.0.0.1:8020/health >/dev/null; do sleep 5; done   # 就绪
PORT=8020 bash tools/p27_curl_f2_prompts.sh                                  # 四 prompt 连贯
curl -sS -X POST http://127.0.0.1:8020/generate -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是","sampling_params":{"max_new_tokens":64,"temperature":0}}'
# 期望连贯（"北京…"）;统计 grep KT_DECODE_TIMING / "gen throughput" /tmp/kt_mxfp4_serve.log
```

> ⚠️ **`--max-running-requests 1`，别并发多发**（并发撞争抢窗口会触发 NPU runtime 失稳崩）。单发顺序跑稳。
> 收服务：跑服务的终端 `Ctrl-C`（优雅释放 HBM）;**绝不 `pkill -f sglang.launch_server`**（杀别 session + 自杀 shell）。

### 4.6 编译 depool 的 AscendC MXFP4 算子（**仅 depool 路径需要**；普通 MXFP4 GGUF 路径不用）

depool（`KT_MXFP4_DEPOOL=1`）用一个 **AscendC device kernel**（MXFP4 dequant + NZ 转换，device→device、ctypes 调）。
**源码随 G 合入已 tracked**：`tools/ascendc_mxfp4/`（`mxfp4_{fused,dq,nz,oscale}_kernel.cpp` + host launcher `mxfp4_fused_op.py`
+ 测试 + `HANDOVER_SESSION_C.md`）。**编译产物 `*.so` 被 gitignore（不入库），换机/首跑须现编。**

一次性环境（bisheng/tikcpp 头文件来源）：
```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0
export ASCEND_RT_VISIBLE_DEVICES=<空卡>      # npu-smi 选空卡，避开 card 2
```

- **方式 1（推荐，自动）**：depool 服务首次用到算子时**按需 bisheng 编译并缓存**（源码比 `.so` 新会自动重编）。
  启动时 `KT_MXFP4_OP_DIR` 指向 `$REPO/tools/ascendc_mxfp4`（**指主干自己的目录，别再指 kt-G worktree**，G worktree 会删）。什么都不用做。
- **方式 2（手动，想自己确认编译链）**：
  ```bash
  cd tools/ascendc_mxfp4
  CANN=$ASCEND_TOOLKIT_HOME; TK=$CANN/aarch64-linux/tikcpp
  bisheng -x asc --cce-aicore-arch=dav-c220 -O2 -std=c++17 -fPIC -shared \
    -I$TK/tikcfw -I$TK/tikcfw/impl -I$TK/tikcfw/interface -I$TK/tikcfw/lib \
    -I$CANN/aarch64-linux/include \
    mxfp4_fused_kernel.cpp -o libmxfp4fused.so \
    -L$CANN/aarch64-linux/lib64 -lruntime -lascendcl   # 无输出即成功
  ```
- **算子自验**（确认对，再集成）：`ASCEND_RT_VISIBLE_DEVICES=<空卡> python3 tools/ascendc_mxfp4/test_fused_e2e.py`
  → 期望 `cos(fused, fp32-golden) = 0.99999976 PASS`。
- AscendC 写核硬坑（UB 192KB、偏移须 int64、grid<65536、嵌套循环编译极慢等）见 `HANDOVER_SESSION_C.md` + memory。

---

## 5. 全坑汇总

| # | 现象 | 根因 | 修复 / 现状 |
|---|------|------|------|
| ① | CMake 找不到 hwloc | 系统未装 | `apt-get install -y libhwloc-dev libhwloc15`(**每容器**) |
| ② | llamafile 编译 `ggml-impl.h: No such file` | llama.cpp 子模块版本错 | 钉 b3173(`a94e6ff`),头在根目录 |
| ③ | `undefined symbol: iqk_mul_mat_moe_arm82` | `iqk_mul_mat_arm82.cpp` 两行 rename 被注释 | 取消注释 + 重编(已 commit 进主仓) |
| ④ | `--verify-sample` 报 `newbyteorder` removed | gguf-py NumPy 2.0 不兼容 | patch `0001`(只影响读取/校验) |
| ⑤ | 启动崩 `quant fp8 != compressed-tensors` | sglang 切错 fork | 切 `dsv4_release@456687a0f` |
| ⑥/⑥b | graph capture `aclrtMemcpy 107030` / 重放 `Unsupport run graph` | mask H2D 同步 / `@torch.compile` 跨 stream 子图 | **✅ 已修(06-08)**,见 §6.3 |
| ⑦ | eager 出 token 但乱码 | CPU MoE async submit 未 flush → 静默算错 | **✅ 已根治(e5f53ad,坑⑱)**;eager `force-sync=0` 即对 |
| ⑧ | Q8_0 aarch64 NaN(历史) | iqk/tinyBLAS dotprod-only NaN | 回退 ggml `vec_dot_q8_0_q8_0`,Q8_0 可用 |
| ⑨ | `--chunked-prefill-size -1` → malloc 越界 | `max_len=-1` 按 1 分配 | 默认 2048;`llamafile.py` ≤0 回落 |
| ⑩ | GGUF layout 错 → 输出退化 | nibble/permute 与 pointer 算术不一致 | 去 permute、expert 维 axis=0(Q8_0);**MXFP4 见 ⑬** |
| ⑪ | Triton-on-NPU `0 active drivers` | triton×triton-ascend 错配 | torch fallback;`SGLANG_NPU_ALLOC_FORCE_NAIVE=1` |
| ⑫ | apt 镜像签名 403 | Huawei ports GPG | `--allow-unauthenticated -o Acquire::AllowInsecureRepositories=true` |
| **⑬** | **MXFP4 GGUF 输出乱码/对账偏** | **nibble 序：官方 consecutive vs GGUF half-block 未重排** | **转换器逐 32-group 重排 nibble（非 byte copy）;`verify_mxfp4_layer.py` bit-exact 闸门(§3.2)** |
| ⑭ | 服务跑一会儿 `main process disappeared` | remote/后台拉的服务父进程上下文被回收 | **长跑服务在自己终端前台拉**（见 §4.4） |
| ⑮ | 离线单层对账 cand 全零 | 孤立单层调用 stream-callback 路径不回写 | 竞态已根治(e5f53ad);对账脚本仍内置 sync,无需手设 |
| ⑯ | **prompt > `chunked-prefill-size` 即崩** `loc.numel()=1024 vs cache.shape[0]=513` | **DSv4 NSA-Compressor 跨 chunk 索引对齐缺口**：分块 prefill 时带 prefix 的 chunk，Lightning Indexer 按逻辑长度选 `loc`，但压缩池(c4/c128)只增量建到当前 chunk → loc 数与压缩池长度脱节。压缩/稀疏簿记只在**单遍完整 prefill**成立（普通 MLA/MHA 无此问题） | **单 chunk 绕过**：`CHUNKED_PREFILL_SIZE ≥ prompt_len`（32k→32768）。⚠️ 与 64k OOM 是同一死结两面（单 chunk 避 bug 但 prefill activation 撑爆 HBM §7.1）。根治需修 ascend backend 的 c4/c128 跨 chunk 累积+loc 对齐（sglang 侧，见 §6.6 / 合 upstream 时检查） |

---

## 6. 当前修改与进展

### 6.1 功能进展状态

| 模块 | 状态 |
|---|---|
| Phase 0 编译期 NPU 适配 | ✅ |
| 单卡整网 wiring（SGLang + CPU MoE） | ✅ HTTP 200,连贯 |
| NPU graph + ACL callback worker | ✅ 闭合(06-08,§6.3) |
| CPU 权重加载加速(zero-copy + 并行重排) | ✅ 43 层 ~47s(`CPU权重加载加速_P0-P1.md`) |
| graph decode 提速(kt-cpuinfer 24→96→128 + GEMV prefetch) | ✅ 06-09 |
| **CPU MoE 原生 MXFP4(字节减半 + kernel 2.4×)** | **✅ 合入主干(06-11,§6.7);decode ~13–16 tok/s** |

### 6.2 ✅ graph capture 修复（2026-06-08 闭合，保留）

真因两层（均不在当时嫌疑栈）：① `kt_ep_wrapper.py::mask_cpu_expert_routing` 的
`gpu_experts_mask.to(device)` 在 capture 期做同步 H2D（107030）；② 该函数被 `@torch.compile`→NPU
torchair 编成绑定 stream 的独立子图，与外层图跨 stream replay 冲突（`Unsupport run graph`）。
**改动（sglang `456687a0f`）**：capture 前预搬 mask 到 device（`.to()` no-op）+ 去掉 `@torch.compile`
改 eager 折进外层图 + 预订阅 ACL report stream；kt-kernel `experts_base.py::_wait_device` 加 capture
兜底。详尽根因见 [graph_mode_fix_handoff.md](graph_mode_fix_handoff.md)（5 个旧嫌疑点均证伪）。

### 6.3 🔧 dbg 期绕过 CPU MoE 慢加载（`KT_DUMMY_CPU_WEIGHTS`）

调 graph/capture 反复重启时,真实权重 GGUF 读取是主要开销。`KT_DUMMY_CPU_WEIGHTS=1` **跳过磁盘读取**,
按张量元数据 fabricate 同字节布局零 buffer（C++ MOE/load_weights_task 路径不变,capture 与 forward
忠实执行）。MXFP4 类型注册后天然支持。⚠️ 输出无意义,**严禁精度验收**,仅"图能否跑通"。

### 6.4 ⚡ graph decode 提速（kt-cpuinfer + GEMV prefetch，06-09）

CPU MoE 是内存带宽瓶颈,旧 `--kt-cpuinfer 24` 只用 24/192 核。提到 **96→128**（隔离微基准证明
旧"≥128 thrash"是在线争抢假象,**只有 192 满核才崩**）+ GEMV 行内预取 → decode 3.6→8.5 tok/s。
详见 [graph_decode_bandwidth_findings.md](graph_decode_bandwidth_findings.md)。

### 6.5 NPU 侧天花板 / MTP（Session B，已收口，未采用）

纯 NPU decode 天花板 ≈ 19.8 tok/s（逐层 host callback ~7.4ms/token 往返不可消除）;独立 stream 并行
−35% 回归（同步开销 > 可重叠工作量）;MTP accept_len ~1.8 < 盈亏线 2.5,不划算。详见
[npu_decode_ceiling_and_callback_findings.md](npu_decode_ceiling_and_callback_findings.md) /
[mtp_on_npu_findings.md](mtp_on_npu_findings.md)。**结论：CPU 侧（字节/带宽）是当前主杠杆。**

### 6.6 其它开放项

| 优先级 | 任务 | 备注 |
|---|---|---|
| P1 | 热专家放置（EPLB 动态） | 当前硬编码前 32;按 activation 频次取最热 → 更多 top-6 命中 NPU、CPU 搬字节再降 |
| P2 | CPU↔NPU overlap | MXFP4 后 CPU MoE ~17–27ms vs NPU ~50ms,NPU 成主导,overlap 价值上升（Session B 线） |
| P3 | 长序列 prefill 流式加载 + 热专家常驻 | **部分已落地（§6.9，merge `a65995b`）**：depool 池 model-load 并行构建 + 流式 prefill + dynamic-resident 已进主干（opt-in）；实时 cache/evict 策略仍待续 |
| P4 | 预取距离扫描 / down 短行 | 512B 未调优;down nrc=2、跨专家预取已试为负结果（Session D 收口） |
| P2 | **修 chunked prefill 跨 chunk（sglang 侧）** | 坑⑯：NSA-Compressor c4/c128 跨 chunk 累积 + indexer loc 对齐。修好 → 长 prompt 可分块 prefill → activation 峰值降 → **64k 不靠单 chunk 也能跑、不 OOM**（解 64k 的另一条路，对比降常驻专家）。⚠️ **合 sglang upstream 时优先查这条**：upstream 刚支持 ascend DSv4，看它的 chunked-prefill/compressor 是否已正确处理跨 chunk（若已修，直接采纳;若没修，把本问题反馈/PR 上去） |

### 6.7 🔥 CPU MoE 原生 MXFP4（Session D，2026-06-11 合入 `91b9c92`）

- **P1 类型注册**：`GGML_TYPE_MXFP4=39`（`block_mxfp4{e;qs[16]}` blck32 size17，`vec_dot_type=Q8_0`）。
  改 vendored ggml.h/ggml-common.h/ggml-quants.{c,h}/ggml.c + kt loader.py + gguf-py，全在 **patch 0002**。
- **P2 无损转换器**：consecutive→half-block 逐 32-group nibble 重排，e8m0 字节直存。layer16 bit-exact。
- **P3 NEON kernel**：`ggml_vec_dot_mxfp4_q8_0`（`vqtbl1q_s8`+`vdotq_s32`）+ 行内 +512B 预取 + 双 FMA 累加链
  → 微基准 **2.4×**（0.95→0.40ms/层 @128t）;cosine 0.999939。`kt_llamafile_sgemm` 加 MXFP4×Q8_0 分支。
- **P4 merge 并行**（7×，merge 98→14µs/层）。
- **Q8_0 路径同款优化**（commit `18d25b7`）：`kt_vec_dot_q8_0_q8_0` 自包含变体 2.38×，**ggml 零改动**，
  `KT_Q8_REF=1` 回退。DSv4 生产不走（已用 MXFP4），价值在惠及其他 W8A8→Q8_0 走 CPU offload 的模型。
- **实测**：cpu_moe_wall 55→**median ~22–27ms / min 17ms**;decode 8.5→**~13–16 tok/s**（清净窗口 ~16，
  中等争抢 ~13–14）;DRAM 275→**137 GiB**;F2 四 prompt 连贯。负结果（down nrc=2 / 跨专家预取）已记录回退。
- 完整记录：[mxfp4_cpu_moe_handoff.md](mxfp4_cpu_moe_handoff.md)。**后续 CPU 侧迭代主要基于 MXFP4。**

### 6.8 decode 冷启调查（Session F，2026-06-12）：park 阈值证伪，warmup 采纳

现象：decode 开机第一发慢、连发几个升到稳态。两条假说,实测:

- **❌ worker park 阈值（证伪，别再调）**：`worker_pool.cpp` 的 spin-park 50ms 阈值**不是**冷启因。受控 A/B
  （50ms vs 2000ms，env `KT_PARK_IDLE_MS` 可配，两处都改）：吞吐/首发/抖动**无可测差异**;连续 decode 每 token
  按 43 层逐层 submit、worker 空闲永远 <50ms → **根本不 park**（active CPU 两档都 ~129 核）。拉大阈值唯一净效果是
  **坏处**（sub-2s 间隔活动把 128 核钉死 100%，共享机不友好）。代码已 revert 回基线，无残留。
- **✅ `SKIP_WARMUP=0`（开机预热，采纳）**：去掉启动脚本默认的 `--skip-server-warmup`，开机跑一次 dummy decode
  暖 NPU graph/cache → **开机第一发不再冷**。独立验证（清净窗口 load~21，每臂 boot×2）：基线 req1 **2.08–2.49s** vs
  预热 req1 **1.82–1.84s**，**两臂区间不重叠**;预热把 req1 拉到稳态、消掉 ramp;预热 pass 仅 ~5s，几乎零 boot 代价。
  脚本已 `${SKIP_WARMUP:-1}` 门控（默认 1 保基线，serving 设 0）。**边界**：只治开机冷启，不治会话中途空闲后又掉速
  （那是 per-request 首 token transition + 邻居争抢，需周期性 keep-warm）。
- 完整数据/机制见 [workerpark_tune_handoff.md](workerpark_tune_handoff.md) 的 Closeout + Follow-up。

### 6.9 🔀 G 线合入主干（2026-06-18 merge `a65995b`）：depool + 流式 prefill + AscendC MXFP4 kernel

G 线 `mxfp4-dequant-kernel`（与 dev 在 `4755d7f` 分叉）真三方 merge 进主干。带入：
- **depool（MXFP4 池 model-load 时并行 O_DIRECT 构建）**：在加载阶段并行构建 MXFP4 专家池（overlap 在 load 内，
  **122s** 完成、在 fired up 前），取代老的「第一个请求里懒构建 347s」。主干复测：池 122s、prefill(n=1) 19.56s、
  inline dynamic-resident `top-32×43 share=0.553`、回答正确。
- **⚠️ depool 必须用 MXFP4 CPU GGUF（commit `76304bd` 修复）**：launch 脚本现在 `KT_MXFP4_DEPOOL=1` 时
  `KT_GGUF_TEMPLATE` **默认 `_mxfp4.gguf`**（原默认 Q8_0）。根因：CPU MoE 是带宽 bound，depool 默认走 Q8_0 CPU GGUF
  会让 decode off_cpu **~2×**（22 vs 16.8ms、13 vs **15.5 tok/s**）、CPU 池 6.8 vs 3.4GB/层。改默认 MXFP4 后 **≈ plain**。
  （这正是「merge 后 decode 变慢」调查的真因 = 纯配置；非 plain 回归、非 dynamic-resident。memory `depool-decode-needs-mxfp4-gguf`。）
- **流式 prefill 逐层加载**（`KT_PREFILL_STREAM`）、**AscendC MXFP4 dequant/NZ kernel**（`tools/ascendc_mxfp4/`，
  device→device、ctypes 调，首 import 自动 bisheng 编译）、longseq 调试工具。仅改 `experts_base.py`（Python，**.so 不重编**）。
- **merge 细节**：唯一冲突 `p27_launch_ds4flash_npu.sh`，解为「取 G 的 depool-MXFP4 默认块 + 保 dev 的
  SKIP_WARMUP/custom-op-source/set+eu/param」；sglang 子模块 ff 到 **`8ff3924a9`**（是 dev `456687a0f` 的直接后代）。
- 来龙去脉见随 merge 带入的 [longseq_prefill_handoff.md](longseq_prefill_handoff.md)、
  [mxfp4_dequant_kernel_handoff.md](mxfp4_dequant_kernel_handoff.md)、`tools/ascendc_mxfp4/HANDOVER_SESSION_C.md`。
  **depool 是 opt-in 实验路径，默认生产路径仍是 MXFP4 GGUF（§4.4）。**

> **每个 request 的首 40-token 窗口偏慢（~14 vs 稳态 ~16 tok/s）是正常现象**：prefill→decode transition（首 decode
> token 的 KV 转换 + 首次 graph replay + cache 重热，~0.3–0.5s 一次性）摊进第一个 40-token 日志窗口。每 request 一次，
> 生成越长占比越小。已证与 worker park（§6.8）/ TASK_QUEUE 无关。`SKIP_WARMUP=0` 只治开机第一发，治不了 per-request transition。

---

## 7. 性能数据（参考）

| 项 | 值 |
|---|---|
| Decode 吞吐 — **MXFP4** graph `kt-cpuinfer 128`（06-11） | **~13–16 tok/s**（清净窗口 ~16，中等争抢 ~13–14；min cpu_moe_wall 17ms） |
| Decode 吞吐 — Q8_0 graph `kt-cpuinfer 128`（06-09） | ~8.5 tok/s |
| Decode 吞吐 — eager | ~1.6 tok/s |
| Graph capture | ~6.8s（bs=1，真实权重） |
| 模型加载（MXFP4 138 GiB） | **~2–3.5 min**（热 cache，page cache 全驻;冷盘首启另加磁盘读） |
| 开机第一发（20-tok）— `SKIP_WARMUP=1` 基线 / `=0` 预热 | req1 **2.1–2.5s → 1.8s**（清净窗口 load~21，§6.8;预热消掉冷启 ramp） |
| HBM 占用（静态，N=32，见 §7.1） | **~44.7 GB**（模型 42.32 + KV pool 1.34 + graph 1.08）;拉起后剩 **~15.8 GB** |
| DRAM 占用 | **~137 GiB（MXFP4）** / 275（Q8_0）/ 555（BF16） |

> 共享机 load 长期 ~400 时所有绝对带宽是被邻居挤占后的下限;清净独占聚合 ~442 GB/s。decode 的
> median−min 抖动主体是邻居噪声（G1，max-of-8-NUMA 放大）。

### 7.1 HBM 预算与 prefill/decode 两阶段分布（实测 MEM_FRACTION=0.72 / N=32 / context 65536）

HBM 总可用 **60.50 GB**（64GB − 驱动 ~3.5GB）。拉起日志实测：

| 块 | 大小 | 性质 |
|---|---|---|
| **模型权重** | **42.32 GB** | 静态。常驻 32 专家（MXFP4 NZ，每个 ~20MB）+ MLA attention + shared + emb/lm_head + Lightning Indexer + router。**大头 70%** |
| **KV pool**（预分配） | **~1.34 GB** | 静态。MLA+NSA+SWA 多池，容量 `max_total=172032 token` 却仅 1.3GB → **MLA+NSA 压缩极强，KV/token 极小** |
| **NPU graph buffer** | **1.08 GB** | 静态（decode bs=1 图） |
| **静态小计** | **~44.7 GB** | prefill/decode 都常驻 |
| **拉起后剩余** | **~15.8 GB** | ← prefill/decode 的临时分配全从这出 |

**两阶段临时分配（关键差异）**：

| 临时块 | Prefill | Decode |
|---|---|---|
| activation（中间张量） | **单层 forward 的瞬时峰值**（逐层 buffer 复用，**非 43 层累加**），∝ chunk token 数：attention QKV 投影 / NSA-indexer 打分 / MoE `[tok×2048]` 中间都随 token 数放大 | **极小（M=1，MB 级）** |
| depool 流式 NZ staging slot | ~4 GB（重试时） | 不需要 |
| 实际 KV 填充 | prompt KV（64k 也才 ~0.5GB） | 随生成增长，压缩后 ~MB |
| **临时峰值 vs 剩 15.8GB** | 32k ✅ / **64k 实测峰值 ~14GB（OOM 刻 allocated 58.74 − 静态 44.7 反推）且仍要更多 → OOM → hybrid fallback** | **<1GB → 永不 OOM** |

> ⚠️ 精度：静态 44.7GB / 剩 15.8GB / 64k OOM 刻 allocated 58.74GB 是**实测**；64k 临时峰值 ~14GB 由 allocated−静态
> **反推**；其内部 activation vs 流式 slot 的拆分、及 32k 的 ~7GB **是估算**（∝token 线性外推），未分项实测。
> activation 是**单层瞬时**（层间复用），不是整网累加。

**结论**：
- **瓶颈在 prefill 阶段的 activation + 流式 slot，不是 KV、不是 mem-fraction。** decode 阶段 HBM 极宽裕 → 长上下文 decode 不掉、不 OOM（32k decode 实测 15.3 tok/s ≈ 短上下文）。
- **调 `mem-fraction-static` 无效**（印证 [dsv4-npu-expert-capacity]）：它管 KV pool，而 KV pool 才 1.34GB、没吃满额度，缩它腾不出空间；真正占 HBM 的是 42GB 模型权重。
- **64k prefill 差 ~2GB**：唯一有效杠杆是**降 `KT_NUM_GPU_EXPERTS`**（每常驻专家 ~20MB，32→24 省 ~7GB → 够 64k）；代价 decode 略慢（更多专家落 CPU）。或减小 `CHUNKED_PREFILL_SIZE` 分块降 activation 峰值，但撞 compressor 跨 chunk bug（坑⑬同类，runbook `loc.numel` 报错）。
- **32k 是单卡稳定上限**（验证过，临时 ~11GB < 15.8GB 余量足）。

---

## 8. 关键约束 / 红线

| # | 红线 | 后果 |
|---|---|---|
| R1 | 不上 SVE/BF16/I8MM;march 固定 `armv8.2-a+fp16+dotprod`;kernel 只用 `vqtbl1q_s8`+`vdotq_s32` | SIGILL |
| R2 | C++ pybind 不 `#include <torch_npu/...>` | ABI 不稳 |
| R3 | ACL callback 必须专用 poller 线程 subscribe+process | 卡 sync、NPU 空闲 |
| R4 | **MXFP4 nibble 序：consecutive(ckpt) vs half-block(GGUF) 必须重排**;Q8_0 不可 reinterpret int8 块 | 数值错但不报错 |
| R5 | 不把环境路径硬编码进**代码** | 换环境撞死 |
| R6 | SGLang 不 fork 整模型实现,只加分支/继承 | 升级 submodule 破坏 |
| R7 | **vendored ggml 改动不 commit 进子模块,走 patch 0002** | 合并污染 gitlink |
| R8 | shared_experts / router gate 不 offload,留 NPU | 路由/精度 |
| R9 | `first_k_dense_replace` 层无 256 expert,offload 要 skip（本模型=0） | KeyError |
| R10 | NEXTN(MTP)不开 | sglang NPU NEXTN 有坑且不划算(§6.5) |
| — | 绝不 `pkill -f sglang.launch_server`;按 PID/端口杀,SIGTERM 优雅释放 HBM | 杀别 session、自杀 shell、孤儿占 HBM |

---

## 9. 命令速查

```bash
# 每容器
apt-get install -y libhwloc-dev libhwloc15

# 体检
find kt-kernel -name "kt_kernel_ext*.so"; ls /workspace/models/cache/dsv4_layer*_mxfp4.gguf | wc -l
npu-smi info | head -20

# 编译（必要时）
cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace

# 转 MXFP4 GGUF（现行主路径）+ 全集校验
/usr/local/python3.11.14/bin/python3.11 tools/batch_convert_mxfp4_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --jobs 16 --verify-sample 3
python3 tools/verify_mxfp4_gguf_set.py --dir /workspace/models/cache --sha256-manifest tools/mxfp4_gguf_sha256.txt

# 拉起（MXFP4，graph-on；自己终端前台跑）
NPU_DEVICE_ID=<空闲卡> PORT=8020 \
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  KT_CPUINFER=128 MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh

# MXFP4 kernel 离线对账（cosine 0.9999）
PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel" /usr/local/python3.11.14/bin/python3.11 \
  tools/p27_cpu_moe_reference_check_mxfp4.py --model-dir /workspace/models/DeepSeekV4/DeepSeek-V4-Flash \
  --gguf /workspace/models/cache/dsv4_layer16_mxfp4.gguf --layer-idx 16
```

---

## 附录 A：int8（Q8_0）CPU 权重（旧路径，附录）

> Q8_0 是 MXFP4 之前的 CPU offload 路径（int8，1.0625 B/元素，275 GiB）。**现行生产已换 MXFP4，
> 后续 CPU 迭代不再基于 Q8_0。** 保留此附录供：① 无原生 MXFP4 权重时的回退；② 对照基线;
> ③ 复用 Q8_0 kernel 优化（`KT_Q8_REF`）的其他 W8A8 模型。

**W8A8 → Q8_0 GGUF 转换**（与 MXFP4 不同：这是 dequant→requant 的**再量化**，不是无损 repack）：

```bash
/usr/local/python3.11.14/bin/python3.11 tools/batch_convert_w8a8_layers_mp.py \
  --input /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 --output-dir /workspace/models/cache \
  --layer-start 0 --layer-end 42 --quant q8_0 --jobs 32 --verify-sample 3
# dequant→requant：W_fp32 = int8 * fp32_scale[out_ch]；再按 Q8_0 block（每 32 元 fp16 scale + int8 qs[32]）。
# 输出 dsv4_layer{0..42}.gguf（无 _mxfp4 后缀）;--jobs 32 较优（聚合 ~129/192 核，磁盘 I/O 成瓶颈）。
# 也支持 --quant bf16（dsv4_layer{i}_bf16.gguf，数值基线 cosine 0.999997）。
```

**用 Q8_0 拉起**（不传 `KT_GGUF_TEMPLATE` 即走 Q8_0 默认模板）：

```bash
NPU_DEVICE_ID=<空闲卡> PORT=8000 KT_CPUINFER=128 \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh
# 默认 KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}.gguf'

# eager 回退（仅对照/排障）：竞态已根治(e5f53ad),不再需要 force-sync=1
NPU_DEVICE_ID=<卡> EXTRA_FLAGS="--disable-cuda-graph" bash tools/p27_launch_ds4flash_npu.sh
```

**Q8_0 离线对账**：

```bash
PYTHONPATH="$PWD/third_party/sglang/python:$PWD/kt-kernel" /usr/local/python3.11.14/bin/python3.11 \
  tools/p27_cpu_moe_reference_check.py --w8a8 /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --gguf /workspace/models/cache/dsv4_layer3.gguf --layer-idx 3 --method LLAMAFILE
# Q8_0 cosine 0.9999;BF16 0.999997
```

**历史结论修正**：Spec/Handoff（05-12）的「Q8_0 在 aarch64 会 NaN」「MOE_INT8/KML 必须 BF16」已过时——
实测 Q8_0(int8) CPU offload 可用（坑⑧ 已修，无 i8mm 时回退 `ggml_vec_dot_q8_0_q8_0`）。

**坑⑰（modelslim 权重 + CPU GGUF 基底不一致 → 乱码）**：跑 modelslim checkpoint（`DeepSeek-V4-Flash-w8a8-mtp`，带 **quarot 全局旋转**）单卡时，**绝不能**复用从 native-MXFP4/W8A8 转的老 GGUF——NPU 侧是旋转基底、CPU 老 GGUF 是非旋转基底，224/256 专家算错 → 乱码（曾误判"attention 坏/不同模型"，实为基底不一致，raw 权重 cos≈0.01 是 quarot 旋转所致，非不同模型）。修法：用 `tools/convert_w8a8_to_gguf_q8_0.py` 给该 checkpoint 自己转一份 Q8_0 GGUF（mxfp4 converter 读不了 modelslim int8）。（注:旧版本曾加 `force-sync=1` 稳 greedy,竞态已根治 e5f53ad 后 `force-sync=0` 即对且确定,无需再加。）详见 [`modelslim_quarot_basis_gguf_pitfall.md`](./modelslim_quarot_basis_gguf_pitfall.md)。**换 NPU 侧 checkpoint，CPU GGUF 必须同步换成同一基底的。**

**坑⑱（异步竞态已根治 —— `KT_FORCE_SYNC_SUBMIT=0` 现在又对又快，曾经的"必须开"已过时）**：早先 prefill 的 stream-callback 异步路径有竞态——`ACL_CALLBACK_NO_BLOCK` 的 host callback 非 host 可观测,H2D 偶发在 CPU 写完前读 `output_cpu` → 静默算错（关 force-sync 时 GPQA off 掉到 15%、temp=0 问 15×17 答 "245"）。**已被 commit `e5f53ad` 根治**：prefill 改同步 `cpu_infer.submit` + 无条件 `_wait_device`（等输入 D2H 拷完再读）+ 保留 `subscribe_ascend_stream`（这才是 decode 快的来源,不是异步 submit）。现在 **`KT_FORCE_SYNC_SUBMIT=0` 又对又快**：GPQA off **75.25%（prefix）/ 72.22%（depool）** 都对齐 PR 73.23%;decode 稳态 **16（prefix）/ 18.9（depool）**;与 force-sync 路径逐字对齐、bit 确定（含重 prefill）。`force-sync=1` 现在只是**慢路径**（关掉 NPU/CPU overlap → ~10 tok/s）、**不再是正确性必需**。详见 memory `prefill-async-race-fixed-forcesync-off-fast`、[`accuracy_report.md`](./accuracy_report.md)。

---

*整合自多份来源,以本文为现行总纲。维护:`dsv4_one_card_dev`。现行：CPU MXFP4（§6.7）;
后续 CPU 侧迭代主要基于 MXFP4（热专家放置 / overlap / 长序列流式）。*
