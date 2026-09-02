# `ascend_glm53` —— 脚本参考

GLM-5.3-Flash 在**单个昇腾 910C die** 上服务，路由专家 offload 到主机 DDR。
NPU 侧 INT8 W8A8（注意力、3 个 dense 层、常驻专家），其余专家以 MXFP4 常驻主机，
由 kt-kernel 的 `LLAMAFILE` MoE 计算。

**标准配置是流式 prefill + 动态热专家**（`GLM53_PREFILL_STREAM=1`）。
本文只讲脚本怎么用。为什么是这个形状（容量账、NUMA 悬崖、roofline、流式设计）看
[`DESIGN.md`](./DESIGN.md)；精度怎么建立的、每个数字有多硬看
[`ACCURACY.md`](./ACCURACY.md)；未决项与清理的出口标准看 [`PLAN.md`](./PLAN.md)。

## 脚本

| 脚本 | 作用 |
|---|---|
| `glm53_env.sh` | 环境解析。被其余脚本 source。`bash glm53_env.sh --show` 打印解析出的一切 |
| `setup.sh` | `probe｜submodules｜kt-kernel｜gguf｜check｜all`，每步幂等 |
| `serve.sh` | 起服务。`--foreground` 前台；`GLM53_DRY_RUN=1` 只打印命令不占卡 |
| `verify.sh` | 验收门。`verify.sh chat` 是交互客户端 |
| `serve_fg.sh` | 前台起一个 bs=1 标准配置服务，sglang 打印直接进终端 |
| `ask.sh` | 独立 curl。打印 prompt、完整输出，以及**客户端侧的 decode 速率** |
| `bench.sh` | 带污染门的吞吐测量，双长度相减 |
| `run_ppl.py` | **精度判据**：对运行中的服务做 teacher-forced 困惑度，4096 token 窗口。零噪声地板，见 [`ACCURACY.md`](./ACCURACY.md)。⚠ 用 `"$GLM53_PYTHON" run_ppl.py` 调，不要用 `./`——shebang 会取到 PATH 上的 python，而 transformers 在项目 venv 里 |
| `experiments.sh` / `analyze_profile.py` / `_hot_verdict.py` | 扫描与 profile 分析 |

## 前置条件（setup.sh **不会**替你装的部分）

`setup.sh` 从"依赖"这一步往后全包，但下面四样必须先就位。它们要么是系统级安装、
要么是需要你自己做版本决定的东西，脚本替你选会更糟。

### 1. 硬件

- **昇腾 910C**（A3 / `ascend910_93`）。服务只用**一张 die**，需要它 HBM 空着
  （64 GiB，约 61.3 GiB 可用）；`npu-smi info` 看谁在用。
- 主机内存：非常驻专家运行时约 **134 GiB** 常驻，加 page cache。开发机实测 1.8 TB，
  目标单卡镜像约 200 GB 也放得下。
- 磁盘：三份权重合计约 **630 GB**（见下），其中 GGUF 那 151 GiB 是 `setup.sh gguf` 产出的。
- ⚠ `npu-smi` **分不出 A2 和 A3**，都显示 `Ascend910`。认型号只能用
  `torch.npu.get_device_name(0)`（A3 是 `Ascend910_9362`）。`setup.sh probe` 会打印它。

### 2. CANN toolkit

装在 `$HOME/Ascend`、`/home/developer/Ascend`、`/usr/local/Ascend` 或 `/opt/Ascend`
任一处（`glm53_env.sh` 按这个顺序找；`ASCEND_INSTALL_ROOT` 可覆盖）。

⚠ **包版本和编译器版本可以不一致**，本机就是：`ascend_toolkit_install.info` 报 9.2.0
而 compiler 报 9.1.0。**算子包是按 compiler 版本编的**，所以 `probe` 两个都打印。

### 3. Python 环境 —— 这是唯一真正需要你做决定的一步

⚠ **`torch` 与 `torch_npu` 必须和 CANN 版本对得上。** 这个对应关系由昇腾发布决定，
不在本仓库控制内，装错的表现是 import 就崩或者算子找不到，而不是跑得慢。

**本线实测通过的组合**（不是唯一可行的组合，是有据可依的一组）：

| | 版本 |
|---|---|
| CANN | 9.2.0 包 / 9.1.0 compiler |
| Python | 3.12.9 |
| `torch` | 2.10.0 |
| `torch_npu` | 2.10.0.post4 |
| SoC | `ascend910_93` |

建一个 venv 并在里面装好 `torch` + `torch_npu`，然后把它指给脚本：

```bash
export GLM53_VENV=/path/to/.venv-glm53      # 或直接 GLM53_PYTHON=/path/to/python
```

其余 50 个依赖交给 `./setup.sh deps`——它会**用你已装的 torch 做 constraint**，
所以某个依赖想升级 torch 时会明确失败，而不是把 torch_npu 配套的那个 torch 换掉。
先看清单不安装：`./setup.sh deps --dry-run`。

### 4. 权重

| 变量 | 是什么 | 大小 | 谁产出 |
|---|---|---:|---|
| `GLM53_MODEL_PATH` | INT8 W8A8（compressed-tensors），`--model-path`。die 上的注意力、3 个 dense 层、常驻专家 | ~307 GB | **你准备** |
| `GLM53_MXFP4_CKPT` | MXFP4（`mixed-mxfp4-int8`），GGUF 转换的**源**，服务时不读 | ~170 GB | **你准备** |
| `GLM53_GGUF_DIR` | 逐层 MXFP4 GGUF，CPU MoE 内存映射的那份 | ~151 GB | `setup.sh gguf` |

两份都放在 `$GLM53_MODEL_ROOT`（默认 `/mnt/workspace/models`）下，或各自单独指路径。

⚠ **A3 硬件不支持 fp8**，连分配一个 fp8 张量都会报错——所以厂商的 FP8 发布**不能直接服务**，
必须先转成 INT8。

⚠ **GGUF 少一层不会报错**：kt-kernel 对那层加载零个专家，模型照常回答，只是胡说。
`serve.sh` 和 `setup.sh check` 都会数文件个数，别绕过。

### 一句话核对

```bash
./setup.sh probe     # 逐项打印：CANN 包/编译器版本、SoC、python、各个包、submodule、GGUF 层数
```

缺的会标黄色 `build`。`torch`、`torch_npu` 标 `build` 的话，**先回到第 3 步**——
后面所有步骤都建立在它们之上。

⚠ **`probe` 会短暂打开 NPU device 0**（SoC 探测只能靠 `torch.npu.get_device_name(0)`，
`npu-smi` 分不出 A2/A3）。共用机器上 die 0 是别人的时候，先
`export GLM53_SOC=ascend910_93` 跳过探测。

## 从零到跑起来

```bash
cd kt-kernel/tools/ascend_glm53

# 前置条件（CANN / venv / torch+torch_npu / 两份权重）见上一节
bash glm53_env.sh --show      # 先核对：CANN、SoC、权重路径、NUMA、常驻专家数
./setup.sh probe              # 这个镜像已经有什么、还缺什么
./setup.sh all                # 装依赖 → 建算子 → 建 kt-kernel → 转权重 → preflight
GLM53_PREFILL_STREAM=1 ./serve.sh
./verify.sh                   # 全部 PASS 才算起来了
```

`setup.sh` 的步骤（每步幂等，都能单独跑）：

| 步骤 | 做什么 | 镜像已提供时 |
|---|---|---|
| `submodules` | llama.cpp / pybind11 | — |
| `deps` | 按 sglang 的 `pyproject_npu.toml` 装依赖，**用已装的 torch 做 constraint**，防止被静默升级 | `--dry-run` 可先看 |
| `sgl-kernel` | 构建 `sgl_kernel_npu` / `deep_ep` / `attentions`（tag 钉死） | 跳过 |
| `cann-ops` | 构建 CANN 自定义算子：`customize`（mHC、量化 swiglu/routing）、`custom_ops` torch 绑定、`custom_transformer`（DSA 的 compressor / quant_lightning_indexer / sparse_attn_sharedkv），两个仓库按 commit 钉死 | 跳过 |
| `kt-kernel` | 编 wheel 并安装 | — |
| `gguf` | MXFP4 → 42 层 GGUF，逐位验证 | — |
| `check` | preflight | — |

⚠ **前四样（CANN、venv、torch/torch_npu、两份权重）不在这张表里**，见上面的
「前置条件」——那是这套脚本唯一不替你做的部分。

`setup.sh all` 里最慢的是 `gguf`（42 层，约 15 分钟，产出 151 GiB）。
`sgl-kernel` 和 `cann-ops` 在镜像已提供时是秒回；真要从零构建它们要几十分钟。

交互式看吞吐：

```bash
./serve_fg.sh     # 终端 A，等 "The server is fired up and ready to roll!"
./ask.sh          # 终端 B
```

停服务 —— **只按端口杀，方括号防止匹配到自己，这台机器是共用账号**：

```bash
pkill -f -- "[-]-port ${GLM53_PORT:-30013}"
```

## 常用配置

在 source 之前 export 即可覆盖；`glm53_env.sh --show` 打印生效值。

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_NPU_DEVICE_ID` | `0` | 用哪个 die，**Phy-ID**（0..15）。用前先 `npu-smi info` |
| `GLM53_PORT` | `30013` | 别和同机其他人撞 |
| `GLM53_PREFILL_STREAM` | `0` | **标准配置置 1**。会改 `MEM_FRACTION` 与 `MAX_TOTAL_TOKENS` 的默认值 |
| `GLM53_NUM_GPU_EXPERTS` | `32` | 每层常驻专家数。die 上权重 = `15.60 + 6.75×[流式] + 0.9925×N` GiB（实测拟合，残差 0.13）|
| `GLM53_MEM_FRACTION` | `0.85`／流式 `0.95` | 流式槽位在 KV 定容**之前**预留，0.85 下调度器直接拒绝启动 |
| `GLM53_MAX_TOTAL_TOKENS` | 不设／流式 `40960` | KV 上限，见下「KV 与并发要一起定」 |
| `GLM53_CONTEXT_LENGTH` | `32768` | |
| `GLM53_CHUNKED_PREFILL_SIZE` | 流式 `6144`／hybrid `8192` | **必须是正数且 128 的倍数**。流式路径下 `8192` 会 OOM，见下 |
| `GLM53_MAX_RUNNING_REQUESTS` | `1` | 已在 4 / 8 下跑过，但**必须与 KV 一起定** |
| `GLM53_KT_NUMA_NODES` | 无 | 权重放在哪几个 NUMA node，如 `0,1`。**只写同一对**，见下 |
| `GLM53_PIN_CORES` | 无 | taskset 核表，如 `0-79`。⚠ 只在 node 0,1 与 2,3 上验证过 |
| `GLM53_THREADPOOL_COUNT` | NUMA 节点数 | ⚠ 默认值在多 NUMA 机器上是错的，见下 |
| `GLM53_CPUINFER` | NUMA×16，上限 `nproc×3/4` | `--kt-cpuinfer` |
| `GLM53_GGUF_DIR` | `$GLM53_MODEL_ROOT/GLM-5.3-Flash-MXFP4-gguf` | 逐层 GGUF |
| `GLM53_DRY_RUN` | `0` | `1` 只打印 `serve.sh` 要跑的命令 |
| `GLM53_EAGER` | `0` | `1` 关图。⚠ 图模式在这条路上值约 5× 解码吞吐 |

## 全部变量

上面那张表是**你通常会设的**。这里是脚本实际读的全部 70 个，按用途分组——
之前只有 17 个进了文档，而其中四个 `SGLANG_OPT_*` 是**精度开关**。

### 路径发现（都能覆盖，都有推导出的默认值）

| 变量 | 默认 | 说明 |
|---|---|---|
| `KTRANSFORMERS_REPO` | 由脚本位置推导 | 仓库根 |
| `SGLANG_REPO` | `$KTRANSFORMERS_REPO/third_party/sglang`，其次同级 checkout | **必须**解析到它里面，否则 `serve.sh` 拒绝启动 |
| `GLM53_WORKSPACE` | `dirname $KTRANSFORMERS_REPO` | |
| `GLM53_ENV_ROOT` | `$GLM53_WORKSPACE/env`，其次 `$GLM53_WORKSPACE/*/env` | venv、算子包、语料所在的同级目录。**仓库里不放任何人的家目录路径** |
| `GLM53_VENV` / `GLM53_PYTHON` | `$GLM53_ENV_ROOT/.venv-glm53` → `python3.12/3.11/3` | |
| `GLM53_EVAL_DIR` | `$GLM53_ENV_ROOT/eval` | 真实散文语料（`wikitext/test.parquet`），`ask.sh` 与 `bench.sh` 用 |
| `GLM53_ARTIFACT_ROOT` | `/var/tmp/glm53` | 一切产出的根；日志与构建树都由它推导 |
| `GLM53_LOG_DIR` | `$GLM53_ARTIFACT_ROOT/logs` | `verify.sh` 与 `bench.sh` 都读这里的 `serve.log` |
| `GLM53_BUILD_ROOT` | `$GLM53_ARTIFACT_ROOT/ktbuild` | 让 22 MB 的 CMake 树离开近满的 `/mnt/workspace` |
| `ASCEND_INSTALL_ROOT` | 依次找 `$HOME/Ascend`、`/home/developer/Ascend`、`/usr/local/Ascend`、`/opt/Ascend` | ⚠ `/home/developer` 是容器镜像的安装位置，不是谁的家目录 |
| `GLM53_OPP_CUSTOM_DIRS` | `$CANN_ROOT/opp/vendors:$GLM53_ENV_ROOT/opp_custom/vendors` | 冒号分隔，先命中先用 |
| `GLM53_TOOLS` | 脚本所在目录 | |
| `GLM53_GIT_PROXY` | 空（直连） | 只在 `setup.sh submodules` 里用；网络需要时才设 |

### 权重与层范围

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_MODEL_ROOT` | `/mnt/workspace/models` | 下面三个的基准 |
| `GLM53_MODEL_PATH` | `$GLM53_MODEL_ROOT/GLM-5.3-Flash-W8A8` | `--model-path`，die 上的那份 |
| `GLM53_MXFP4_CKPT` | `$GLM53_MODEL_ROOT/GLM-5.3-Flash-MXFP4` | GGUF 转换的**源**，服务时不读 |
| `GLM53_GGUF_NAME_PREFIX` / `GLM53_GGUF_NAME_SUFFIX` | `glm53_layer` / `_mxfp4` | |
| `GLM53_GGUF_TEMPLATE` | 由上面拼出 | ⚠ 含 `{layer_idx}`，**不能**写进 `${VAR:-default}` |
| `GLM53_MOE_LAYER_START` / `GLM53_MOE_LAYER_END` | `3` / `44` | 层 0-2 是 dense，层 45 是 MTP 不转 |

### 服务

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM53_HOST` | `127.0.0.1` | |
| `GLM53_SIDE_STREAM` | `1` | 传给 `KT_SIDE_STREAM`；**与流式 prefill 无关**，每个 decode step 都走 |
| `GLM53_ALLOW_SINGLE_SUBPOOL` | `0` | `1` 关掉"线程池只有 1 个而机器有多个 NUMA 节点"的告警 |
| `GLM53_EXTRA_FLAGS` | 空 | 原样追加到 `launch_server` 后面 |
| `GLM53_SOC` | 空 → 探测 | ⚠ 探测会调 `torch.npu.get_device_name(0)`，**会打开 die 0** |
| `GLM53_JOBS` | `min(nproc, 32)` | 构建并行度 |
| `GLM53_FORCE_KT_KERNEL` | `0` | `1` 强制重建 kt-kernel wheel |
| `GLM53_DERIVED_FOR_STREAM` | 内部 | 记录派生值出自哪个分支，翻转 `GLM53_PREFILL_STREAM` 时重新推导 |

### `KT_*` 透传（`serve.sh` 设，kt-kernel / sglang 读）

⚠ **这些在模块 import 时读取并冻结成全局**，起服务之后改无效。

| 变量 | 默认 | 说明 |
|---|---|---|
| `KT_SIDE_STREAM` | `1` | 见 `GLM53_SIDE_STREAM` |
| `KT_NUMA_NODES` | 由 `GLM53_KT_NUMA_NODES` 设 | CPU MoE 子池放在哪几个 node |
| `KT_PREFILL_STREAM` | 由 `GLM53_PREFILL_STREAM` 设 | |
| `KT_PREFILL_STREAM_THRESHOLD` | `512` | 超过这么多 token 的 chunk 才走流式 |
| `KT_PREFILL_STREAM_CKPT` | `$GLM53_MODEL_PATH` | W8A8 流式读的 checkpoint |
| `KT_MXFP4_CKPT` | `$GLM53_MXFP4_CKPT` | depool 路径必需 |
| `KT_MXFP4_OP_DIR` | 算子包目录 | depool 路径必需 |
| `KT_MXFP4_NZ_CHUNK` | `32`／`serve.sh` 置 `16` | MXFP4→W8A8-NZ 转换的每块专家数。32 时瞬时约 3 GB，和 KDA 的 Triton workspace 抢同一份显存 |
| `KT_MXFP4_GGUF_DEDUP` | **`1`** | 标准配置。不建 safetensors 池，逐层从 CPU MoE 的 GGUF 现读。⚠ 模块内的默认值是 `0`，但 `serve.sh:184` 置 `1`——**看这一列，不要看模块默认** |
| `KT_GGUF_TEMPLATE` | `$GLM53_GGUF_TEMPLATE` | dedup 路径必需 |
| `KT_DYNAMIC_RESIDENT` | `1`（流式时） | 动态热专家 |
| `KT_HOT_TAIL_TOKENS` | `512`（流式时） | 只用 prompt 最后 N 个 token 统计热专家 |
| `KT_STREAM_WARMUP` | `0` | |

### ⚠ `SGLANG_*` —— 这四个是精度开关，不是性能开关

`serve.sh:50-52` 自己写着"每一个都是照着一次测量选的，改一个就静默改变精度基线的含义"。
它们全是 `${X:-...}`，**调用者环境里一个陈旧值就能悄悄覆盖**，而此前任何文档都没提过。

| 变量 | 默认 |
|---|---|
| `SGLANG_OPT_BF16_FP32_GEMM_ALGO` / `SGLANG_OPT_DEEPGEMM_HC_PRENORM` / `SGLANG_OPT_FP8_WO_A_GEMM` / `SGLANG_OPT_USE_FUSED_HASH_TOPK` | 见 `serve.sh` |
| `SGLANG_MAMBA_CONV_DTYPE` / `SGLANG_SET_CPU_AFFINITY` | 见 `serve.sh` |

### 测量工具

| 变量 | 默认 | 说明 |
|---|---|---|
| `NAME` | `run` | `bench.sh` 结果 JSON 的名字 |
| `DIE_IDLE_MIB` | `6553` | 判定 die 空闲的 HBM 阈值（约 65536 的 10%）|
| `LOAD_MAX` | `8` | load1 门限 |
| `DIE_WAIT` / `UP_WAIT` | `600` / `900` | 等 die 回落 / 等服务起来的秒数 |
| `BENCH_FORCE` | `0` | ⛔ 绕过污染门。**拒绝时不要用它**，见测量纪律 |
| `BENCH_PROMPT_TOKENS` | 由 items 推算 | |
| `BENCH_PROMPT_SYNTHETIC` | `0` | ⚠ `1` 恢复旧的重复填充文本，**只用于和历史数据对齐**，它会虚高常驻命中率 |
| `BENCH_PROMPT_ITEMS` | `60` | 只在合成分支下有意义 |
| `RESIDENCY_ORDER` | `32 8 16 32 40 32` | `experiments.sh residency` 的扫描序（首尾重复是 A/A 对照）|
| `NTOK` / `PROMPT_TOKENS` / `FULL` | `256` / `1024` / `0` | `ask.sh` |

## 会咬人的几条

### 配置

- **KV 与并发要一起定，定错不会干净地失败。** 流式路径默认把 KV 封在 40960 token，
  那是给 `--max-running-requests 1` 选的。宽度 8 时每请求只剩 5120 token，
  而实测最长的一条回答是 5328——池打满后**不是报「KV 不足」，是 vector core 异常
  （error 507035）直接把调度器打死**。反方向也有硬边界。实测区间：

  | `GLM53_MAX_TOTAL_TOKENS` | 结果 |
  |---|---|
  | 40960 | ✅ |
  | 98304（1.30 GB KV，余 ~3.7 GB）| ✅ 跑过 40 分钟 |
  | 212992 | ❌ SGLang 夹到 192704，可用显存 4.51→2.40 GB，forward 里 SIGABRT |

  `serve.sh` 会在启动时算 `KV/并发` 并在低于 4096/请求时告警。
  **不要靠降 `max_tokens` 去凑**——截断的回答没有答案行会被判错，
  而它偏向压制思考更长的那一臂。

- ⛔ **`--disable-shared-experts-fusion` 不是优化开关，是不加就起不来。**
  融合会让 `FusedMoE` 变成 289 宽（`glm5_next.py` 的
  `n_routed_experts + num_fused_shared_experts`），而 `kt_expert_masks` 按 288 建表，
  在**模型加载时**撞上 `kt_ep_wrapper.py` 的宽度守卫直接失败。
  这是响亮的失败不是静默降级——守卫的报错里就写着修法。
  ⚠ SGLang 有一张「自动置位这个 flag」的表（给 FlashInfer CuteDSL / TRTLLM / MoE-A2A），
  **KT wrapper 不在里面**。把它加进去让 `--kt-*` 隐含这个 flag 才是正解，但那要动上游文件。
- ⛔ **流式路径下 chunk 不能是 8192，默认已降到 6144。** chunk 大小和常驻专家数
  花的是同一份显存：K=32 的流式配置加载后只剩约 5.5 GiB，而 KDA 层在一次多千 token 的
  prefill 里要向 Triton 要一块 ~3 GiB 的 workspace。实测（die 8，K=32，其余完全相同）：

  | `GLM53_CHUNKED_PREFILL_SIZE` | 结果 |
  |---|---|
  | 8192 | ❌ `NPU out of memory. Tried to allocate 3.00 GiB ... 3.03 GiB free`，在 `eager_runner._execute_extend` 里 SIGABRT。复现两次，其中一次是全新起的服务，所以不是碎片 |
  | 6144 | ✅ `verify.sh` 在它那个 14379 token 的 prompt 上全部 PASS |

  ⚠ **这道坎在 4096 token 的窗口上看不见**（单 chunk），所以困惑度那条线一路是绿的，
  是验收门先撞上的。**别用 PPL 去证明长 prompt 能跑。**
  ⚠ 调大 chunk 是 TTFT 最大的杠杆（TTFT 主项是 chunk 数），但要先在别处腾出显存。

- **`--chunked-prefill-size` 绝不能是 `-1`。** kt-kernel 的 `LLAMAFILE` MoE 按最大 chunk
  长度分配 fp32 输出缓冲，`-1` 会塌成 1，任何超过一个 token 的 prefill 都写越界，
  在 glibc 里 abort。
- **代理会劫持 `127.0.0.1`。** `glm53_env.sh` 已经 unset。手动 curl 拿到 502/503 时
  先看 `env | grep -i proxy`。
- **GGUF 少一层不会报错。** kt-kernel 对那层加载零个专家，模型照常回答，只是胡说。
  `serve.sh` 和 `setup.sh check` 都会数文件个数，别绕过。
- **`{layer_idx}` 不能写在 `${VAR:-default}` 里**——右花括号会提前终止参数展开，悄悄弄坏路径。

### NUMA 与绑核

- ⛔ **8 个 NUMA node 两两配对成 4 个快速域：(0,1) (2,3) (4,5) (6,7)。**
  对内 ~150 GB/s，**跨对塌到 ~20 GB/s（7.4×）**。
  `/sys/devices/system/node/*/distance` 报均匀的 `10 20 20 ...`，**它是错的，且不会警告你**。
  `GLM53_KT_NUMA_NODES` 只写同一对内的节点。
- ⚠ **跨对读长得和「带宽饱和」一模一样**：4-8 线程就饱和，加到 40 线程完全不动。
  一条漂亮的饱和曲线可能只是页放错了地方。**先查页放置，再谈带宽。**
- ⚠ **`GLM53_THREADPOOL_COUNT` 的默认值（NUMA 节点数）在这台机器上是错的**：
  8 个子池会把权重摊到全部 4 对，必然跨对。单实例请用 `2`（一对）配 `GLM53_CPUINFER=32`。
- ⛔ **sglang 和 kt 线程池自己不绑核**——所有线程的亲和性掩码都是全部核。
  单实例靠内核 autonuma 能收敛；**两个实例同时跑会互相把线程赶到对方的 pair 上**，
  实测从 24.7+22.8 塌到 6.9+6.2 tok/s。并行请用 `GLM53_PIN_CORES`。
- ⛔ **绑核必须在启动时生效，事后 taskset 没用**——页放置在权重加载期由 first-touch 定死。
- ⛔ **`GLM53_PIN_CORES` 在核编号不从 0 开始的 node 上会让 scheduler 起不来。**
  核枚举的缺陷本来就在（`Core N inside NUMA node 4 not found` 早在加这个变量之前就会刷，
  当时只是绑核部分失败、不影响数值）；**用 taskset 把可见核集限制到非零起始区间后，
  它从警告变成致命**：
  绑到 node 6,7（核 240-319）会刷屏 `Core 7 inside NUMA node 6 not found` 然后
  `Rank 0 scheduler died during initialization (exit code: -6)`。
  **已验证可用：node 0,1（核 0-79）与 2,3（核 80-159）。node 4-7 请留空。** 未修。
- ⚠ **GGUF 的 page cache 全机器只有一份，且散在全部 8 个 node 上**（实测某实例 302 GiB
  这样分布）。它在任何绑核之前就由 first-touch 定死，**绑核修不了，杀进程也不会让它重排**。
  影响流式 prefill 反复重读 checkpoint 那条路（TTFT），decode 读的是 depool 后的本地副本。

### 环境

- **`import custom_ops` 单独会失败**（`libc10.so`），必须先 `import torch`。
  算子来自 `ASCEND_CUSTOM_OPP_PATH` 的 vendor 包，那个 wheel 缺席也可能一切正常——
  要查就查 `torch.ops.custom.<op>`。
- **`npu-smi` 分不出 A2 和 A3**，都显示 `Ascend910`。认型号只能用
  `torch.npu.get_device_name(0)`（A3 是 `Ascend910_9362`）。
- **开发机不是目标机。** 这台 320 核 / 8 NUMA / 1.8 TB，目标单卡镜像是 ~40 核 / 1 NUMA /
  ~200 GB。CPU MoE 是主机内存带宽瓶颈，**在这台机器上量到的吞吐不是目标机的吞吐**。

## 测量纪律

这台机器是共用的，几条规矩是拿时间换来的：

- ⛔ **绝不 `pkill -f sglang`**——多人共用一个 OS 账号。只按端口杀，并加方括号
  （`pkill -f -- "[-]-port 30013"`）防止匹配到发起命令的那个 shell 自己。
- ⛔ **不要把 pkill 和它可能匹配到的命令放在同一次 shell 调用里。**
- ⛔ **脚本要给自己起的服务加 `trap ... EXIT INT TERM`。** 杀掉 wrapper 而留下服务，
  下一个跑的人会拿到 `NPU out of memory ... 166 MiB free`——**报错指向受害者，不是肇事者**。
- ⛔ **`wall/generated_tokens` 不是 decode 速率**，它把 TTFT 折进去了，而流式路径的
  TTFT 是几十秒量级。用双长度相减（`bench.sh` 的做法），或者读服务端
  `Decode batch ... gen throughput` 那一行，或者用 `ask.sh` 从第二个 token 起算。
- ⛔ **邻居作业会产出「看起来很干净」的错数据。** `bench.sh` 有三道门（foreign 进程、
  die 独占、前后 load 对比），拒绝时不要用 `BENCH_FORCE=1` 绕过。
  load1 是滞后指标，机器真空闲时可以只放宽 `LOAD_MAX`，但 foreign 门要留着。
- ⛔ **单点不是测量。** 同配置重复跑实测离散 5.7%，所以任何 A/B 比值都需要多点，
  且两臂要**交错**（A B A B），让机器的慢漂移平摊而不是全压在后跑的那一臂。
- ⚠ **benchmark 的 prompt 要用真实散文。** 重复填充文本路由到的专家少得多，
  会虚高常驻命中率——曾把一个 1.15× 的加速报成 1.27×。`bench.sh` 与 `ask.sh` 用 wikitext。

## 流式 prefill 与动态热专家

超过 `KT_PREFILL_STREAM_THRESHOLD`（默认 512 token）的 prefill chunk 不走 hybrid：
整层 288 个专家从 GGUF 流进一个复用的 HBM 槽，MoE 全部在 NPU 上算，不做 CPU 往返。
顺带把**动态热专家常驻**折进去——每层按这次 prefill 的实际激活取 top-K 换掉静态 prefix。

⚠ **`kt_stream_prefill.py` 里每个异常都被吞掉并回退 hybrid，所以坏掉的流式路径
从外面看和好的一模一样。** 唯一的证据是日志，且必须在**超过阈值的 prompt** 上看：

```bash
grep -c 'inline resident' $GLM53_LOG_DIR/serve.log            # 必须 > 0
grep -cE 'streaming failed|hybrid fallback' .../serve.log      # 必须 == 0
```

`verify.sh` 在 `GLM53_PREFILL_STREAM=1` 时会自动查这两条。

⚠ **阈值决定哪些负载真的用到它，而它对短 prompt 是负收益。**
TTFT 交叉点实测在 ~1100 token：14650 token 时流式是 5.1× 收益，
但 630 token 时反而更慢（18.3 s vs 11.4 s）。因为流式**每个 chunk 都要重流全部 288 个
专家**，所以 `TTFT ≈ chunks × 18.0 s + 0.49 ms/token`——**主项是 chunk 数，不是 token 数**。
阈值留给业务按自己的 prompt 分布配置。

⚠ **常见的短 prompt 评测集都够不到这条路径**（GSM8K 60-120、MMLU 5-shot 中位数 408、
GPQA 中位数 92），所以它们量到的是 hybrid 路径。能覆盖的是 4096-token 窗口的困惑度。

⚠ **混合负载下有个没测过的状态依赖**：短 prompt 不触发流式，因此**不重置常驻集**，
一个短请求会继承前一个长请求装好的热专家。

## 和 DeepSeek-V4 配方的差异

同源，但这几条不能照抄：

| | DeepSeek-V4-Flash | GLM-5.3-Flash |
|---|---|---|
| `--page-size` | 128 | **64**（DSA pool 有 `assert page_size == 64`）|
| `--attention-backend` | `ascend` | **不传**，GLM 自己选 KDA / DSA |
| MoE 层 | 0..42（`first_k_dense_replace=0`）| **3..44**，另有层 45 是 MTP，不转 |
| 专家 / top-k | 256 / 6 | **288 / 8** |
| CPU 侧张量名 | `layers.{L}.ffn.experts.{i}.{w1,w3,w2}.{weight,scale}` | `model.language_model.layers.{L}.mlp.experts.{i}.{gate,up,down}_proj.{weight_packed,weight_scale}` |
| 自定义算子 | 需自建 NSA 算子包 | **复用本线已有的 `opp_custom` vendor 包** |
| shared expert | 本就不融合 | 本分支会融合，**必须显式关掉，否则模型加载即失败** |
