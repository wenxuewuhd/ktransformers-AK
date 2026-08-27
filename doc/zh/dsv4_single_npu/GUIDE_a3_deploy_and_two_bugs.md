# DeepSeek-V4-Flash 单 NPU 部署与两个正确性 bug

> 本文提到的补丁、脚本、日志等产物在复现工作区 `/mnt/workspace/dsv4-repro-2159/`，未随本文档入库。

Ascend Atlas A3 单 die ｜ Pan-Boyi/sglang 的 `dsv4-cann9-no-patch` 分支 ｜ 2026-08-24 ~ 08-26

本文档包含两部分：

- **第 1 章**：从一台干净机器开始，到把服务拉起来并验证的完整步骤
- **第 2–4 章**：过程中发现的两个正确性 bug（定位、修复、实测），以及性能与精度数据

---

# 1. 从零搭建到拉起服务

## 1.1 前提

| | 本次实测的配置 |
|---|---|
| NPU | Atlas A3 单 die，`Ascend910_9362`，61.3 GB HBM |
| Host CPU | 鲲鹏 40 核 / 1 NUMA / 229 GB RAM |
| CANN | 9.0.0（`ASCEND_TOOLKIT_HOME=/home/developer/Ascend/cann-9.0.0`） |
| Python | 3.11.4 |
| torch | 2.9.1+cpu |
| torch_npu | 2.9.1 |
| kt-kernel | 0.7.0 |
| 编译器 | **g++-13**（默认的 g++ 9.4 不行，见 §1.7） |
| 磁盘 | 模型约 340 GB + GGUF 约 137 GB |

**两个代码仓，缺一不可：**

| 仓 | 分支 | 作用 |
|---|---|---|
| `Pan-Boyi/sglang` | `dsv4-cann9-no-patch` | 推理框架，NPU 集成 + KT offload |
| `kvcache-ai/ktransformers` | **PR #2157 + PR #2159 的合并** | kt-kernel（CPU MoE）+ 部署脚本 |

> **注意**：ktransformers 的 `main` 上没有 Ascend 后端。裸 clone main 会在 kt-kernel 构建阶段
> 因缺少 `cpu_backend/vendors/ascend_npu.h`、`ascend_callback_worker.*`、`CPUINFER_USE_ASCEND_NPU`、
> `MXFP4=39` 枚举、`tools/ascendc_mxfp4/` 而失败。这些全在**代码 PR #2157** 里；
> **#2159 只有文档和脚本**。教程本身察觉到了这个缺口（它警告 `ascend_npu.h` 可能不在你的
> checkout 里），但既让读者 clone main，又没说去哪儿取 —— 这是个真实的文档断链。

## 1.2 目录布局

全部东西放在一个工作区里，与机器上其它项目完全隔离：

```
/mnt/workspace/dsv4-repro-2159/          # $DSV4_WORKSPACE
├── venv/                                # 独立 venv
├── sglang/                              # Pan-Boyi/sglang
├── ktransformers/                       # #2157 + #2159 合并树
├── cann-recipes-infer/                  # 教程与脚本来源
├── opp/ ops-transformer/ sgl-kernel-npu/  # 自定义算子，私有 opp 前缀
├── dsv4-artifacts/                      # 构建产物（wheel 等）
├── dsv4-logs/                           # 全部运行日志
└── dsv4.env                             # 环境入口
```

## 1.3 代理规则（这台机器上很重要）

本机导出了 `HTTP_PROXY` 但 `no_proxy` 没设全，**python urllib 连 127.0.0.1 都会走代理然后挂死**。

- **只有 github 需要走代理**：`export https_proxy=http://127.0.0.1:1056`
- **其余一律直连**，国内镜像比走代理快约 58 倍
- **访问本机端口必须绕开代理**：`export no_proxy='*'`，或脚本里用 `ProxyHandler({})`

## 1.4 Step 0 —— venv

```bash
export DSV4_WORKSPACE=/mnt/workspace/dsv4-repro-2159
mkdir -p "$DSV4_WORKSPACE" && cd "$DSV4_WORKSPACE"

python3.11 -m venv venv
./venv/bin/pip install -U pip
```

用 venv 不用 conda：这台机器上已有一个同类项目的环境，venv 的隔离足够，且不引入 conda 的 solver 开销。

## 1.5 Step 1 —— 拉代码

```bash
cd "$DSV4_WORKSPACE"
export https_proxy=http://127.0.0.1:1056 http_proxy=http://127.0.0.1:1056   # 只有 clone 需要

git clone -b dsv4-cann9-no-patch https://github.com/Pan-Boyi/sglang.git
git clone https://github.com/kvcache-ai/ktransformers.git
git clone https://gitcode.com/cann/cann-recipes-infer.git       # 教程来源

unset https_proxy http_proxy                                    # 之后一律直连
export no_proxy='*'
```

**合并 #2157 和 #2159**（#2159 依赖 #2157）：

```bash
cd "$DSV4_WORKSPACE/ktransformers"
git fetch origin pull/2157/head:pr2157
git fetch origin pull/2159/head:pr2159
git checkout -b repro-2157-2159 pr2157
git merge pr2159            # #2159 是文档+脚本，冲突极少
```

## 1.6 Step 2 —— 环境入口 `dsv4.env`

```bash
cat > "$DSV4_WORKSPACE/dsv4.env" <<'EOF'
export DSV4_WORKSPACE=/mnt/workspace/dsv4-repro-2159
export DSV4_MODEL_ROOT=/mnt/workspace/models
export DSV4_NPU_DEVICE_ID=0
export DSV4_PORT=18080
export KTRANSFORMERS_REPO=$DSV4_WORKSPACE/ktransformers
export DSV4_TOOLS=$KTRANSFORMERS_REPO/kt-kernel/tools/ascend_dsv4
export DSV4_ARTIFACT_DIR=$DSV4_WORKSPACE/dsv4-artifacts
export DSV4_LOG_DIR=$DSV4_WORKSPACE/dsv4-logs
export CANN_VENDORS_DIR=$DSV4_WORKSPACE/opp/vendors     # 私有 opp 前缀，不污染系统 CANN
export DSV4_PYTHON=$DSV4_WORKSPACE/venv/bin/python
export SGLANG_REPO=$DSV4_WORKSPACE/sglang
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy='*'
export PYTHONPATH="${PYTHONPATH%:}"
EOF

source "$DSV4_WORKSPACE/dsv4.env"
```

`CANN_VENDORS_DIR` 指向工作区内的私有 `opp/vendors`，自定义算子装在这里，**不动系统 CANN**。

## 1.7 Step 3 —— 构建

构建工具在 `$DSV4_TOOLS/setup.sh`，分步或一把梭：

```bash
bash "$DSV4_TOOLS/setup.sh" all        # deps -> kt-kernel -> sgl-kernel -> cann-ops -> gguf -> check
```

各步的含义：

| 子命令 | 做什么 |
|---|---|
| `deps` | 装 SGLang 的 NPU 运行时依赖 |
| `kt-kernel` | 构建 kt-kernel，产出 wheel |
| `sgl-kernel` | 构建 `sgl_kernel_npu`、`deep_ep`、`attentions`、`torch_memory_saver` |
| `cann-ops` | 构建 customize / custom_ops / custom_transformer 三个算子包 |
| `gguf` | 把 checkpoint 转成逐层 MXFP4 GGUF |
| `check` | 环境自检，退出码 0 表示可以启动 |

### ⚠️ 坑 1：kt-kernel 的 CMakeLists 写死了 `/usr/bin/g++`

`kt-kernel/CMakeLists.txt` 用 `CACHE ... FORCE` 强制 `/usr/bin/gcc`/`g++`。
**`CC`/`CXX` 环境变量、`PATH` 前置、`-DCMAKE_CXX_COMPILER` 三种覆盖方式全部无效**（都实测过）。

本机默认 g++ 是 9.4，而 `cpu_backend/worker_pool.h` 用了 C++20 的 `<barrier>`（libstdc++ 从 GCC 11 才提供），
所以**不改 CMakeLists 根本编不过**。

修法（等同上游 `pr2157-a3-adapt` 的 commit `45ee748`）：

```cmake
    # Honor an explicit CC/CXX before falling back to /usr/bin/gcc.
    if(DEFINED ENV{CC} AND DEFINED ENV{CXX})
        set(CMAKE_C_COMPILER   "$ENV{CC}"  CACHE FILEPATH "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE FILEPATH "C++ compiler" FORCE)
        message(STATUS "Honoring explicit CC=$ENV{CC} CXX=$ENV{CXX}")
    elseif(EXISTS "/usr/bin/gcc" AND EXISTS "/usr/bin/g++")
        ...
```

然后：

```bash
sudo apt install g++-13 gcc-13          # 若未装
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
bash "$DSV4_TOOLS/setup.sh" kt-kernel
```

**这条应报上游** —— 任何非默认 toolchain 的机器都会撞上。

### ⚠️ 坑 2：`verify.sh` 的 gate 5 恒失败

`kt-kernel/tools/ascend_dsv4/verify.sh` 里 gate 5 的 `grep -c` 用法有误，导致该项永远不通过。
本地已修，与上面的 CMakeLists 修复同在一个 commit（`1c913ae`）。**这条也应报上游。**

### 建议：开 ARM native

默认构建未开 ARM native，decode 有约 4% 的残余差距：

```bash
export DSV4_ARM_NATIVE=1
```

## 1.8 Step 4 —— 权重

两份权重：

| 用途 | 路径 | 格式 |
|---|---|---|
| NPU 侧（attention + dense + 常驻专家） | `$DSV4_MODEL_ROOT/DeepSeek-V4-Flash-W8A8/` | compressed-tensors **W8A8** |
| CPU 侧（卸载的 routed 专家） | `$DSV4_MODEL_ROOT/cache/dsv4_layer{L}_mxfp4.gguf` | 逐层 **MXFP4 GGUF**，43 个文件 |

GGUF 由 `setup.sh gguf` 生成。若已有现成的，可跳过转换 —— 但**建议做一次校验**，
本次我们做了三级校验（L1 结构 / L2 张量统计 / L3 对第 16 层逐元素反量化比对原始 checkpoint），
全部通过，确认权重可直接复用。

## 1.9 Step 5 —— 拉起服务

```bash
source "$DSV4_WORKSPACE/dsv4.env"
export DSV4_CPUINFER=32                  # CPU MoE 线程数
export DSV4_PREFILL_STREAM=0             # 见下方说明
bash "$DSV4_TOOLS/serve.sh"
```

日志在 `$DSV4_LOG_DIR/serve.log`，pid 在 `serve.log.pid`。启动约 90 秒。

实际下发的命令行（供参考）：

```
python -m sglang.launch_server
  --model-path $DSV4_MODEL_ROOT/DeepSeek-V4-Flash-W8A8
  --device npu --attention-backend ascend
  --tensor-parallel-size 1 --expert-parallel-size 1 --moe-a2a-backend none
  --page-size 128 --quantization compressed-tensors --disable-shared-experts-fusion
  --dtype bfloat16 --trust-remote-code --disable-radix-cache
  --mem-fraction-static 0.81 --context-length 65536
  --max-prefill-tokens 65535 --chunked-prefill-size 32768
  --watchdog-timeout 18000 --max-running-requests 1
  --kt-method LLAMAFILE --kt-num-gpu-experts 32 --kt-cpuinfer 32 --kt-threadpool-count 1
  --kt-weight-path $DSV4_MODEL_ROOT/cache/dsv4_layer{layer_idx}_mxfp4.gguf
  --host 0.0.0.0 --port 18080
```

### `DSV4_PREFILL_STREAM` 决定了一整组开关

`serve.sh` 里整个 `KT_*` 块被 `if [ "${DSV4_PREFILL_STREAM:-0}" = "1" ]` 包住。设为 1 时会**同时**打开：

```
KT_PREFILL_STREAM=1            KT_PREFILL_STREAM_THRESHOLD=512
KT_DYNAMIC_RESIDENT=1          KT_MXFP4_DEPOOL=1
KT_MXFP4_GGUF_DEDUP=1          KT_SIDE_STREAM=1
KT_STREAM_WARMUP=1
```

不设时这些变量**根本不会被创建**。

> **这一点很重要**：教程的精度章节继承 Step 6 的默认值（流式**关闭**），而吞吐表显式全开。
> 修复前这两套配置互斥 —— 全开会掉约 30pp 精度（见第 2 章）。**修复后两者才第一次能同时成立。**

## 1.10 Step 6 —— 验证

```bash
bash "$DSV4_TOOLS/verify.sh"
```

冒烟测试：

```bash
curl -s --noproxy '*' http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":16}'
```

吞吐与精度：

```bash
bash "$DSV4_TOOLS/../../../tools/decode_throughput_test.sh"       # 四档吞吐
REPEATS=3 bash $R/scripts/tools/gpqa_accuracy_repeat.sh           # GPQA-Diamond 三轮
```

## 1.11 运维注意

**停服务用 `kill -INT`，别用 `kill -9`。** `kill -9` 会在 `/dev/shm` 留下残留段；
本次一天下来攒到 **2621 个**（正常 60–100），之后流式 prefill 出现**间歇性段错误**
（`kt_stream_prefill.py:230` 的并发 mmap→pinned 拷贝）。清理后 40 题重放全过、零崩溃：

```bash
find /dev/shm -maxdepth 1 -type f -user "$USER" -mmin +20 -delete
```

**别用 `pgrep -f` / `pkill -f`** 匹配自己脚本里出现过的字符串 —— 它会匹配到发起命令自身的命令行，
造成自杀或等待链永不退出。拆写模式串（`'gpqa_acc''uracy'`）或按 PID 杀。

---

# 2. Bug #1：动态热专家的常驻权重从来没被写进去

**状态：已修复，已提交 PR 给 `Pan-Boyi/sglang`。**

## 2.1 复现

### 最小用例（推荐，几分钟）

见 `repro_dyn_resident.py`。它在**单个服务实例内**做前后对照，不依赖任何数据集：

```bash
# 复现（应当 FAIL，退出码 1）
source ~/dsv4.env
DSV4_PREFILL_STREAM=1 bash "$DSV4_TOOLS/serve.sh"
python3 repro_dyn_resident.py

# 验证修复（应当 PASS，退出码 0）
DSV4_PREFILL_STREAM=1 KT_DYNAMIC_RESIDENT=0 bash "$DSV4_TOOLS/serve.sh"
python3 repro_dyn_resident.py
```

流程：12 个短 prompt（各 <512 token）→ 1 个长 prompt（>512 token，触发流式）→
同样 12 个短 prompt。判据是 `</think>` 在输出中的出现率，触发后升高即为复现。

### 完整证据（GPQA-Diamond）

三臂，同一批 60 道题（GPQA-Diamond idx 70–129），其余配置完全相同：

| 臂 | 配置 | `</think>` 污染率 | 精度 | vs 基线 |
|---|---|---:|---:|---|
| 基线 | 流式 ON, DYN=1 | **36/60 = 60.0%** | **38.3%** | — |
| A | 流式 **OFF** | **0/60 = 0.0%** | **71.7%** | +33.3pp, z=3.89 |
| B | 流式 ON, **DYN=0** | **0/60 = 0.0%** | **70.0%** | +31.7pp, z=3.67 |

**Arm B 是判决性的**：流式 prefill 照常开启，仅关闭 `KT_DYNAMIC_RESIDENT`，
污染完全消失、精度恢复，与"干脆关掉流式"在统计上无差别。

---

## 2.2 根因（2026-08-25 定位并修复）

## 2.3 一行结论

`kt_stream_prefill.py` 的 `_apply_resident_layer_depool`：

```python
torch.index_select(w13, 0, top, out=layer.w13_weight.data)   # ← 静默 no-op
```

**torch_npu 的 `index_select(out=)` 在目标带 ACL 私有格式 `FRACTAL_NZ` 时什么都不写。**
它把 `.data` 返回的临时 Tensor 重绑定到一个新分配的 ND tensor，算完即弃，
`nn.Parameter` 自己的 storage 一个字节没动。**无异常、无告警、无日志。**

## 2.4 为什么后果这么严重

四行相邻写入里，只有权重两行的目标是 NZ：

| 写入目标 | 格式 | 结果 |
|---|---|---|
| `w13_weight` / `w2_weight` | **FRACTAL_NZ** | **丢写，仍是静态前缀专家 0–31** |
| `w13_weight_scale` / `w2_weight_scale` | ND | 写成功，是热专家 `top[i]` 的 scale |
| `gpu_experts_mask` / `logical_to_gpu_index` | ND | 写成功，是热专家集合 |
| kt-kernel C++ pinned mask | ND | 写成功 → CPU **跳过**这 32 个热专家 |

于是槽位 `i` = 专家 `i` 的权重 × 专家 `top[i]` 的 scale，而 CPU 又以为 `top[i]` 已常驻
而不去算它。日志里 `share=0.644` 表明这批错配槽位承担了 **64.4% 的路由激活质量**。
这是纯参数状态，污染所有后续请求直到进程重启。

## 2.5 触发条件

需要 `KT_PREFILL_STREAM=1` + `KT_DYNAMIC_RESIDENT=1` + `KT_MXFP4_DEPOOL=1` 同时打开。
非 depool 路径的 `_apply_dynamic_residency` 用的是 `.copy_()`，**本来就是对的**。
而 `serve.sh` / `launch_ds4flash_npu.sh` 里这三个全是默认 on。

## 2.6 修复

```python
layer.w13_weight.data.copy_(torch.index_select(w13, 0, top))
layer.w2_weight.data.copy_(torch.index_select(w2, 0, top))
layer.w13_weight_scale.data.copy_(torch.index_select(s13b, 0, top))
layer.w2_weight_scale.data.copy_(torch.index_select(s2b, 0, top))
```

`Tensor.copy_` 是格式感知的，原地写 Parameter 自己的 storage（decode NPU graph 捕获的正是它）。
代价：K=32/4096/4096 上 5.67 ms vs no-op 3.15 ms，43 层 × 2 ≈ **每次流式 prefill +0.16 s**。
原注释说 index_select "~free" —— 它 free 是因为它什么都没做。

## 2.7 证据链

| 证据 | 来源 | 结论 |
|---|---|---|
| 直接调生产函数：`WEIGHT changed at all: False`，`SCALE == intended: True` | agent E4 | 机制层直证 |
| `param.data content == ref[:K] (STALE): True`；ND 目标则正常 | agent E2 | 判别因子是 NZ 格式 |
| 生产形状下同样丢写；两棵树的 `w13_weight` 都是 FRACTAL_NZ | agent E3 | 缺陷普遍存在 |
| 强制 `top=[0..31]` → 用例 PASS（漂移 0/15） | 主线 identity 实验 | **盲测印证**：陈旧权重恰好等于 top，三者自洽 |
| 修复后相似度 0.3609 → 0.9125 | 回归用例 | 从"面目全非"变成"零星 token 变化" |

identity 实验的预测是在**知道根因之前**下的：若根因是"换了哪些专家"或"权重来源改变"，
它应当 FAIL；它 PASS 恰恰证明权重根本没被写。

## 2.8 对照树同样有此缺陷

`/mnt/workspace/dsv4-workspace/ktransformers-AK/third_party/sglang/.../kt_stream_prefill.py:1015-1018`
写法逐字相同，`layer.w13_weight` 同为 `FRACTAL_NZ`，五个开关默认值一致。
该树自己的文档 `PLAN_a3_gpqa_accuracy_align.md:675` 已把 `KT_DYNAMIC_RESIDENT=1` 认定为"主因"，
但归因停在现象层（"expert 这轮走 int8、下轮走 MXFP4，两条路数值本就不同"），
处理方式是关掉它（`:585` 的精度对齐配置写死 `KT_DYNAMIC_RESIDENT=0`）。
**那个精度解释是错的** —— 真实机制是权重压根没写进去。

## 2.9 排查方法论上的教训

运行期取证 dump 了 43 层槽位权重指纹，判据是"唯一且非零"——
**但陈旧的前缀专家权重同样唯一且非零**，判据分辨不了 stale 与 correct，
于是得出"三条不变式全部成立"的错误安心结论。
**验证"写进去了什么"时，指纹必须能区分「正确的值」和「旧的合法值」**，
正确做法是拿写入源 `src[idx[0]]` 与目标 `dst[0]` 直接比对。


## 2.10 修复验证（GPQA-Diamond，五开关全开）

服务在 GPQA 开始前已因回归用例发生过 3 次换入，GPQA 自身在 idx 74/76 又触发 2 次 ——
**全程都在"常驻集已被外来 prompt 换过"的状态下服务**，正是崩坏轮里精度掉到 40.91% 的那个状态。

同一批题、配对比较：

| 区间 | n | 修复 acc | 崩坏 acc | 修复 `</think>` | 崩坏 `</think>` |
|---|---:|---:|---:|---:|---:|
| idx 0–19 | 20 | 85.0% | 75.0% | 0.0% | 5.0% |
| idx 20–39 | 20 | 60.0% | 50.0% | 0.0% | 5.0% |
| idx 40–59 | 20 | 70.0% | 75.0% | 0.0% | 5.0% |
| idx 60–79 | 20 | 85.0% | 70.0% | 0.0% | 15.0% |
| **idx 80–99** | 16 | **56.2%** | **12.5%** | **0.0%** | **68.8%** |

**触发点之后合计（n=22）**：修复 acc 59.1% / `</think>` 0.0%；崩坏 acc 18.2% / `</think>` 59.1%。
翻转 +11/−2，**McNemar z=2.50，显著**。

触发点**之前**四段两轮精度互有高低、差异都在 temp=1 采样噪声内（实测同配置重跑约 29% 题目翻转）
—— 这正是应有的样子，因为那时两边代码行为相同。**差异只在触发点之后出现，且单向、显著。**

## 2.11 加固建议（防止这类丢写再次悄悄回归）

这类丢写无声无息，唯一的防线是写完立刻验一次。建议在 `_apply_resident_layer_depool` 加一个
只在首层执行的断言：

```python
if L == 0 and _DEBUG_ASSERT:
    assert torch.equal(layer.w13_weight.data[0].cpu(), w13[top[0]].cpu()), \
        "resident weight write was dropped (private-format out= no-op?)"
```

关键是比对**写入源 `src[top[0]]`** 与**目标 `dst[0]`**。
不要用"指纹唯一且非零"这类弱判据 —— 陈旧的前缀专家权重同样唯一且非零，分辨不了。

## 2.12 修复面完整性核查

全文件扫描确认没有第二处同类问题：

- `_apply_dynamic_residency`（非 depool 路径）八处写入**全部**用 `.copy_()`，本来就正确
  —— 这解释了为什么本 bug 需要 `KT_MXFP4_DEPOOL=1` 才触发
- 全文件仅剩一处 `out=`（`:288 index_select(..., out=buf)`），`buf` 是 host 侧 pinned 缓冲，
  非 NPU 私有格式，不受影响
- `_ACL_FORMAT_FRACTAL_NZ` 的其余用处都是构造新张量或整体 cast，没有再向 NZ 目标做 `out=` 写入


## 2.13 为什么这个 bug 长期没被发现

动态热专家的换入只有在流式 prefill 打开时才可达，而整块 `KT_*` 变量被一个 `if` 包着
（`kt-kernel/tools/ascend_dsv4/serve.sh`）：

```bash
if [ "${DSV4_PREFILL_STREAM:-0}" = "1" ]; then
  export KT_PREFILL_STREAM=1
  export KT_MXFP4_DEPOOL="${KT_MXFP4_DEPOOL:-1}"
  export KT_DYNAMIC_RESIDENT="${KT_DYNAMIC_RESIDENT:-1}"
  export KT_SIDE_STREAM="${KT_SIDE_STREAM:-1}"
  export KT_MXFP4_GGUF_DEDUP="${KT_MXFP4_GGUF_DEDUP:-1}"
  ...
fi
```

而 `DSV4_PREFILL_STREAM` 在文档的 Configuration Reference 表里默认值是 **unset**
（`| DSV4_PREFILL_STREAM | unset | 1 enables streaming prefill |`），Step 6 的启动方式就是
裸的 `bash "$DSV4_TOOLS/serve.sh"`。

所以在默认路径上，**这五个变量根本不会被导出** —— `KT_DYNAMIC_RESIDENT` 不是被显式设成 `0`，
而是压根没有被创建，换入代码一次都走不到。

关键在于精度那一节继承的正是这个配置：`Optional: Accuracy Validation` 紧跟 Step 6，
**并不重启服务**，所以文档发布的 GPQA-Diamond 73.23% 是在换入休眠的状态下测的。
而 Measured Results 的吞吐表显式打开了五个开关。

**两个已发布数字来自两套不同配置，其中只有吞吐那套会走到这段代码。**

后果是：照文档字面操作的人永远碰不到这个 bug；它只在"拿吞吐配置去做吞吐以外的事"时出现 ——
而文档没有任何一处提示这两套配置互斥。本次复现正是因为要对齐 Measured Results 的配置才撞上。

对照树（`ktransformers-AK/tools/launch_ds4flash_npu.sh`）的情况略有不同：那里五个开关是
**无条件默认 on**，所以更容易触发。该树自己的笔记（`PLAN_a3_gpqa_accuracy_align.md:675`）
已经把 `KT_DYNAMIC_RESIDENT=1` 标为精度下降的主因，精度对齐配置（`:585`）也写死了
`KT_DYNAMIC_RESIDENT=0` —— 但归因停在"expert 这轮走 int8、下轮走 MXFP4，两条路数值本就不同"，
没有挖到丢写这一层，处理方式是关掉该特性而非修复它。


---

# 3. Bug #2：SwiGLU clamp 在各路径间不一致

**状态：机理未完全解释，暂不提交。**
统一语义（三处代码里凡是 clamp 的，写法一致）：
```
gate = clamp(gate, max=limit)          # 仅上界
up   = clamp(up, -limit, limit)        # 两侧对称
# clamp 在 silu 之前；limit = 10.0
```

核对基准：
| 代码库 | commit | 日期 |
|---|---|---|
| **P** = Pan-Boyi/sglang（NPU 集成分支，我们复现用的） | `522a8b73d` | 2026-08-15 |
| **G** = kvcache-ai/sglang（ktransformers 的 GPU submodule） | `bc7f005` | 最新 main |
| **U** = sgl-project/sglang 上游 | `ca08d447c` | 2026-08-24 |

---

## 3.1 三库 clamp 现场总表

| clamp 点 | P（NPU 集成） | G（GPU） | U（上游） |
|---|---|---|---|
| **shared expert** | ✅ `deepseek_v2.py:441` | ❌ `deepseek_v2.py:295` 裸 `act_fn` | ✅ `deepseek_v2.py:388` |
| **加速器上的 routed 专家** | ❌ | ✅ | ❌ |
| **CPU 卸载 routed 专家** | ✅ `kt_ep_wrapper.py:428` | ✅ `kt_ep_wrapper.py:3726` | ❌ 无此行 |
| **启动断言 / 校验** | 无 | ✅ `mxfp4_deepseek.py:188` | 无 |

**三个库各 clamp 一个不同的子集，没有两个是一样的。**

---

## 3.2 逐条展开

### P —— Pan-Boyi/sglang @ `522a8b73d`

| 点 | 位置 | clamp |
|---|---|---|
| shared expert | `models/deepseek_v2.py:441` `elif self.swiglu_limit is not None:` | ✅ |
| NPU 常驻专家（量化） | `moe_runner/ascend.py:123` `inner = NPUSwigluQuant()` — 无参构造 | ❌ |
| NPU 常驻专家（非量化） | `moe_runner/ascend.py:135` `inner = NPUSwiglu()` — 无参构造 | ❌ |
| DeepEP 分支 | `ascend.py:114` `NPUSwigluDeepEPKernel(limit=config.gemm1_clamp_limit)` — DeepSeek 填的是 `swiglu_limit`，此字段恒 `None` | ❌ |
| 流式 prefill | `kt_stream_prefill.py:1062` `npu_dequant_swiglu_quant(...)` — 该算子无 limit 参数，全文件 swiglu/clamp 命中 4 处全是注释或算子名 | ❌ |
| CPU 卸载专家 | `kt_ep_wrapper.py:428` `swiglu_limit=layer.moe_runner_config.swiglu_limit or 0.0` | ✅ |

出处：`kt_ep_wrapper.py:428` 那行由 **Pan-Boyi `013b5ae40`**（2026-08-13，"moe: integrate KT offload with Ascend dispatcher"）引入，只存在于该集成分支血统上。

→ **CPU 卸载专家是唯一会 clamp 的 routed 专家**，NPU 上的 routed 专家全部不 clamp。这就是不一致的来源。

### G —— kvcache-ai/sglang @ `bc7f005`（GPU）

| 点 | 位置 | clamp |
|---|---|---|
| shared expert | `models/deepseek_v2.py:295` `x = self.act_fn(gate_up)` — `DeepseekV2MLP` 连 `swiglu_limit` 参数都没有 | ❌ |
| GPU 常驻专家（marlin MXFP4） | `mxfp4_deepseek.py:558` → `v4_marlin_moe.py:487` → `_swiglu_kernel`（`tl.minimum` / `tl.maximum`） | ✅ |
| GPU routed（triton） | `moe_runner/triton_utils/fused_moe.py:582-630`，含 `assert swiglu_limit == 10` | ✅ |
| GPU routed（另一套 triton） | `fused_moe_triton/fused_moe.py:318-319` | ✅ |
| deep_gemm 路径 | `moe_runner/deep_gemm.py:122` | ✅ |
| CPU 卸载专家 | `kt_ep_wrapper.py:3679-3726`，且对非 MXFP4/8 方法归零（PR #61） | ✅ |
| **启动断言** | `mxfp4_deepseek.py:188` `assert is_2604b == (swiglu_limit is not None)` | — |

shared expert 不 clamp 的前提：官方教程的启动命令带 `--disable-shared-experts-fusion` → `num_fused_shared_experts = 0`（`deepseek_v2.py:445-449`）→ shared expert 走 `DeepseekV2MLP`，那条路径没有 clamp。若开启 fusion，shared expert 并入 routed MoE，就会跟着 clamp。

→ **所有 routed 专家一致 clamp，且用断言堵死"limit 没传到"**；shared expert 在文档配置下不 clamp。

### U —— sgl-project/sglang @ `ca08d447c`（上游）

| 点 | 位置 | clamp |
|---|---|---|
| shared expert | `models/deepseek_v2.py:388-394` | ✅ |
| NPU routed 专家 | `hardware_backend/npu/quantization/fused_moe_method_npu.py` — **7 处**全是裸 `torch.ops.npu.npu_swiglu(hidden_states)`；该文件 `clamp`/`swiglu_limit` 命中数 **0** | ❌ |
| CPU 卸载专家 | `kt_ep_wrapper.py` 存在，但**没有** `swiglu_limit` 这一行 | ❌ |
| CUDA/ROCm 路径 | `fused_marlin_moe.py:50-51`、`fused_moe_triton/layer.py:265`、`moe_runner/{deep_gemm,aiter}.py`、`mega_moe.py:260` | ✅ |

上游把 `swiglu_limit` 完整贯通到了 CUDA/ROCm，但**一条都没连到 NPU**。上游没有 `moe_runner/ascend.py`、没有 `hardware_backend/npu/moe/activation.py`、没有 `NPUSwiglu` 这个类，也没有 `kt_stream_prefill.py`。

→ **上游 NPU 路径只 clamp shared expert。**

---

---

## 3.3 四臂精度实测（Atlas A3 单 die，GPQA-Diamond non-thinking，每臂三轮 × 198 题）

命令行与 `KT_*` 开关完全一致（`KT_PREFILL_STREAM=1`、`KT_DYNAMIC_RESIDENT=1`、
`KT_SIDE_STREAM=1`、`KT_STREAM_WARMUP=1`、阈值 512），仅 clamp 配置不同。每臂启动前核验分支状态。

| 臂 | shared | NPU routed | CPU offload | R1 / R2 / R3 | mean | SD |
|---|---|---|---|---|---|---|
| **A** 当前 PR 分支 | ✅ bf16 | ❌ | ✅ fp32 | 72.22 / 75.25 / 74.75 | **74.07%** | 1.62pp |
| **C** 全不 clamp | ❌ | ❌ | ❌ | 73.23 / 74.24 / 73.74 | **73.74%** | 0.51pp |
| **D** 上游语义 | ✅ bf16 | ❌ | ❌ | 72.73 / 73.23 / 70.71 | **72.22%** | 1.34pp |
| **B** 全 clamp | ✅ bf16 | ✅ bf16 | ✅ fp32 | 69.70 / 69.19 / 70.20 | **69.70%** | 0.51pp |

逐题配对（每题按三轮命中率 0/⅓/⅔/1，n=198）：

| 对比 | 差 | t (df=197) | 判定 |
|---|---|---|---|
| C − A | −0.34pp | −0.19 | 不显著 |
| C − D | +1.52pp | +0.92 | 不显著 |
| A − D | +1.85pp | +1.26 | 不显著 |
| D − B | +2.53pp | +1.47 | 不显著 |
| **C − B** | **+4.04pp** | **+2.38** | **p<0.05** |
| **A − B** | **+4.38pp** | **+2.49** | **p<0.05** |

### 结论：只有 NPU routed 专家那一处 clamp 有可测影响

A、C、D 三臂两两之间全部不显著，均值挤在 72.2–74.1%；它们的共同点是 NPU routed 专家不 clamp。
shared 的开关、CPU 的开关，都测不出影响。

按「NPU 是否 clamp」合并 12 轮：

```
NPU 不 clamp  9 轮  mean = 73.34%  SD = 1.38pp
NPU    clamp  3 轮  mean = 69.70%  SD = 0.51pp
逐题配对 +3.65pp  t = 2.53 (df=197)  p < 0.05
  不clamp 更好 50 题 / 更差 29 题 / 持平 119 题   符号 z = 2.25
轮级 Welch t = 6.70
12 轮里最低的三轮，恰好就是 NPU-clamp 那三轮
```

### 实现已验证无 bug

在空闲 NPU die 上直接验 `apply_swiglu_limit_`（`probe_swiglu_halves.py`）：

- `activate_left=True` 确为「左半 = gate」：`silu(L)*R` 相对误差 0.017 / 余弦 0.99985；
  反向假设 `silu(R)*L` 相对误差 0.972 / 余弦 0.532。**半边分配正确。**
- 广播边界向量按预期工作：左半下界完整保留（−12.25 原样，无下界），右半收到 ±10，
  两半均封顶 10。**实现正确。**

所以 3.65pp 不是实现缺陷。

### 为什么只在 NPU 侧显形

开着 `KT_DYNAMIC_RESIDENT` 时，NPU 那 32 个常驻槽装的是**热专家**。运行日志里
`inline resident ... share=0.644` —— 这 32 个专家承载 **64% 的 routed 激活质量**，
CPU 上那 224 个只占 36%。所以不是「32 vs 224」，是「64% vs 36%」。

**尚未验证的部分**：热专家的激活是否系统性更大。我们只测过 NPU 侧的 clamp 命中率
（29.2% 的 forward 至少截断一个元素，峰值 7.3× limit），**CPU 侧的命中率从未测量**。
若 CPU 侧命中率远低于 NPU 侧，这个解释就完整了。

### 两处需要修正的早期说法

1. 轮间 SD 不是 0.51pp。12 轮合并估计 **1.11pp**（各臂 0.51 / 1.62 / 1.34 / 0.51）。
   B、C 两臂三轮挤在 1pp 内是巧合。臂间差要大于约 1.6pp 才能只靠三轮均值判定。
2. 「clamp 越多分越低、单调」是错的 —— A（两处 clamp）比 C（零处）略高。
   单调关系不存在，只有 NPU 那一处有影响。

---

## 3.4 CPU 侧 clamp 命中率实测（2026-08-26）

同一次运行、同一批 token（GPQA idx 70–109 完整重放，40 题），两个计数器都打在 clamp **之前**。
CPU 侧计数器是给 `kt-kernel/operators/llamafile/moe.hpp` 的 `act_fn` 加的 thread_local 计数
（见 `measure_cpu_clamp_hit.py`），用环境变量给的参考 limit 计数，与 `config_.swiglu_limit`
是否生效解耦。本次为绕开一个环境相关的间歇崩溃，`DSV4_PREFILL_STREAM=0`，
因此 NPU 常驻的是静态前缀专家 0..31，不是热专家。

| | 元素数 | 被截断 | 占比 | `gate` 峰值 | `up` 峰值 |
|---|---:|---:|---:|---:|---:|
| **CPU 卸载专家**（224 个） | 26.84e9 | 352,316 | **0.00131%** | **60.99** | **85.64** |
| **NPU 常驻专家**（32 个，静态） | 10.95e9 | 135,694 | **0.00124%** | 28.38 | 45.00 |

### 「clamp 只咬热专家」这个解释被推翻

CPU 侧的命中率**略高于** NPU 侧，峰值高得多（85.6 = 8.6× limit，对 45.0）。
CPU 侧咬得更多更狠，精度上却测不出影响 —— 与「只有 NPU 那处 clamp 有代价」的观测矛盾。

### 但假设里有一半成立：热专家的激活确实更大

同一个 NPU 计数器在两种常驻策略下：

| NPU 常驻的是 | `gate` 峰值 | `up` 峰值 | 命中的 forward |
|---|---|---|---|
| **热专家**（`KT_DYNAMIC_RESIDENT=1`） | **52.75** | **72.50** | 29.2% |
| 静态前缀 0..31（流式关闭） | 28.38 | 45.00 | 25.4% |

热专家峰值高 86% / 61%。所以「热专家激活更大」成立，「因此 clamp 只在 NPU 侧有影响」不成立。

### 对「CPU clamp 无影响」这个说法的修正

更准确的表述是**没有检测到影响**。D→A 是唯一纯粹的 CPU clamp 开关对照（其余现场相同），
差值 **+1.85pp**（clamp 开的一侧更高），t=1.26，不显著 —— 但合并 SD 是 1.11pp、n=3，
这个设计本来就分辨不了 1.85pp。功效不足，不是无影响。
NPU 那 3.65pp 能测出来，是因为它够大。

### 结论

**机理仍未解释。** 已排除：实现 bug（硬件验证过半边分配与广播边界）、CPU clamp 是 no-op
（命中率比 NPU 还高）、热专家激活更大能单独解释（方向反了）。

---

# 4. 实测结果

## 4.1 吞吐（Atlas A3 单 die）

四档，1 次预热 + 3 次测量，`--kt-cpuinfer 32 --kt-threadpool-count 1 --max-running-requests 1`。
测自已提交 PR 的那个分支（`pr/npu-resident-weights`）：

| prompt tok | `KT_DYNAMIC_RESIDENT=1` | `=0` | 动态热专家收益 |
|---:|---:|---:|---:|
| 118 | 18.92 | 17.54 | +7.9% |
| 801 | 20.35 | 17.69 | **+15.0%** |
| 3944 | 19.34 | 17.73 | +9.1% |
| 7823 | 19.15 | 17.61 | +8.7% |

文档给的区间是 19–22.5 tok/s，实测落在区间内。

**重点**：这些收益在修复 bug #1 之前只能以约 30pp 精度为代价换取。动态热专家是**修复后才第一次可用**。

## 4.2 精度（GPQA-Diamond，non-thinking，198 题）

| 配置 | 轮次 | 均值 |
|---|---|---|
| 修复前（五开关全开） | 1 轮 162 题中止 | 54.32%（`</think>` 污染 24.7%） |
| **已提交 PR 的配置** | 3 轮 | **74.07%**（SD 1.62pp） |
| 文档参考值（910B） | 3 轮 | 71.72%（SD 1.80pp） |

`</think>` 污染在修复后全部为 **0.0%**。

**轮间噪声**：12 轮合并估计 **1.11pp**（各臂 0.51 / 1.62 / 1.34 / 0.51）。
臂间差要大于约 **1.6pp** 才能只靠三轮均值判定 —— 这个数字在设计任何 A/B 时都要先算。

---

# 5. 给上游的反馈清单

1. **#2159 依赖 #2157，文档未说明**，且 clone main 拿不到 `ascend_npu.h`。
2. **CMakeLists 写死 `/usr/bin/g++`**，`CC`/`CXX`/`-DCMAKE_CXX_COMPILER` 全部失效（§1.7）。
3. **`verify.sh` gate 5 因 `grep -c` 用法恒失败**（§1.7）。
4. **精度配置与吞吐配置互斥，但同篇文档并列给出**（§1.9）—— 修复前二者不可能同时成立；
   修复后可以，建议在文档中明确写出两组数字各自的开关状态。
5. **bug #1** 已提 PR 给 `Pan-Boyi/sglang`（见第 2 章）。
6. **bug #2** 暂不提交，需要上游先回答「`swiglu_limit` 在推理期到底该不该施加」（见第 3 章）。

# 6. 相关文件

| 文件 | 用途 |
|---|---|
| `patches/0001-*.patch` | bug #1 修复，可 `git am` 到 `522a8b73d`。**已提交 PR** |
| `patches-followup/0001-*.patch` | bug #2 修复，暂不提交 |
| `repro_dyn_resident.py` | bug #1 最小回归用例（temp=0 逐字节比对，自带新鲜度自检与噪声基线） |
| `probe_swiglu_halves.py` | 在空闲 die 上验证 `activate_left` 半边分配与广播边界 clamp |
| `measure_clamp_hit.py` | NPU 侧 clamp 命中率计数器（env 门控，可打/可回滚） |
| `measure_cpu_clamp_hit.py` | CPU 侧（kt-kernel C++）clamp 命中率计数器，thread_local 零共享状态 |
| `ab_stream_test.py` | 流式 prefill A/B 对照台 |
| `dsv4-logs/` | 全部运行日志 |
