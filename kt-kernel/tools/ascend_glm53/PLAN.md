# GLM-5.3-Flash 单卡 MoE CPU Offload —— 计划与进展

**活文档。** 每完成一个阶段就回来改状态和结论。看日期，不要假设它是最新的。

- 最后更新：2026-08-31（P2 跑通；性能根因 = NUMA 页放置，已验证 2.8×）
- 目标：GLM-5.3-Flash 跑在**一个昇腾 910C die** 上。NPU 侧 INT8（含常驻专家），
  其余路由专家 offload 到主机 DDR，用 MXFP4 由 KTransformers 的 kt-kernel 在 CPU 上算。
- 出口判据：能从零配环境、拉起服务、**精度对齐**。

---

## 文档分工

| 看什么 | 去哪 |
|---|---|
| 怎么用：脚本、变量、会咬人的几条 | [`README.md`](./README.md) |
| 为什么是这个形状：容量账、NUMA 悬崖、roofline、profile、流式设计 | [`DESIGN.md`](./DESIGN.md) |
| 精度怎么建立的、每个数字有多硬、测量纪律 | [`ACCURACY.md`](./ACCURACY.md) |
| 还没定的决策、还没干净数据的项、清理的出口标准 | 本文 |

**本文只记当前事实和未决项，不记怎么走到这里的**——过程看 git 历史。

---

## 0. 状态一览

单 die 服务 GLM-5.3-Flash，路由专家 offload 到主机 DDR。**标准配置：流式 prefill +
动态热专家常驻，K=32。**

| 阶段 | 状态 |
|---|---|
| 环境、权重转换（MXFP4 → 逐层 GGUF） | ✅ 42/42 层逐位验证通过 |
| 服务拉起 + 冒烟 | ✅ |
| 性能定位 | ✅ 已达带宽 roofline，见 [`DESIGN.md`](./DESIGN.md) §3 |
| 精度 | ✅ GSM8K 与基线相容（±2.4pp）；PPL 32 窗配对对比臂已完成，代价 +1.13% |
| dynamic hot | ✅ **闭环**：TPOT 1.211× ±0.033、TTFT 5.1× @14650 token、PPL +1.13% |

### 未完成项

| 项 | 状态 |
|---|---|
| ~~TPOT 交错重复实验~~ | ✅ **已完成**，n=4 交错，1.211× ±0.033，见 [`ACCURACY.md`](./ACCURACY.md) §3 |
| ~~PPL 对比臂（32 窗）~~ | ✅ **已完成**，+1.13%，32 窗中 26 窗更差，见 [`ACCURACY.md`](./ACCURACY.md) §3 |
| **dynamic hot 的 TPOT 离散 5.5%** | 是 hybrid（0.7%）的 8 倍。机制推测是每次 prefill 重选常驻集，**未验证** |
| `KT_PREFILL_STREAM_THRESHOLD` | 交叉点实测 ~1100 token。**决定交给业务配置**，不再由本项目定死 |
| `--chunked-prefill-size` 调大 | TTFT 主项是 chunk 数不是 token 数，这是最大的一根杠杆。⛔ **反方向先撞了墙**：流式默认的 8192 会让 KDA 的 ~3 GiB workspace OOM（`3.00 GiB` 要 / `3.03 GiB` 剩），默认已降到 **6144**（09-01 实测，见下 A3）。要调大必须先在别处腾显存。**未测** |
| `GLM53_PIN_CORES` 在 node ≥4 上致命 | 已自动跳过并告警，**根因未修**，见 README |
| DEPOOL=0 的代价 | 关掉会让流式 prefill 的 H2D 字节翻倍，且激活 [`DESIGN.md`](./DESIGN.md) §5 的 S1 隐患。**未测** |
| 混合负载的状态依赖 | 短 prompt 不重置常驻集，会继承上一个长请求的热专家。**未测** |

⚠ **S1（[`DESIGN.md`](./DESIGN.md) §5）不适用于标准配置**：它在 `KT_MXFP4_DEPOOL=0` 臂上，而默认是 1。

---

## 1. 已拍板的决策

| # | 决策 | 理由 | 日期 |
|---|---|---|---|
| D1 | **一切从 `ktransformers-AK` 进**。`third_party/sglang` submodule 改指向 `wenxuewuhd/sglang-dllm @ glm53_cpu_offload` | 上游 `kvcache-ai/sglang` 没有 GLM-5.3 的昇腾适配。我们的 fork 两样都有 | 08-31 |
| D2 | **必须关掉 shared-expert 融合**（`--disable-shared-experts-fusion`） | ⚠ **已从「优化选项」升级为硬要求**：`3f7db2fece` 让 `FusedMoE` 变成 289 宽（`glm5_next.py:1652` = `n_routed_experts + num_fused_shared_experts`），而 `kt_expert_masks` 按 288 建表，撞上宽度守卫 `kt_ep_wrapper.py:440-447`，**模型加载时直接失败**。不是静默降级——守卫的报错里就写着修法 | 09-01 |
| D3 | **GGUF 转全部 288 个专家**，不只转非常驻的 256 个 | 多占 17 GiB 磁盘，换取常驻数（32/24/16）可随意调而不必重转。重转一次要几十分钟 | 08-31 |
| D4 | ~~暂不删 FP8 原版~~ **已被 D8 取代**：MXFP4 checkpoint 审核通过后已删，只留元数据清单 | | 08-31 |
| D5 | 常驻专家数起步 **32**，备选 24 | 容量账见 [`DESIGN.md`](./DESIGN.md) §2 | 08-31 |
| D6 | ~~先关 `KT_PREFILL_STREAM`~~ **已被 D9 取代**（当时它的张量名还是 DeepSeek 拼法，见 [`DESIGN.md`](./DESIGN.md) §6，后已修） | | 08-31 |
| D7 | `KT_SIDE_STREAM` **默认开**，且与流式 prefill 解耦 | 它在 kt_ep_wrapper 导入时读取、用在捕获图的提交路径上，每个 decode step 都走，和流式 prefill 无关。我最初错误地把它锁在 `GLM53_PREFILL_STREAM=1` 里 | 08-31 |
| D9 | **出口标准配置 = 流式 prefill + 动态热专家，K=32** | TTFT 收益确凿（5.1× @14650 token）；TPOT 收益见 [`ACCURACY.md`](./ACCURACY.md) §3 的区间。阈值 `KT_PREFILL_STREAM_THRESHOLD` **交由业务按自己的 prompt 分布配置**，本项目不定死 | 09-01 |
| D8 | FP8 原版已删，先落一份**元数据清单** | `GLM-5.3-Flash-FP8-metadata/`：76108 个张量的 dtype/形状/字节范围 + 每分片大小与首 1 MiB 的 sha256（71 MB）。重新下载后可逐张量核对 | 08-31 |

---

## 1.5 构建与环境的耐久事实

- kt-kernel wheel：`kt_kernel-0.7.0.post1-cp312-cp312-linux_aarch64.whl`，编译约 1 分钟。
  `CPUINFER_USE_ASCEND_NPU=1`，ARM SVE/BF16/I8MM 全 OFF（与 DSV4 验证过的配方一致），
  实际 `-march=armv8.2-a+fp16+dotprod`。
- ⚠ **submodule 不能只用 `--depth 1`**：它拉的是默认分支 tip，不含钉住的 SHA，
  而 git 的回退直接 fetch 被 `protocol.file` 加固（CVE-2022-39253）挡掉。
  必须显式 fetch SHA 再 checkout。`setup.sh submodules` 已按此写。
- 权重转换：单层约 20 秒 / 3.85 GB，42 层约 15 分钟。逐位验证通过。
- 验收：`setup.sh check` 打印 `PREFLIGHT OK`；`verify.sh` 全 PASS。

其余环境类的坑（`import custom_ops`、`npu-smi` 认不出 A3、代理劫持 127.0.0.1、
开发机 ≠ 目标机）见 [`README.md`](./README.md)「会咬人的几条」。

---

## 2. 出口标准（clean code / 开源前）

面向的是**清理与开源准备**，不是新功能。范围两块：`kt-kernel/tools/ascend_glm53/`
（外层脚本），以及 sglang fork 上本项目拥有的两个代码文件
（`layers/moe/kt_stream_prefill.py`、`layers/moe/kt_ep_wrapper.py`）加一个回归测试。
⚠ 与 int8 单卡线**零文件重叠**——那条线的文件不在范围内，逐条结论见 git 历史。

### A. 判据：改完必须过的四道

按锐利度排序。**前两道是零噪声的，任何差异都是真差异。**

| # | 判据 | 怎么跑 | 通过条件 | 要机器吗 |
|---|---|---|---|---|
| A1 | **困惑度逐位一致** | 改动前后各跑一次 `run_ppl.py --window 4096 --limit 12` | 每个窗口的 `nll_sum` **float 完全相等（0 ULP）**，不是「小数位看起来一样」 | **要一张空 die**。对邻居免疫：它是正确性判据，争抢只影响耗时 |
| A2 | **启动命令逐字节一致** | 固定同一组环境变量，`GLM53_DRY_RUN=1 ./serve.sh` 前后 diff | 完全一致。**shell 脚本重构的回归测试就是这个** | **不占 die，但要完整环境**：见下 |
| A3 | 冒烟与验收 | `setup.sh check` → `serve.sh` → `verify.sh` | `PREFLIGHT OK` + 全部 PASS | **要一张空 die**。同样对邻居免疫 |
| | ⚠ **A3 不是"粗糙的冒烟"** | 它用一个 14379 token 的 prompt，是**唯一覆盖多 chunk prefill 的判据**。A1 的窗口是 4096 = 单 chunk，看不见那条路径——09-01 的 chunk-8192 OOM 就是 A3 撞出来的，A1 全绿 | | |
| A4 | 吞吐没有退化 | `bench.sh` | decode 落在同臂已测区间内（见 [`ACCURACY.md`](./ACCURACY.md) §3），**且不要用单点下结论** | **要整机安静**。计时判据，`bench.sh` 的门本来也会拒 |

⛔ **A3 在未改动的代码上就 FAIL 过，两个独立原因，都已修（09-01，die 8）。**
清理开工前采基线时撞到的，记在这里因为两条都会再咬人：

1. **判据自己的算术错**：`verify.sh` 的容量公式漏了流式的 6.75 GiB convert slot，
   于是对一台**正确**的服务器报 `loaded 54.11 GB against 47.36 GiB predicted`。
   `15.60 + 6.75 + 0.9925×32 = 54.11`，正是实际加载值。`glm53_env.sh` 的 `--show`
   同一处也漏了。两处已修。
2. **真的 OOM**：流式默认 chunk 8192 下，KDA 在多千 token 的 prefill 里要 ~3 GiB
   workspace，而只剩 3.03 GiB → `eager_runner._execute_extend` 里 SIGABRT。
   全新起的服务上复现，不是碎片。默认已降到 **6144**，`verify.sh` 在它 14379 token 的
   prompt 上全部 PASS。

⚠ **教训比修复重要：PPL 全绿而验收门在崩。** A1 的窗口是 4096 token = 单 chunk，
永远碰不到这条路径；是 A3 先撞上的。**A1 锐利但覆盖窄，它证明不了长 prompt 能跑。**

⚠ **A1 是唯一能覆盖模型/MoE 代码路径的零噪声判据**：串行 teacher-forcing，
对服务重启、对 `--max-running-requests` 1→8 都逐位不变（见 [`ACCURACY.md`](./ACCURACY.md) §2）。
**任何触碰 `kt_stream_prefill.py` / `kt_ep_wrapper.py` 的改动都必须过 A1。**

⛔ **四道门守的是「没改坏输出」，守不住「改坏了设计意图」。**
2026-09-02 有一个 bug **在四道门全绿的情况下存在**：清理时把后台池排空判据改成
`_last_moe_layer()`，而该函数读的 `_REGISTRY` 正由它上方三十行在填充——
处理第 L 层时注册表只有 3..L，于是**每层都判真，排空发生在第一层**，
后台 O_DIRECT 读与模型加载的重叠（这段代码存在的全部理由）被扔掉。

四道门为什么都看不见：
- **A1** 不改数值，逐位相同——它量的是模型算什么，而这个从未改变；
- **A4** 是加载期不是 decode，不在它的窗口里；
- **A3** 照样起得来，只是加载慢了；
- **A2** 命令行没变。

抓到它的是**读代码时注意到注册表的填充顺序**。⚠ **所以这四道门不能替代 review。**
它们证明的是「行为没变」，而一个把优化空转掉的改动，行为恰恰没变。
（已 revert，`2ce28af1d4`。警告写进了 `_last_moe_layer()` 自己的 docstring 而非调用点：
所有从 `_moe_layers()` 派生的东西都是 forward 时的工具，加载期注册表不完整——
下一个人是在那个函数上产生念头的。）

⚠ **A2 曾经会打开一张卡，而它安全只是因为流程恰好不带参数。**
`glm53_env.sh` 末尾有 `case "${1:-}" in --show|show)`，而 **被 source 的脚本共享调用者的
位置参数**——所以 `./serve.sh --show` 里那个 `$1` 是 `serve.sh` 的，于是走进
`glm53_show_env` → `glm53_detect_soc` → `torch.npu.get_device_name(0)`，**在 die 0 上开一个
context**，而 `serve.sh` 原来只认 `--foreground`、其余一律忽略。已修（`eaeb3dc`）：
`glm53_env.sh` 只在被**执行**时响应 `--show`，`serve.sh` 拒绝未知参数。
**教训不是「别传那个参数」，而是一个判据的安全性不该取决于使用者恰好没做某件事。**

⚠ **A2 不是「纯字符串比对、随处可跑」。** dry-run 会先做 GGUF 计数检查
（`serve.sh` 在打印命令之前 `exit 1`），也要能 `import sglang`、要 CANN。
在缺这些的机器上，diff 比的是**两条一模一样的 FATAL —— 空过**。
跑 A2 之前先确认它真的打印出了命令行，别只看 diff 是否为空。

⚠ **A4 分辨率很粗**：同配置重复跑实测离散 5.7%，所以它只能发现大退化，
发现不了 5% 以内的。别拿它当「没改坏」的证据，那是 A1 的活。

### B. 看起来冗余、其实是补丁的地方 —— 删之前先看这里

清理最容易踩的就是这些。每一条都对应一次真实故障。

**sglang 侧**

- `kt_ep_wrapper.create_weights` 里的**宽度守卫**（`29b33ce2bd`）：层可以比放置表窄，
  但绝不能宽。看着像多余的 assert。
- `_apply_dynamic_residency` 的**掩码提交协议**（`952dca8a73`）：流式 prefill 的
  **每一个出口**都要提交常驻掩码，包括异常路径。看着像重复代码，它是权重与掩码
  不一致导致静默错误输出的修复。
- `map_logical_expert_id_for_gpu_load()` **返回 −1** 表示「跳过这个专家」。
  看着像哨兵值滥用，它是非常驻专家在加载期被跳过的唯一机制。
- **每个异常都被吞掉并回退 hybrid**：这是当前设计。⚠ 如果清理时改成让异常抛出
  （这本身是好事），**A3 的验收门也要同步改**——现在 `verify.sh` 靠日志里的
  `inline resident` 计数来判断流式是否真的生效，因为坏掉的流式路径从外面看是正常的。
- `KT_SIDE_STREAM` 在**导入时**读取，且与流式 prefill **解耦**（每个 decode step 都走）。
  别把它折进 `GLM53_PREFILL_STREAM` 的分支里——那是本项目早期犯过的错（D7）。
- `_ensure_slot` 在 **depool 与非 depool 两条路径上都要调用**。两条路最后都要把一层的
  完整专家权重 H2D 到同一个复用 HBM 槽，slot 不是 depool 专属的。

**脚本侧**

- `pkill -f -- "[-]-port 30013"` 里的**方括号**：防止模式匹配到发起命令的 shell 自己。
  看着像笔误，删掉会让脚本杀死自己。
- `grep -oP '\d+(?=\s*/ 65536)'` 里的 **`\s*`**：npu-smi 的输出对齐会变。
- `grep -ac ... | head -1`：grep 无匹配时**既打印 0 又退出 1**，`|| echo 0` 会产出
  `"0\n0"` 并让 JSON 写入器崩掉。这个 `head -1` 是修复不是赘余。
- `{layer_idx}` **不能写进 `${VAR:-default}`**：右花括号会提前终止参数展开，
  悄悄弄坏路径。`glm53_env.sh` 用 `if [ -z ... ]` 绕开，看着笨拙，是必要的。
- `serve.sh` 里三个**必须显式传**的参数：`--page-size 64`（DSA pool 有 assert）、
  `--disable-shared-experts-fusion`（否则 289 个槽静默降级）、
  以及**不传** `--attention-backend`（让 GLM 自己选 KDA/DSA）。
- `GLM53_CHUNKED_PREFILL_SIZE` 必须是**正数且 128 的倍数**：`-1` 会塌成 1，
  任何多于一个 token 的 prefill 都写越界，在 glibc 里 abort。
- ⛔ **`trap ... EXIT INT TERM` 目前一个脚本都没有**——这是**待补的要求，不是已有的代码**。
  `bench.sh:72` 起服务，`:69/:77/:250` 三条退出路径和 Ctrl-C 都不停它。
  后果实测过：杀掉 wrapper 留下服务，下一个人拿到
  `NPU out of memory ... 166 MiB free`，**报错指向受害者不是肇事者**。清理时补上。

### B2. 跑判据时会踩到的两个环境坑

- ⛔ **`env -i` 起 `serve.sh` 会挂死在 CANN 自己的 `set_env.sh`**
  （`line 31: CMAKE_PREFIX_PATH: unbound variable`，它在 `set -u` 下需要一个非空环境）。
  想复现干净环境只能逐个 `env -u GLM53_* KT_* SGLANG_*`，**不能清空**。
- ⛔ **`test_kt_stream_resident_commit.py` 直接跑会 `ModuleNotFoundError: No module named 'sglang'`**，
  需要 `PYTHONPATH=<repo>/python` 且 `unset LD_PRELOAD`。
  ⚠ 它在未改动的树上是 **PASS**（三种中断场景全部 consistent），
  所以它可以当清理的第一道门：**纯 CPU、不占 die、直接守住 `kt_stream_prefill.py`**。

### D. 不在清理范围内

- **算法与数值行为不要动。** 本项目的性能与精度结论都建立在当前行为上，
  改了就要重跑 [`ACCURACY.md`](./ACCURACY.md) §3 的全部测量，而其中一部分**至今没有干净数据**（见 §0 未完成项）。
- **`$GLM53_ARTIFACT_ROOT` 下的实验产物不要清理**，那是已发布数字的原始出处。
