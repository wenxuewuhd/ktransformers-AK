# DSv4 单卡 NPU 开源方案与执行计划（基于 sgl-project 主线）

> 更新：2026-08-11。路线已定：sglang 侧基于 **sgl-project 主线**（upstream NPU DSv4 已实测验证 OK），
> 不再走 kvcache-ai/sglang fork 路线。本文是唯一的执行底稿，包含全部 PR 清单、先后顺序与逐阶段任务。

---

## 1. 已确认的事实（盘点结论，勿重查）

### 1.1 我们的改动分布（三处）

| 位置 | 规模 | 内容 | 上游去向 |
|---|---|---|---|
| 主仓 `kt-kernel/` | 48 文件 / +7018 行 | Ascend vendor 后端（callback worker、`vendors/ascend_npu.h`）、K920 KML int8/int4 prefill GEMM、llamafile MXFP4 ARM 路径、python 侧 experts_base/loader | kvcache-ai/ktransformers（K1/K2） |
| 主仓 `third_party/llamafile/`（in-tree） | 2 文件 | ARM82 GEMV 软预取优化（K920 2.4×） | 随 K2 |
| 主仓 `script/`、`doc/`、`tools/` | ~350 文件 | 绝大部分是内部件（p27 脚本、patch 系列、handoff），**不上游**；只精选启动脚本+新写文档 | K3 |
| sglang fork（`wenxuewuhd/sglang-dsv4@dsv4_release`） | 76 文件 / +14158 行 | = sgl-project 2026-04-09 底座 + 3 commit（Yijie Zhu，将被主线取代）+ **46 commit（我们的）** | 移植后投 sgl-project（S1–S4） |
| `third_party/llama.cpp` 子模块 | ~150 行**未提交**工作区改动 | 见 1.3 | 不提 PR，随 K1 处置 |

### 1.2 upstream 现状

- **sgl-project 主线 NPU DSv4 已就绪**：PR #28980（6/30，MTP）+ #31931（7/28，PD 分离、chunked prefill 跨 chunk 保留 compression state、fused compressor、双流 decode）。**用户已实测验证 upstream 可用（2026-08）**。
- 环境要求：CANN 9.0.0 + torch_npu 2.10.0；官方镜像 `quay.io/ascend/cann:9.0.0-910b` / `9.0.0-a3`。
- 量化双格式都支持：ModelSlim 与 compressed-tensors 两个前端，落到同一套 NPU kernel
  （`NPUCompressedTensorsW8A8Int8[DynamicMoE]` 等）。PR 实测只用过 ModelSlim W8A8。
- 代码结构已重构：`hardware_backend/npu/`（`dsv4/`、`attention/ascend_dsv4_backend.py`、`quantization/`、
  `graph_runner/`）、`layers/moe/moe_runner/ascend.py`。移植按新结构对位。
- kvcache-ai/ktransformers 的 GPU DSv4-Flash（RTX 5090 单卡 + CPU MoE）已发布，kt-kernel 的 MXFP4
  只有 x86 AMX 路径——**我们的 llamafile/ARM MXFP4 + Ascend 后端是净增量**，无冲突。
- `kvcache-ai/kt-kernel` 独立仓不存在（404），kt-kernel 只活在 ktransformers 主仓树里。

### 1.3 llama.cpp 未提交改动 = 两个独立补丁

| 补丁 | 内容 | 性质 | 处置 |
|---|---|---|---|
| ① C 侧 ~130 行（`ggml.h/.c`、`ggml-quants.c/.h`、`ggml-common.h`） | `block_mxfp4` + `dequantize_row_mxfp4` + `ggml_vec_dot_mxfp4_q8_0`（NEON）+ type_traits 注册 id=39；含 K920 软预取调优 | **生产必需**（llamafile CPU MoE 走 ggml type_traits 分发）；master 回移植 + 自研调优 | Phase 0 落盘；随 K1 vendor 进 kt-kernel 或 build patch，RFC 时让 maintainer 拍板 |
| ② gguf-py 21 行（MXFP4 enum 2 行 + NumPy2 兼容 19 行） | 只服务 `tools/` 内部脚本对子模块 gguf-py 的 sys.path 引用 | **冗余**：生产 loader 用 pip gguf≥0.17（0.18 自带 MXFP4 + NumPy2 兼容） | 改脚本 import 后丢弃 |

**不给 ggerganov/llama.cpp 提 PR**：b3173 是历史 tag 收不了 PR，而 master 早有 MXFP4（我们 id=39 就是对齐它选的）。

### 1.4 upstream CI 与精度要求

- 唯一 PR 门：`.github/workflows/kt-kernel-tests.yml`。触发条件：PR 碰 `kt-kernel/**` + maintainer 打
  `run-ci` 标签 + 非 draft。跑在 self-hosted **x86 AMX** runner（`kt-cpu`）：
  `install.sh build` → `test/run_suite.py --hw cpu --suite default`。
- **精度只有算子级**：随机权重单层 MoE forward vs 纯 torch 参考，相对误差阈值（AMX int8 为
  `mean(|out−ref|)/mean(|ref|) < 0.05`）。无任何模型级（GPQA 类）CI；模型级精度靠 PR 里贴证据。
- 测试注册制：`test/ci/ci_register.py`，`HWBackend` 只有 CPU/CUDA/AMD（CUDA/AMD 也只是 placeholder）。
  NPU 测试进不了他们 CI，照 placeholder 先例处理，真跑留本地作证据。
- commit 格式强制（CONTRIBUTING regex）：`^[type](scope): msg`，type ∈ feat|fix|docs|…；每 commit 带
  `Signed-off-by`；不带 Co-Authored-By。

---

## 2. PR 总清单与先后顺序

### 2.1 sgl-project/sglang（1 个 RFC + 4 个 PR，串行 stack）

先开 **RFC issue**：单卡 910B + CPU 异构跑 DSv4-Flash（对比现状 16 卡 A3 全 NPU），附实测数字，
说明 S1–S4 拆分，并确认 kt-kernel 作为 optional dependency 的接受方式。

| # | 主题 | 内容 | 规模 | 依赖 |
|---|---|---|---|---|
| S1 | KT-EP wrapper：CPU MoE offload 核心 | NPU 版 kt_ep_wrapper（submit/sync/merge 与 NPU MoE 并行）、kt-kernel 对接、per-layer GGUF 路径解析、基础 `gpu_experts_mask`（前 N 专家上卡） | 中 | 无 |
| S2 | Expert placement 与 remap | prefix/frequency mask、逻辑→物理专家 remap（grouped matmul + checkpoint 加载）、mask 随常驻权重一起 clone（NSA stall 修复） | 中 | S1 |
| S3 | 流式 prefill + depool + 动态常驻 | 流式权重池、slot 分时复用 + reserve、动态常驻 gather、NPU 现转链（MXFP4→int8 NZ）、启动暖机 | 大 | S1、S2 |
| S4 | 单卡部署文档 + 收尾 | `docs/.../ascend-npus/` 单卡异构部署指南、参数说明、杂项 | 小 | S1–S3 |

### 2.2 kvcache-ai/ktransformers（1 个 issue + 3 个 PR）

先开 **issue**：宣布 Ascend 单卡支持，链接 sgl-project RFC，让 maintainer 拍板 llama.cpp 方案
（vendor 进 kt-kernel vs build patch）。

| # | 主题 | 内容 | 规模 | 依赖 |
|---|---|---|---|---|
| K1 | kt-kernel Ascend NPU 后端 | `cpu_backend/ascend_callback_worker.{cpp,h}`、`vendors/ascend_npu.h`、cpuinfer/ext_bindings 接线、CMake/install.sh Ascend 分支、python 侧 experts_base/loader、**llama.cpp C 侧 MXFP4 处置（vendor/patch）** | 中大 | llama.cpp 方案定案 |
| K2 | ARM/K920 CPU 算子 | KML int8/int4 prefill GEMM、llamafile MXFP4 路径（`operators/llamafile/moe.hpp`）、`third_party/llamafile` 2 文件、**新增 MXFP4 算子精度测试（per_commit 模式）** | 中 | K1（可同分支连发） |
| K3 | 文档 + 启动脚本 | `doc/en/DeepSeek-V4-Flash-Ascend.md`（对标 GPU 版文档格式）、`script/` 启动脚本、指明配套 sglang 主线版本 | 小 | K1、K2 合入，S 系列有着落 |

### 2.3 third_party 各仓：零 PR

| 仓 | 处置 |
|---|---|
| ggerganov/llama.cpp | 不提（见 1.3）。K920 预取优化回馈 master 是可选 nice-to-have，不在计划内 |
| kvcache-ai/sglang | 不提。NPU 用户直接用 sgl-project 主线 |
| Mozilla llamafile | 不提。改的是 ktransformers in-tree 副本，随 K2 |
| pybind11 / custom_flashinfer | 零改动 |

### 2.4 顺序总览

```
Phase 0（保全+CI预演，本地）
   │
   ├─► RFC issue（sgl-project）───► S1 ► S2 ► S3 ► S4      （串行 stack）
   └─► issue（kvcache-ai）───────► K1 ► K2 ──────► K3
                                    （K1/K2 与 S 系列并行；K3 等两边都齐）
```

- 两条线大部分并行；唯一硬串行点是 **K3**（文档要写"配哪个 sglang 版本"）。
- S 系列内部 stack 串行；K1/K2 可以同分支连发。
- 合计 **7 个 PR + 2 个 issue**。

---

## 3. 逐阶段执行计划

### Phase 0 — 保全 + CI 预演（本地，不依赖新机器，~2 天）

- [ ] **P0.1 llama.cpp 补丁① 落盘**：C 侧 ~130 行做成 patch 文件提交进本仓（唯一有丢失风险的资产，
      `git submodule update` 一次就没）。
- [ ] **P0.2 补丁② 处置**：`tools/` 里引用子模块 gguf-py 的脚本（`batch_convert_*_mp.py`、
      `ascendc_mxfp4/test_*` 等）改用 pip gguf；子模块工作区改动清零。
- [ ] **P0.3 上游件清单**：逐文件标注 上游/不上游/需清理（p27 标记、内部路径、代号）。
- [ ] **P0.4 CI 预演（ARM 侧）**：K920 本机跑通 `install.sh build` + `run_suite.py --hw cpu --suite default`。
- [ ] **P0.5 CI 预演（x86 编译门）**：笔记本（MateBook KLV-WX9，Whiskey Lake-U）WSL2 + Ubuntu，
      `install.sh --manual` 强制 AVX512=ON、AMX=ON 全量编译——验证 ARM/Ascend 改动被架构 gate 干净、
      x86 构建不破。⚠️ 该机只能覆盖编译门：无 AVX512/AMX、内存 ≤16GB，运行时精度测试会 OOM，
      留给 CI 首跑或后续 x86 服务器。
- [ ] **P0.6 新增 MXFP4 算子精度测试**：把 `p27_cpu_moe_reference_check_mxfp4.py`（cosine 0.99994 对账）
      改写成 upstream 的 `register_cpu_ci` + pytest 模式（随机权重→MXFP4→forward vs torch 参考→阈值），
      x86 上干净 skip；NPU 相关按 placeholder 先例。
- [ ] **P0.7 GPQA 证据包**：整理对外精度口径成 PR 可贴形态（见 §4）。

### Phase B — 46 commit 移植到 sgl-project 主线（最大件，1.5~2 周）

- [ ] **B0** fork sgl-project，钉一个 #31931 之后的 ref 拉分支；46 commit **按功能重组**（非机械 rebase）
      为 S1–S4 四个逻辑系列。Yijie Zhu 的 3 个底座 commit 被主线取代，不再需要。
- [ ] **B1** 对位新结构：老代码在 `models/deepseek_v4.py` 一把梭 → 新结构拆进
      `hardware_backend/npu/dsv4/`、`moe_runner/ascend.py`、quantization schemes；
      MXFP4-CPU + NPU int8 现转链与 upstream 的 NPUW8A8/compressed-tensors schemes 融合。
      kt-kernel 与 sglang 解耦，**不用动**。
- [ ] **B2** 单卡回归（910B/A3，CANN 9.0.0）：
      - GPQA 70.88% 口径不退；decode 19~22.5 tok/s 不退；
      - chunked prefill 从此可开（upstream 已修跨 chunk，验证 + 长 prompt 回归）；
      - **MTP × KT CPU-offload 兼容性**（新收益，也是最难交互点，单独验证项）；
      - 长上下文 needle 中段检索在新底座复测（旧底座 NSA 选块丢中段）。
- [ ] **B3** 环境统一：910B 侧 AscendC 自定义算子在 CANN 9.0.0 重编译验证
      （core-type 坑修法：static + always_inline）。

### Phase C — 投递（1 周 + 评审周期）

- [ ] **C1** 开两个 issue（sgl-project RFC + kvcache-ai），等 maintainer 表态期间完成 C2 准备。
- [ ] **C2** 按 §2 顺序发 PR：每个 PR 过一遍 §4 检查单；sgl-project 侧每个 PR 附本地验证证据
      （GPQA/吞吐/msprof 截图）。
- [ ] **C3** 评审跟进：CI 首跑（需 maintainer 打 run-ci 标签）、review 意见、K3 收尾。

---

## 4. 每个 PR 的出门检查单

**commit 规范**：`[type](scope): msg`（regex 强制）；每 commit `Signed-off-by`（DCO）；无 Co-Authored-By。

**代码清理**：p27/phase/任务N/Fx 标记清零（文件名/函数/日志/注释/文档）；内部路径与代号清除；
print→logging 等脚本规范（参照既有开源脚本标准）。

**CI（凡是碰 `kt-kernel/**` 的 PR）**：x86 编译门本地预演过；存量 AMX/AVX2 套件不退；
新算子带 per_commit 精度测试且 x86 干净 skip。

**精度证据包（对外口径，勿混用）**：
- GPQA-Diamond thinking-off：**70.88%（evalscope 1.9.1）**；evalscope 1.8.1 历史数字（68.99/67.53/68.13）作废勿引。
- 910B 三轮：69.19 / 72.73 / 73.23，mean **71.72% / SD 1.80pp**。
- perf（19~22.5 tok/s、cpu_moe 16ms）出自 A3，accuracy 出自 910B，注明勿混。
- 与 upstream 数字对比时注明口径差异：他们 86.87% 是 **thinking-on + 16 卡 A3 全 NPU**。

---

## 5. 机器分工

| 机器 | 用途 | 状态 |
|---|---|---|
| K920 + 910B（本机） | ARM 侧 CI 预演、MXFP4 算子测试、B2 单卡回归、GPQA/perf 证据 | 现成（B3 需切 CANN 9.0.0 容器） |
| 笔记本 MateBook（WSL2） | x86 编译门 pre-flight（仅编译，强制全 flag） | 需搭 WSL2 |
| 8 卡 910B | upstream 原生路径参照系（已验证 OK；后续可复用于对照实验） | 用户已跑通 |
| x86 AMX 服务器（可选） | 运行时 AMX/AVX2 精度测试本地全过 | 没有则由 CI 首跑兜底 |

---

## 6. 风险清单

1. **B1 移植量是大头**：新旧结构差异大，是重新落位不是 rebase；预算 1.5~2 周。
2. **主线移动快**：全程钉 ref，合并前追一次头。
3. **transformers/hf_hub pin**：曾踩 upstream `@strict` 不兼容；装环境时先确认 pin 组合。
4. **MTP × KT CPU-offload 交互未知**：draft/verify 的 batch 形态对 CPU MoE 提交路径的影响，B2 单列。
5. **compressed-tensors 路径在 upstream DSv4 上验证程度未知**（PR 只测了 ModelSlim）：
   我们两个 checkpoint（A=compressed-tensors、B=ModelSlim）各冒烟一次，有坑早暴露。
6. **run-ci 标签在 maintainer 手里**：RFC 先行建立沟通，避免 PR 挂着没人跑 CI。
