# HANDOFF：在 CANN 9.0.0 环境执行 upstream PR 计划

> 写于 2026-08-12。读者：在 **CANN 9.0.0 机器**上新 clone 本仓、从零接手开源投递工作的
> session（无本机记忆，一切以本文 + 仓内文档为准）。
> 总计划：**同目录 `UPSTREAM_opensource_plan.md`（中）/ `UPSTREAM_opensource_plan_EN.md`（英）**
> ——先通读它，本文只补"怎么开工"和计划里没有的环境/隐性知识。

---

## 0. 使命一句话

把单卡 910B/A3 + CPU 异构跑 DeepSeek-V4-Flash 的工作开源到 upstream：
sglang 侧改动投 **sgl-project 主线**（S1–S4，四个 PR 串行 stack），
kt-kernel/CPU 侧投 **kvcache-ai/ktransformers**（K1–K3）。共 7 PR + 2 issue。
路线依据：sgl-project 主线 NPU DSv4 已就绪（PR #28980 MTP、#31931 chunked-prefill/双流），
且已在 8 卡 910B 上实测验证 OK（2026-08）。**不投 kvcache-ai/sglang fork，不给
llama.cpp/llamafile/pybind11/custom_flashinfer 提任何 PR。**

## 1. 新机器开工步骤（按顺序）

```bash
# 1) clone 主仓(分支 dsv4_one_card_dev)
git clone -b dsv4_one_card_dev <本仓地址> ktransformers-AK && cd ktransformers-AK

# 2) 子模块:只需 sglang(含我们全部 46 个 commit 的历史)。
#    .gitmodules 里 sglang 指向 git@github.com:wenxuewuhd/sglang-dsv4.git(SSH);
#    没配 SSH key 就先改成 https 再 update:
#    git config submodule.third_party/sglang.url https://github.com/wenxuewuhd/sglang-dsv4.git
git submodule update --init third_party/sglang

# 3) llama.cpp 子模块:init 后【必须】套回 MXFP4 补丁(否则 kt-kernel CPU MoE 读不了 MXFP4 GGUF):
git submodule update --init third_party/llama.cpp
cd third_party/llama.cpp
git apply ../../doc/zh/dsv4_single_npu/patches/llamacpp-b3173-mxfp4-cside.patch
cd ../..
#    同目录 llamacpp-b3173-ggufpy-DEPRECATED.patch 是废弃件,【不要】套
#    (只服务 tools/ 老脚本;生产 loader 用 pip gguf>=0.17,自带 MXFP4)。

# 4) pybind11 需要则 init;custom_flashinfer 是 CUDA 件,NPU 机器不用。
```

⚠️ 在任何已套补丁的工作区里，**永远不要裸跑 `git submodule update`**（会把 llama.cpp
补丁洗掉）；要更新 sglang 就指名 `git submodule update third_party/sglang`。

## 2. 该环境的定位与任务顺序

CANN 9.0.0 + torch_npu 2.10.0 正是 sgl-project 主线的要求 ⇒ 这台机器是
**Phase B（46 commit 移植 + 单卡回归）的主战场**。顺序：

1. **先补 Phase 0 未完项**（对照总计划 §3 勾选）：P0.2（tools 脚本改用 pip gguf，
   弃 gguf-py 补丁）、P0.3（上游件清单）、P0.6（MXFP4 算子测试改写成 upstream
   per_commit 模式）、P0.7（GPQA 证据包）。P0.1（llama.cpp 落盘）**已完成**——
   就是你在步骤 1-3 里套的那个 patch。P0.4/P0.5 CI 预演可与 Phase B 并行。
2. **Phase B**：fork sgl-project → 钉 #31931 之后的 ref → 46 commit 按功能重组为
   S1–S4 → 对位新结构（`hardware_backend/npu/dsv4/`、`moe_runner/ascend.py`、
   quantization schemes）→ 本机单卡回归（§4 验收线）。
3. **Phase C**：先开两个 issue（sgl-project RFC + kvcache-ai），再按
   S1→S2→S3→S4、K1→K2→K3 发 PR（K 线可与 S 线并行，K3 最后）。

我们的 46 个 commit 在哪：sglang 子模块 `dsv4_release` 分支，
底座 = sgl-project `6d79c6099`（2026-04-09），`git log 6d79c6099..HEAD` 即全系列
（其中 3 个是 Yijie Zhu 的老底座 commit，移植时被主线取代、直接丢弃）。

## 3. 硬性规范（每个 commit / PR 出门前过一遍）

- commit：`[type](scope): msg`（upstream regex 强制）+ 每条 `Signed-off-by`（DCO）；
  **不带 Co-Authored-By、不带任何 AI 署名**。
- 清理：p27/phase/任务N/Fx 等内部标记**全部清零**（文件名/函数/日志/注释/文档）；
  内部路径、机器名、代号不出现；脚本 print→logging 等规范同既有开源标准。
- upstream CI（碰 `kt-kernel/**` 必触发）：x86 runner 跑 `install.sh build` +
  `test/run_suite.py --hw cpu --suite default`。⇒ ARM/Ascend 新代码必须被 CMake/架构
  检测干净 gate 掉，**x86 构建不能破**；新算子照 `test/per_commit/` 模式带精度测试
  （随机权重 vs torch 参考、相对误差阈值，参考 AMX int8 的 <0.05），x86 上干净 skip；
  NPU 测试照 `test_cuda_placeholder.py` 先例放占位，真跑留本地作 PR 证据。
- 对外数字口径（勿混）：GPQA-Diamond thinking-off **70.88%（evalscope 1.9.1）**；
  910B 三轮 mean **71.72%/SD 1.80pp**；perf 19~22.5 tok/s 出自 A3。
  evalscope **1.8.1 的历史数字全部作废**（其 GPQA 适配器删方括号毁 15/198 题，
  详见本目录 REPORT/REPRODUCE 文档）。对比 upstream 的 86.87% 时注明口径：
  他们是 thinking-on + 16 卡 A3 全 NPU。

## 4. Phase B 单卡回归验收线（本机跑）

| 项 | 验收 | 工具 |
|---|---|---|
| 精度 | GPQA off ≥ 70.88% 口径不退（evalscope 锁 1.9.1，脚本已强制） | `script/dsv4_single_npu/1_serve.sh` + `2_gpqa_5x.sh` |
| 吞吐 | decode 稳态 19~22.5 tok/s 不退；warm+median 报数 | `tools/p27_decode_timing.py`（自带 1 发暖机；`--big --runs 3` 测 4K prompt） |
| chunked prefill | 跨 chunk 长 prompt 不崩且答案正确（主线已修；fork 里 c8063ed 也有一版修复，接手后先确认其性质：绕过还是根治，与主线方案对比后取舍） | `tools/p27_curl_long_prompt_sweep.sh` |
| MTP×KT | MTP 开启时 CPU-offload 路径正确性+收益（全新验证项，最难交互点） | 移植后新增 |
| 双格式 | checkpoint A（compressed-tensors）与 B（ModelSlim W8A8）各冒烟一次 | upstream PR 只测过 ModelSlim，A 有坑要早暴露 |

## 5. 隐性知识（不在总计划里、但会咬人的）

1. **测量纪律**：冷服务器读数假性偏低（mmap 惰性缺页，off_cpu 差 2×）——先打暖机
   请求再测；同一结论必须同窗口配对对照，不跨窗口比数。
2. **boot 间 greedy 非确定是底层固有**（MXFP4 kernel 数值微扰 + greedy 临界分叉）：
   精度回归看 GPQA 分数/连贯性/cosine，**不要**按"两次 boot 逐 token 一致"验收。
3. **AscendC 自定义算子在 CANN 9.0.0**：8.5.0 能跑的核 9.0.0 可能报
   `Get kernel function failure`（拆分核函数破坏 core-type 推导）——修法
   static + always_inline；本仓算子已带修复，重编译后真卡跑一遍再信。
4. **kt_kernel 导入**：报 "kt_kernel is not installed" 先查 libhwloc15 是否随容器
   重启丢了；单独跑脚本报 No module named kt_kernel 是包名没注册
   （`kt-kernel/kt_kernel` 须为指向 `python/` 的软链，serve 脚本自带该闸门）。
5. **sglang 移植时的已知修复别丢**：mask 必须随常驻权重一起 clone（否则每次 prefill
   重写 mask 触发 weight-region flush、NSA 被拖慢 ~30%）；流式 prefill 需启动暖机
   （否则 CPU MoE 冷、decode 掉 1/3）；depool 满 context 需 reserve 流式 slot
   （plain ND，不能 NZ format_cast）。这些都在 46 commit 里，重组时对号入座。
6. **hf_hub/transformers pin**：sgl-project 主线曾与 hf_hub 1.12.0 `@strict` 冲突,
   装环境先确认 pin 组合再大规模装依赖。
7. **run-ci 标签在 maintainer 手里**：kvcache-ai 的 PR 不打标签 CI 不跑——RFC/issue
   先行沟通，别让 PR 干挂。
8. 长跑服务在远程会话里要么前台跑要么 `DETACH=1`（setsid），交给后台工具拉会被回收。

## 6. 本目录文件地图

| 文件 | 用途 |
|---|---|
| `UPSTREAM_opensource_plan{,_EN}.md` | 总计划（PR 清单/顺序/阶段/检查单/风险） |
| `patches/llamacpp-b3173-mxfp4-cside.patch` | llama.cpp C 侧 MXFP4（**必套**；K1 时 vendor 进 kt-kernel 或转 build patch） |
| `patches/llamacpp-b3173-ggufpy-DEPRECATED.patch` | 废弃备份，不套；P0.2 完成后可删 |
| `gpqa_accuracy_align_REPRODUCE.md` 等 GPQA 文档 | 精度口径与复现方法（证据包素材） |
| `../../script/dsv4_single_npu/1_serve.sh`、`2_gpqa_5x.sh` | 拉服务/GPQA（含全部环境闸门） |
| `../../tools/p27_decode_timing.py` | 吞吐/TTFT 测量 |
