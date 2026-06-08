# DeepSeek-V4-Flash 单卡 NPU + KT MoE：全量 Patch 包说明
 
本目录把 **相对开源主仓基线** 的全部工程改动导出为可 `git apply` / `git am` 的 patch 系列，便于在公开 GitHub 上从零复现 **DeepSeek-V4-Flash 单卡 Ascend NPU + KT CPU MoE** 推理路径。
 
> 关联文档（打完全部 patch 后）：  
> [Handoff](../../doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md) · [Phase0/1 复现手册](../../doc/zh/Phase0_Phase1_变更记录与复现手册.md) · [Profiling 路线图](../../doc/zh/DeepSeek-V4-Flash_NPU_KT_Profiling_优化路线图.md)
 
---
 
## 一、项目背景
 
**KTransformers（ktransformers-AK）** 在 CPU–NPU 异构服务器上跑 **DeepSeek-V4-Flash**（671B-class MoE）：
 
| 维度 | 说明 |
|------|------|
| **硬件** | 单卡 **Atlas 910B（64GB HBM）** + **鲲鹏 920**（8 NUMA，~1.5TB DRAM） |
| **模型** | DSv4-Flash：43 个 MoE 层，每层 **256 routed experts，top-6**；NPU 侧 **W8A8**（compressed-tensors） |
| **痛点** | 全量 expert 无法都放 NPU；需把「冷」expert 卸载到 CPU |
| **思路** | **Hybrid MoE**：少量热 expert 留 NPU，其余走 **kt-kernel**（llamafile / NEON SDOT）在 CPU 上算 GGUF |
 
主线 **不** 走已删除的 `archive/ktransformers` 本地 chat，而是：
 
```bash
export REPO=/path/to/ktransformers-AK
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-W8A8
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel/python"
cd "$REPO/kt-kernel" && CPUINFER_USE_ASCEND_NPU=1 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=0 bash "$REPO/tools/p27_launch_ds4flash_npu.sh"
```
 
**SGLang 基线选型**：由 `kvcache-ai/sglang` 的 CUDA 向 fallback 改为 **`iforgetmyname/sglang@dsv4_release`**（Ascend DSv4 原生算子），再叠 **5 个 KT patch** 到公开 fork `wenxuewuhd/sglang`。
 
---
 
## 二、当前实现方案（架构摘要）
 
```
┌─────────────────────────────────────────────────────────────┐
│  SGLang launch_server (p27_launch_ds4flash_npu.sh)          │
│  --device npu --kt-method LLAMAFILE --kt-num-gpu-experts 32 │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────────┐
│  NPU 主干 (sglang)   │           │  CPU MoE (kt-kernel)     │
│  Attention/Dense/    │           │  per-layer GGUF Q8_0     │
│  Gate + 前 32 expert │◄─graph───►│  experts_base.py         │
│  W8A8 grouped matmul │  callback │  ascend_callback_worker  │
└─────────────────────┘           └─────────────────────────┘
```
 
| 组件 | 路径 | 作用 |
|------|------|------|
| NPU 图捕获 + CPU MoE 回调 | `kt-kernel/cpu_backend/ascend_callback_worker.*` | NPUGraph replay 时插入 CPU expert 计算 |
| KT MoE Python | `kt-kernel/python/experts_base.py` | pinned buffer、deferred 提交、G 项去掉热路径 `.item()` |
| SGLang KT 封装 | `third_party/sglang/.../kt_ep_wrapper.py` | `--kt-weight-path` 按 `{layer_idx}` 加载 43 层 GGUF |
| NPU stream 同步 | `third_party/sglang/.../kt_accel.py` | CUDA↔NPU event/stream 抽象 |
| 启动脚本 | `tools/p27_launch_ds4flash_npu.sh` | hybrid **N=32**，graph on，chunked prefill 2048 |
 
**性能基线**（graph on，hybrid N=32，`deferred=0`）：decode 约 **3.6–3.7 tok/s**。
 
**已完成 vs 待做**（clean 阶段不改推理逻辑）：
 
| 状态 | 项 |
|------|-----|
| ✅ | Phase 0–2 kt-kernel NPU 编译、43 层 GGUF、N=0/N=32 e2e、NPUGraph + callback |
| ✅ | 优化 G：热路径去掉 `.item()`（~3.65 tok/s 验收） |
| ✅ 代码已接线 | 优化 A：hot expert mask / `--kt-expert-placement-strategy` |
| ⏳ | 优化 D/E/I、GGUF 加载加速、mxfp4、EPLB 动态 mask |
 
---
 
## 三、基线（Baseline）
 
| 仓库 | 基线 commit | 说明 |
|------|-------------|------|
| **ktransformers-AK 主仓** | `d7b5b49`（`[release]: 0.6.2.post1`） | 上游结构重构后 release；**此前无** DSv4 NPU 单卡工程 |
| **third_party/sglang** | `298193eb3` @ `iforgetmyname/sglang@dsv4_release` | Ascend DSv4 公开基线 |
| **+ KT patches** | → `68a0bce65` @ `wenxuewuhd/sglang@dsv4_release` | 5 个 patch，**必须**在基线上叠加 |
| **third_party/llama.cpp** | `a94e6ff`（tag **b3173**） | 公开 pin |
| **custom_flashinfer / pybind11** | 上游子模块 | 无额外 patch |
 
主仓逻辑终点 commit：**`3340670`**（`d7b5b49..3340670` 共 13 个 commit）。
 
---
 
## 四、目录结构
 
```
tools/kt_dsv4_npu_patches/
├── README.md              # 本文件
├── apply_series.sh        # 按 SERIES 顺序应用 main_repo / llama_cpp
├── split_patches.py       # 从 _originals/ 重新生成 ≤80KiB 分片
├── main_repo/
│   ├── SERIES             # 193 个分片严格顺序（含 WIP）
│   ├── 0001-…-p01.patch   # 逻辑 commit 0001 的第 1 片（≤80KiB）
│   ├── …                  # 0001～0013 + WIP-open-source-cleanup-* 
│   └── _originals/        # 未拆分完整 patch（备份）
├── sglang/                # 5 个 KT commit（均已 <80KiB，无需再拆）
│   ├── 0001～0005.patch
│   └── README.md
├── llama_cpp/             # 可选 1 patch（2.4KiB）
│   └── SERIES
└── _optional_large/       # Phase0 含 KML 二进制的完整 patch（~296MB，一般不用）
```
 
**分片命名规则**：`{逻辑名}-p{NN}.patch` 或 `WIP-…-p{NN}.patch` 表示同一逻辑变更被 `split_patches.py` 切成多片；**必须按 `SERIES` 顺序**，不可只打 `-p01`。
 
---
 
## 五、主仓 `main_repo/`：13 个逻辑 commit + WIP
 
### 5.1 逻辑 commit 一览（`0001`～`0013`）
 
| 逻辑 patch | Git commit | 含义 | 关键改动 |
|------------|------------|------|----------|
| **0001** | `e95c561` | gitignore 与 KML 预置 | 忽略 `model/DeepSeek-R1-GGUF`；引入 `third_party/kml` 与 kt-kernel KML GEMM 骨架（SLIM 版不含 zip/rpm 二进制） |
| **0002** | `b97b298` | **Phase 0 done** | `kt-kernel` Ascend NPU vendor（`ascend_npu.h`）、CMake/llamafile 编译适配、`script/launch_ds4flash_sglang.sh`、容器脚本等 |
| **0003** | `062b5bc` | **Phase 1.2** | W8A8→GGUF 转换工具、`phase12_llamafile_moe_smoke.py`、kt-kernel MoE llamafile 路径、Handoff 初稿 |
| **0004** | `ca5fa46` | **Phase 2-B** | 多 GGUF、`LlamafileMoEWrapper` 按路径缓存、`{layer_idx}` 占位符展开、Phase0 手册 §2.3a |
| **0005** | `6ede0ed` | sglang 子模块 + P2.7 启动 | bump `third_party/sglang`；首版 `p27_launch_ds4flash_npu.sh`；Handoff 大幅补充 |
| **0006** | `833ff2c` | ds4-npu 工具链 | `.gitmodules` 调整、权重 dtype 检查脚本、Handoff 变更清单 |
| **0007** | `19d64a6` | 子模块指向内网 fork | sglang/llama.cpp → codehub（**WIP patch 会改回 GitHub 公开 URL**） |
| **0008** | `eb1c3c7` | **P2.7 e2e F1–F3** | 单卡 NPU+KT 端到端；`experts_base.py` / `moe.hpp`；43 层 GGUF；N=0/N=32 验证；chunked_prefill 越界修复 |
| **0009** | `29c082e` | **NPU Graph + ACL callback** | `ascend_callback_worker`；graph capture 期间 CPU MoE；sglang `kt_ep_wrapper` graph host callback |
| **0010** | `e23be3c` | **moe_microbench** | `tools/moe_microbench/` 独立工程：W8A8 grouped GEMM、msprof D4/D5 |
| **0011** | `5948172` | **attn_microbench** | `tools/attn_microbench/`：SWA/CSA/HCA decode profiling |
| **0012** | `2fadd64` | Profiling 文档 + Q8 修复 | 删除 bundled 权重；decode profiling runbook；`moe.hpp` Q8_0 GEMM 部分修复；P27 容器手册 |
| **0013** | `3340670` | **N=32 生产默认** | `p27_ensure_kt_kernel.sh` 对齐；Q8 GGUF 默认；`p27_build_hot_expert_mask.py`；sglang hot expert placement |
 
**分片数量**（`SERIES` 行 1–31）：0001×4 + 0002×4 + 0003×2 + 0004×1 + 0005–0007 各×1 + 0008×2 + 0009×1 + 0010×4 + 0011×5 + 0012×4 + 0013×1 = **31 片**。
 
> **注意**：0005–0013 中 `third_party/sglang` 的 diff **仅是子模块 SHA**；SGLang **源码**改动见下文 **`sglang/0001～0005`**。
 
### 5.2 WIP：`WIP-open-source-cleanup-uncommitted`（逻辑 1 组 → 162 片）
 
**性质**：开源 **clean** 阶段、**尚未单独 commit** 的工作树 diff（`git diff HEAD`），**不是** DSv4 功能开发。
 
**作用**：把仓库收成可推 GitHub 的公开形态。
 
| 类别 | 内容 |
|------|------|
| 配置 | `.gitignore`（msprof/acsend/构建产物）；`.gitmodules` → 公开 GitHub URL |
| 文档 | `CLEANUP_REPORT.md`；README DSv4 Quick Start；R1 Wiki / Profiling 路线图迁入 `doc/zh/` |
| 脱敏 | Handoff proxy 改占位符；`sglang_dsv4_ascend_cann850.sh` 等 |
| 删除 | 整目录 **`archive/`**（历史 KT 集成树）——占 WIP 绝大部分行数 |
| 其它 | `p27_launch_*.sh` 路径占位符；`doc/assets/WeChatGroup.png`；`tools/sglang_kt_patches_public/` |
 
**分片**（`SERIES` 行 32–193，共 **162 片**）：
 
| 片号 | 大致内容 |
|------|----------|
| **p001** | 核心 clean：gitignore、gitmodules、CLEANUP_REPORT、README；`archive/` 删除开始 |
| **p002～p006** | 继续删 `archive/csrc/balance_serve/kvc2/`、`sched/` 等 |
| **p007～p160** | 删 `archive/ktransformers`、`kt-sft`、`third_party` 等剩余树 |
| **p161～p162** | 非 archive 收尾：`doc/zh/*` 脱敏、patch 工具目录、子模块指针等 |
 
**是否必须打**：若你已在主仓 **直接 commit** 了 clean 改动，**可跳过**整段 WIP；若要从 `3340670` 手动复现公开树，则在 `0013` 之后按 `SERIES` 打完 p001～p162。
 
---
 
## 六、SGLang `sglang/`：5 个 KT patch
 
基线：`298193eb3` @ `iforgetmyname/sglang@dsv4_release`  
目标：`68a0bce65` @ **`wenxuewuhd/sglang@dsv4_release`**（用户已 push）
 
| Patch | Commit | 含义 | 关键文件 |
|-------|--------|------|----------|
| **0001** | `0ddd0e12d` | NPU Triton 兜底 | Triton 不可用时 KV/`req_to_token` 走 torch；KT EP NPU 适配 |
| **0002** | `c99920aee` | P2.7 KT EP 单卡子集 | `kt_ep_wrapper.py`：`{layer_idx}` GGUF、`gpu_experts_mask`、submit/sync/merge |
| **0003** | `db577d2e8` | **Graph host callback** | capture 期间通过 NPU stream 调 CPU MoE（hybrid 生产必备） |
| **0004** | `38bda36e8` | Decode profiler | 一次性 decode profiler；SparseAttn `T=1` squeeze guard |
| **0005** | `68a0bce65` | **Hot expert placement** | `kt_expert_masks.py`；prefix/frequency mask；`--kt-expert-placement-strategy` |
 
```bash
git clone https://github.com/iforgetmyname/sglang.git /tmp/sglang
cd /tmp/sglang && git checkout dsv4_release   # 应在 298193eb3
git am /path/to/ktransformers-AK/tools/kt_dsv4_npu_patches/sglang/*.patch
# 推送到 wenxuewuhd/sglang@dsv4_release
```
 
---
 
## 七、llama.cpp `llama_cpp/`：可选 1 patch
 
| Patch | Commit | 含义 |
|-------|--------|------|
| **0001** | `ac315ccc`（在 b3173 之上） | NumPy 2 下 `GGUFReader` 字节序视图兼容 |
 
**默认**：子模块 pin **`a94e6ff`（b3173）**，**不**打此 patch。仅当读 GGUF 报错时再 `git am`。
 
---
 
## 八、推荐应用顺序
 
```text
1. 主仓 checkout d7b5b49
2. bash tools/kt_dsv4_npu_patches/apply_series.sh main_repo
      → 0001～0013（31 片）+ 可选 WIP（162 片）
3. sglang fork：298193eb3 + am sglang/0001～0005 → push wenxuewuhd/sglang
4. git submodule sync && git submodule update --init
5. llama.cpp checkout a94e6ff（可选 am llama_cpp/0001）
6. cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 python3 setup.py build_ext --inplace
7. ASCEND_RT_VISIBLE_DEVICES=0 bash tools/p27_launch_ds4flash_npu.sh
```
 
一键应用主仓（含 WIP）：
 
```bash
export REPO=/path/to/ktransformers-AK
cd "$REPO" && git checkout d7b5b49
bash tools/kt_dsv4_npu_patches/apply_series.sh main_repo "$REPO"
```
 
若不要 WIP：编辑 `main_repo/SERIES`，删掉 `WIP-open-source-cleanup-*` 行后再跑 `apply_series.sh`。
 
---
 
## 九、不需要单独 patch 的部分
 
| 内容 | 说明 |
|------|------|
| 模型 W8A8 / GGUF | 用户自备路径，不进仓库 |
| msprof 采集产物 | 已 gitignore；按 runbook 重跑 |
| `custom_flashinfer` / `pybind11` | 标准 `git submodule update` |
| sglang 已 push 的 fork | 克隆主仓 + submodule 即可，无需再 am |
 
---
 
## 十、逻辑 patch → SERIES 行号对照
 
| SERIES 行 | 逻辑组 |
|-----------|--------|
| 1–4 | 0001 |
| 5–8 | 0002 SLIM |
| 9–10 | 0003 |
| 11 | 0004 |
| 12–14 | 0005–0007 |
| 15–16 | 0008 |
| 17 | 0009 |
| 18–21 | 0010 |
| 22–26 | 0011 |
| 27–30 | 0012 |
| 31 | 0013 |
| 32–193 | WIP（可选） |
 
---
 
## 十一、维护说明
 
- 重新拆分：`python3 tools/kt_dsv4_npu_patches/split_patches.py`（读各目录 `_originals/`）
- 旧路径 `tools/sglang_kt_patches_public/` 与 `sglang/` 内容相同
- Phase0 含 KML 二进制完整版：`_optional_large/`（~296MB，一般用 SLIM 0002 即可）