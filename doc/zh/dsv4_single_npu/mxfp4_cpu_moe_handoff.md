# Handoff — CPU MoE 换原生 MXFP4（Session D）：搬运字节减半，decode 再提 ~20–25%

> **状态**：开放（Session D 起点）｜**日期**：2026-06-10｜**隔离 worktree**：`/workspace/code/kt-D-mxfp4`
> **基线**：主干 `dsv4_one_card_dev` @ `22aac3d`（decode `--kt-cpuinfer 128` + GEMV prefetch → ~8.5 tok/s client；
> CPU MoE 是 DDR 带宽瓶颈，~55ms/token，详见
> [graph_decode_bandwidth_findings.md](graph_decode_bandwidth_findings.md)——它点名的两条出路之一就是本任务）。

---

## 启动提示词（开新 session 时整段贴）

> 你接手 **DeepSeek-V4-Flash 单卡 NPU 的「CPU MoE 换原生 MXFP4 权重」任务**。
> 现状：CPU offload 的 224 个专家/层用 Q8_0 GGUF（1.0625 B/元素），decode 是纯 DDR 带宽瓶颈;
> DeepSeek-V4-Flash 官方有**原生 MXFP4 权重**（E2M1 nibble + ue8m0 per-32-group scale）。
> 目标：CPU MoE 改吃 MXFP4 GGUF（0.53125 B/元素），**搬运字节精确减半** →
> cpu_moe_wall ~55→~36ms/token，端到端 ~8.5→~10+ tok/s，DRAM 常驻 275→~137GB。
> NPU 侧（attention/shared/32 常驻专家，W8A8）**完全不动**。
>
> **本文（这份 handoff）就是你的完整起点，从 §0 往下读。**
>
> 工作区：`/workspace/code/kt-D-mxfp4`（分支 `mxfp4-cpu-moe`，独立 sglang 分支 `mxfp4-sglang`——
> 但本任务**预计不用改 sglang**；llama.cpp/llamafile 齐全可重编，基线 `.so` 已就位）。
> 启动脚本自动用本 worktree 的 sglang+kt-kernel，**不用 export PYTHONPATH；端口 8014**。
>
> ⚡ **开工状态**：原生 MXFP4 权重正在下载到
> `/workspace/models/DeepSeekV4/DeepSeek-V4-Flash`（git-lfs 渐进式：135B 指针文件 → 真实 shard）。
> **格式核验已完成（2026-06-10，见 §3）**：张量命名/dtype/shape 与
> `MXFP4SafeTensorLoader`（kt-kernel/python/utils/loader.py:1277）完全吻合;分片是**每层专家
> 独占一个 shard**;**layer 16（shard 00018）已下载完整**——直接拿 layer 16 开工 P1–P4
> 单层闭环，**不用等全量**（全量只有 P5 端到端验收才需要）。
>
> 纪律：任何性能/正确性结论用真实权重 + 输出非零校验；只杀自己 PID/端口、**绝不广播
> `pkill -f sglang.launch_server`**（内联 pkill 还会自杀 shell）；拉服务前 `npu-smi info` 选空卡
> （避开卡 2=别的容器）+ `ss -ltnp | grep :8014`；ISA 红线 R1：**无 SVE/i8mm/BF16 指令**，
> march 固定 `armv8.2-a+fp16+dotprod`，kernel 只能用 NEON `vqtbl1q_s8` + `vdotq_s32`（SDOT）。

---

## 0. 任务与收益

| 项 | Q8_0（现行） | MXFP4（目标） |
|---|---|---|
| 字节/元素 | 1.0625（34B/32 块） | **0.53125（17B/32 块：1B e8m0 scale + 16B nibbles）** |
| 单专家 | 26.7 MB | **13.4 MB** |
| 最恶劣每层（top-6 全 CPU） | 160 MB | **80 MB** |
| DRAM 常驻（43 层） | ~275 GB | **~137 GB**（顺手给 C 线长上下文腾 ~138GB） |
| cpu_moe_wall/token（@128 线程） | 55.1 ms | **~36 ms（估）** |
| 端到端 decode | ~8.5 tok/s | **~10–10.5 tok/s（估，+20–25%）** |

> 估算依据（bandwidth findings §4）：每层 1.28ms ≈ 0.89ms 字节搬运 + 0.39ms 固定开销;
> 字节减半只砍前者。**不是 2×**——固定开销与 NPU 侧 ~50ms 不随字节缩。
> 精度：MXFP4 是官方发布的量化，转 GGUF 全程 **bit 级无损 repack**（不是再量化），
> 比"W8A8→Q4 双重量化"干净得多。CPU 专家 MXFP4 + NPU 专家 W8A8 混用没问题（各专家独立近似
> 同一母权重），照例对账收口。

## 1. 工作区 / 隔离（已建好，别碰别的目录）

| 项 | 值 |
|---|---|
| 仓库 | `/workspace/code/kt-D-mxfp4`（git worktree，父分支 `mxfp4-cpu-moe`，自主干 `22aac3d`） |
| sglang | 独立 clone：`third_party/sglang`（分支 `mxfp4-sglang` @ `456687a0f`）。**本任务预计零改动**（LLAMAFILE wrapper 只传 GGUF 路径，量化类型从 GGUF header 自取） |
| llama.cpp | 平拷的 b3173 checkout（含坑④ NumPy2 patch 的 apply 态，gguf-py 可直接 import；`.git` 指针文件失效属预期，B/C 同款） |
| kt-kernel | llamafile vendored 齐全，基线 `.so` 已拷入 `kt-kernel/python/`。**本任务要改 C++，必须重编**（见 §5） |
| 端口 | **8014**（A=8000/8011，B=8012，C=8013） |
| 提交 | 父仓 Python/C++/工具/文档 → `mxfp4-cpu-moe`；（万一动了 sglang → `mxfp4-sglang`） |
| 重编 | `cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace` |

## 2. 已有资产（先读，省一半工作量）

1. **`MXFP4SafeTensorLoader`**（`kt-kernel/python/utils/loader.py:1277`）：原生 V4-Flash MXFP4
   checkpoint 的解析器**已写好**——每专家 `{base}.ffn.experts.{i}.w1/w3/w2.weight`
   （I8 `[N, K/2]` nibble-packed E2M1）+ `.scale`（F8_E8M0 `[N, K/32]`），含 ue8m0→bf16 无损位移。
   转换器直接复用它读 checkpoint。
2. **x86 MXFP4 kernel 作语义参考**（`kt-kernel/operators/amx/fp4-moe.hpp`，AVX512 专用编不进
   aarch64）：E2M1 16 值 LUT `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`、nibble 解包顺序（lo/hi interleave）、
   per-32-group scale 语义都在里面，**照它对齐数值定义**。
3. **arm 插入点模板**（`kt-kernel/operators/llamafile/moe.hpp:64` `kt_llamafile_sgemm`）：坑⑧ 的
   修复就是在这里给 Q8_0×Q8_0 dispatch 到 `ggml_vec_dot_q8_0_q8_0` + GEMV prefetch。
   **MXFP4×Q8_0 加同款分支即可**，prefetch 逻辑直接复用。
4. **`LLAMA_MOE_TP` 对权重类型泛化**：buffer 尺寸、激活量化（`from_float` 到 `vec_dot_type`）、
   NUMA TP、P0/P1 加载加速、graph callback 全部经 ggml `type_traits` 走，**注册好新类型后这条线
   不用改**。
5. **上游 llama.cpp（新版）有现成 NEON 实现**：`ggml_vec_dot_mxfp4_q8_0`
   （`kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}`（×0.5 折进 scale），
   `vqtbl1q_s8` 查表 → `vdotq_s32` SDOT → `GGML_E8M0_TO_FP32_HALF(e) * act_scale`）。
   **从上游源码移植**（WebFetch github raw 或 pip 新版 gguf 源码），全程只用 K920 有的指令。
6. **工具链**：`tools/batch_convert_w8a8_layers_mp.py`（多进程按层转换的骨架，加 `--quant mxfp4`
   路径）、`tools/p27_cpu_moe_reference_check.py`（cosine 对账，要加 torch 侧 mxfp4 LUT 反量化
   参考）、bandwidth handoff §5 的隔离微基准方法（`KTMoEWrapper` + 单层真实权重 + norm>0 校验）。

## 3. ✅ 权重格式核验（已完成 2026-06-10，实测事实）

权重目录：`/workspace/models/DeepSeekV4/DeepSeek-V4-Flash`（git-lfs clone，46 shard 渐进下载中）。

| 项 | 实测值 | 与 loader 预期 |
|---|---|---|
| 专家张量命名 | `layers.{L}.ffn.experts.{i}.w1/w3/w2.weight` + `.scale`（**无 `model.` 前缀**） | ✅（loader 探测 stripped 形式） |
| gate/up（w1/w3） | weight `I8 [2048, 2048]`（K=4096 nibble-packed 成 K/2）+ scale `F8_E8M0 [2048, 128]`（K/32） | ✅ |
| down（w2） | weight `I8 [4096, 1024]` + scale `F8_E8M0 [4096, 64]` | ✅ |
| scale 分组方向 | 沿 **K（输入维）**，group=32 | ✅ 与 GGUF 块方向一致（gate/up 沿 hidden、down 沿 intermediate，同 Q8_0 布局） |
| config | `expert_dtype: "fp4"`；非专家权重 FP8 e4m3（NPU 侧不受影响，继续用 W8A8 ckpt） | — |
| 分片规律 | **每层专家独占一个 shard**（如 shard 00018 = layer 16 全部 1536 个专家张量） | 单层转换只读一个文件 |
| 已就绪 | **shard 00018（layer 16）完整**（8+header+data == 文件大小，已校验）；shard 00034（layer 32）同尺寸大概率完整 | **拿 layer 16 开工** |

> 判断某 shard 是否下载完：文件 >135B（非 LFS 指针）且 `8 + header_len + max(data_offsets)
> == 文件大小`（python struct 读前 8 字节 + json header 即可验）。新就绪的层照此自查。

## 4. 工作计划（按依赖排序，单层即可闭环 P1–P3）

| 阶段 | 内容 | 验收 |
|---|---|---|
| P1 类型注册 | vendored ggml（b3173）加 `GGML_TYPE_MXFP4`：`block_mxfp4{uint8 e; uint8 qs[16]}`，`ggml.c` type_traits 表加项（blck=32, size=17, `vec_dot_type=Q8_0`, dequantize_row 供对账）；`loader.py` `GGML_QUANT_SIZES` 加 `(32, 17)`。**enum id 与上游对齐用 39**，C++/Python/转换器三处一致 | 编译过；`GGUFLoader.tensor_info` 能识别 |
| P2 转换器 | mxfp4 safetensors → 按层 GGUF（`dsv4_layer{i}_mxfp4.gguf`）：`MXFP4SafeTensorLoader` 读 → nibble 原样 repack 成 17B 块 + e8m0 scale 直存（**无损，不过 fp32**）。gguf-py（b3173）不认 39 → 本地扩展 enum 或按 raw bytes 写。⚠️ nibble 序（lo/hi 先后）必须与 P3 kernel 的解包约定一致——坑⑩ 同类雷区 | 单层（**layer 16**，shard 00018 已就绪）对账：torch LUT 反量化 GGUF vs `MXFP4SafeTensorLoader` 反量化 ckpt，**逐字节/逐元素相等**（无损所以不是 cosine 是相等） |
| P3 NEON kernel | 移植上游 `ggml_vec_dot_mxfp4_q8_0`（NEON tbl+SDOT 路径 + scalar 兜底）进 vendored ggml-quants；`kt_llamafile_sgemm` 加 MXFP4×Q8_0 分支（复用 prefetch，行距改 17B 块）；重编 `.so` | `p27_cpu_moe_reference_check.py` layer 16：KTMoEWrapper(MXFP4 GGUF) vs torch 参考，cosine ≥ 0.999（激活 Q8 量化是唯一损失源） |
| P4 微基准 | bandwidth handoff §5 隔离微基准：layer 16 真实权重、norm>0 校验，扫 96/112/128/144 线程；**A/B 对照用同层 Q8_0**（`/workspace/models/cache/dsv4_layer16.gguf` 现成） | 单层 ms 接近减半（Q8_0 ~1.41ms→~0.8ms @128）；记录带宽曲线（字节减半后 knee 可能左移，96 也许就够） |
| P5 全量+端到端 | 等全量下载完 → 43 层转换（`--jobs` 多进程）→ `EXTRA_FLAGS`/`KT_WEIGHT` 指向 mxfp4 GGUF 拉服务（端口 8014）→ `KT_DECODE_TIMING=1` 量 cpu_moe_wall | `PORT=8014 bash tools/p27_curl_f2_prompts.sh` 四 prompt 连贯；cpu_moe_wall ~36ms；gen throughput 报数 |
| P6 收尾 | 文档（本文改 findings）+ commit；`KT_DUMMY_CPU_WEIGHTS` 路径确认对新类型可用（loader 注册后天然支持） | 主干合并见 §7 |

> P2/P3 的顺序可换或交错——先有"哪边定义 nibble 序"都行，**两边必须同一约定**，对账工具是裁判。
> 上游 GGUF MXFP4 的块内布局是 `qs[j]` 低 nibble = 元素 j、高 nibble = 元素 j+16（半块交错）——
> 建议直接采上游约定，kernel 与转换器都照它来，省得自创布局。

## 5. 代码地图

- **类型注册**：`third_party/llama.cpp/ggml/src/ggml.c`（type_traits 表）+ `ggml-quants.{c,h}`
  （vec_dot + dequantize_row）。kt-kernel 编译时 include 这棵树（坑②：头文件布局钉 b3173，别动结构）。
- **GEMV dispatch**：`kt-kernel/operators/llamafile/moe.hpp` `kt_llamafile_sgemm`（:64，aarch64
  无 SVE 分支）；decode 走 `forward_one`（:373），prefill 走 `forward_many`——两者经 type_traits
  泛化，注册类型后自动可用。
- **加载**：`kt-kernel/python/utils/loader.py`（`GGML_QUANT_SIZES` :56、`GGMLQuantizationType` :32、
  `MXFP4SafeTensorLoader` :1277）；`llamafile.py` `load_weights`（按 GGUF tensor_info 取类型，泛化）。
- **转换器**：`tools/batch_convert_w8a8_layers_mp.py` + `tools/convert_w8a8_to_gguf_q8_0.py`（骨架参考）。
- **对账**：`tools/p27_cpu_moe_reference_check.py`。
- **拉起**：`tools/p27_launch_ds4flash_npu.sh`（REPO 按脚本位置解析，worktree 内自动用本树）。

## 6. 纪律（硬要求，血泪沉淀）

- 性能/正确性结论只认**真实权重 + 输出非零/对账**；dummy 只做"图能不能跑通"。
- ISA 红线 R1：无 SVE/BF16/I8MM（`smmla`/`usdot`/`ptrue`/`__bf16` 都会 SIGILL）；新 kernel 只用
  NEON `vqtbl1q_s8` + `vdotq_s32`；march 不动。
- 别破坏 Q8_0 现行路径：dispatch 加分支不改原逻辑，回归跑一次 Q8_0 对账。
- 杀进程只用自己 PID / 端口 8014；**绝不** `pkill -f sglang.launch_server`。
- 拉服务前 `npu-smi info` 选空卡（避开卡 2）+ `ss -ltnp | grep :8014`；杀服务 SIGTERM，等 HBM
  回落再重拉（SIGKILL 留孤儿 scheduler 占 HBM）。
- 共享机 load ~400：吞吐数字尽量挑清净窗口，A/B 同窗口对比。

## 7. 合并回主分支

- 父仓改动（ggml 类型 + kernel + loader + 转换器 + 文档）→ 主 checkout
  `git merge --no-ff mxfp4-cpu-moe`，合后主 checkout 重编 `.so`。
- ⚠️ `third_party/llama.cpp` 在主仓是 **submodule**（钉公开 b3173），本 worktree 是平拷目录——
  对 vendored ggml 的改动**要走主仓的 patch 机制**（`tools/kt_dsv4_npu_patches/llama_cpp/` 加
  patch 文件，仿坑④ NumPy2 patch 的做法），不能 commit 进 submodule。合并前把 ggml 改动导出成
  patch 并验证裸 clone + apply 可复现。
- sglang 预计零改动；万一有 → `mxfp4-sglang` 分支出 patch，同 B/C 流程。
- 与 B/C 的边界：本任务只动 CPU 权重格式与 kernel，不碰 submit/sync/overlap 编排（B）与
  流式加载/常驻策略（C）;但 **C 的流式加载将来搬的也是这份 mxfp4 GGUF**（字节减半对 C 直接利好），
  合并次序无硬依赖。
