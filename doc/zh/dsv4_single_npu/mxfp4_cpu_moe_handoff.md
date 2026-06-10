# Handoff — CPU MoE 换原生 MXFP4（Session D）：搬运字节减半，decode 再提 ~20–25%

> **状态**：P1–P4 完成并对账通过；P5（全量+端到端）待全量下载｜**端口改 8020**（不再用 8014）
> ｜**日期**：2026-06-10｜**隔离 worktree**：`/workspace/code/kt-D-mxfp4`

---

## ✅ 进度（2026-06-10，Session D 实测）

| 阶段 | 状态 | 实测结论 |
|---|---|---|
| P1 类型注册 | ✅ | `GGML_TYPE_MXFP4=39`（block_mxfp4{e;qs[16]} blck32 size17，vec_dot_type=Q8_0）。改 vendored ggml.h/ggml-common.h/ggml-quants.{c,h}/ggml.c + kt loader.py + gguf-py。导出 patch `tools/kt_dsv4_npu_patches/llama_cpp/0002-add-ggml-type-mxfp4.patch`（对 pristine b3173 `git apply --check` 通过） |
| P2 转换器 | ✅ | `tools/convert_mxfp4_layer_to_gguf.py`。**nibble 序（坑⑩）实锤**：原生 ckpt 是 **consecutive**（`inference/convert.py`: `stack([low,high]).flatten` → byte i = Kpos 2i/2i+1），上游 GGUF 是 **half-block**（qs[j]=Kpos j / j+16），转换器**逐 32-group 重排 nibble**（非 byte copy），e8m0 scale 字节直存。`tools/verify_mxfp4_layer.py` layer16：GGUF dequant == 原生 dequant **逐元素 bit-exact**。文件 3.42GB（Q8_0 6.85GB 的一半） |
| P3 NEON kernel | ✅ | 移植上游 `ggml_vec_dot_mxfp4_q8_0`（vqtbl1q_s8+vdotq_s32+scalar 兜底）；`kt_llamafile_sgemm` 加 MXFP4×Q8_0 分支（复用 prefetch）。`tools/p27_cpu_moe_reference_check_mxfp4.py` layer16：**cosine=0.999939 / max_rel 1.12%**（唯一损失激活 Q8）。⚠️ 离线对账须 `KT_FORCE_SYNC_SUBMIT=1`（stream-callback 路径在孤立单层调用返回全 0，Q8_0 同样，非 kernel bug；脚本已内置） |
| P4 微基准 | ✅ | layer16 真实权重 A/B（同窗口，norm>0+确定性签名）。@同线程 MXFP4 wall 全面低于 Q8_0；MXFP4 搬运 80.2MB/tok（Q8_0 160.4，正好一半）：<br>96t: MXFP4 1.238 / Q8 1.967 (1.59×)；112t: 1.123 / 1.565 (1.39×)；128t: 1.024 / 1.314 (1.28×)；144t: **0.926** / 1.264 (1.37×)。MXFP4 best 0.926ms≈手册 ~0.8 目标。**knee 反而右移**：字节减半后没那么吃带宽，144t 仍在涨（值得加大线程） |
| P5 全量+端到端 | ✅ | 全 43 层转换（`batch_convert_mxfp4_layers_mp.py`，并发转换曾把 layer9 写截断成 576B→已 catch 单独重转，**收尾务必逐层 audit 文件大小**）。端口 **8020** 卡 6 拉服务，四 prompt 全连贯（Fibonacci+代码/ML 解释/中文 Transformer，无 NaN/乱码）= **nibble 约定全模型级正确性闸门通过**。**cpu_moe_wall 55→~39ms（−29%），decode 吞吐 8.5→~10–11 tok/s（+20~30%）**。KT_PHASE 实测 F：gateup ~0.77ms(67%)/down ~0.36ms(31%)/merge ~98us 单核(8%)/quant ~20us(可忽略) |
| P4.1 线程 A/B | ✅ | 端到端 `--kt-cpuinfer` 128 vs 160：**128 胜**。160 隔离微基准赢(0.93ms/层)，但端到端和 sglang scheduler/NPU callback/tokenizer 抢核(20线程/NUMA 只剩 4/NUMA)→ 吞吐不升反抖(尖峰 63ms)。**serving 保持 128** |

> 复现命令见各工具 docstring。重编 `.so` 需 `apt-get install -y libhwloc-dev`（容器重启会丢，连 import 用的 libhwloc15 也会丢）。
> 拉服务：`NPU_DEVICE_ID=<空卡> PORT=8020 KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' KT_CPUINFER=128 MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 KT_DECODE_TIMING=1 bash tools/p27_launch_ds4flash_npu.sh`
>
> **F-opt Phase 1 已做（2026-06-10）**：merge_results 按 (token,hidden-chunk) 分块，qlen=1 也铺满 pool。
> **merge 98→14.3us/层（7×）**，cpu_moe_wall min 38→33.4ms，吞吐峰值 11.5→12.0 tok/s；微基准 sig 不变+生成连贯。
> 仅 hidden_type blck==1 时分块（块量化回退 per-token）。commit 75ffdbb。
>
> ~~F-opt 剩余：CPU MoE kernel 侧到此基本到头~~ ——**被 2026-06-10 下午的 roofline 复查推翻，还有 2.4×！见下。**

---

## 🔥 Roofline 复查 + 行内预取修复（2026-06-10 下午，5 连实锤）

旧 roofline 两处错误 + 一个被掩盖的真瓶颈（复现工具 `tools/bw_probe_mlp.c`、`tools/kernel_compute_probe.c`）：

1. **DIMM 实配**（dmidecode）：每 NUMA 只插 **3/4 通道**（24 DIMM/32 槽），DDR4-**3200** → 真 spec **614 GB/s**（非 751）。
2. **清净窗口纯读探针**（load~130）：单 NUMA 24t = **65.7 GB/s**（修正 spec 的 85.5%，正常效率）；全机 192t = **442.8 GB/s**。旧"上限143/清净45"皆 load-400 污染测量。
3. **cache-resident kernel 算力**：MXFP4 dot 2.80 GB(w)/s/核（0.93 cyc/B）、Q8_0 6.69（0.39）——算力不是墙。
4. **真 GEMV 探针**（真 kernel+DRAM 流、无 KT 调度）@128t = **108 GB/s ≈ MoE 实测** → KT 调度无罪，**瓶颈在 kernel 行内**。
5. **+行内软预取(512B)+向量累加** @128t = **349.7 GB/s（3.2×）**。根因：**TSV110 硬件预取器跟不上 GEMV 的低密度 load 流**，行内 34 线 miss 延迟全暴露；外加每 2 块 vaddvq+标量累加的串行 reduce 链。

**修复**（进 `ggml_vec_dot_mxfp4_q8_0` NEON 路径，即 patch 0002；Q8_0 路径未动）：
主循环 `__builtin_prefetch(+512B)` + 双 float32x4 FMA 累加链（行尾才 horizontal）。

| 闸门/指标 | 修复前 | 修复后 |
|---|---|---|
| 微基准 layer16 @128t | 0.952ms / 84 GB/s | **0.402ms / 199.5 GB/s（2.4×）** |
| 确定性 sig | — | **一字不差** |
| cosine 对账 | 0.999939 | **0.999939（不变）** |
| cpu_moe_wall（load~150 窗口） | 33.4ms | **median 26.8 / min 18.0ms** |
| decode 吞吐 | 11-12 tok/s | **12.4-13.3 tok/s** |
| f2 四 prompt | 过 | 过（贪心分叉属预期，质量噪声双向） |

> token 级注意：FMA 重排 ≈1e-7 数值差会让贪心解码在近平局 token 分叉，输出与旧 kernel 非逐字相同——判据是连贯性+cosine，非 token 相等。
> 下一步若再压：①同款预取改造 Q8_0 路径（动 ggml 原函数，需单独回归）②预取距离扫描（512B 未调优）③NPU 侧成为新主导（~45ms），CPU↔NPU overlap（B 线）价值更大了。
>
> **❌ down 短行 nrc=2 已试，负结果（2026-06-10）**：双行 vec_dot（共享激活、双权重流、行级数值与单行逐位一致）@128t **中性**
> （median 0.402→0.421ms，min 持平 199.7 GB/s）。机理：任务内行连续，行内 +512B 预取在 136B down 行上天然**跨行**预取 ~3.7 行，
> 短行开销已被预取修复顺带覆盖。按"无实测收益不合并复杂度"已回退（moe.hpp 留 Note 注释防重蹈）。
>
> **❌ down 跨专家 chunk 预取也试了，负结果（2026-06-10 晚）**：down 任务专家循环里预取下一专家 4.4KB chunk 头 1KB
> （+首专家在 zero-init 时预取）→ down 113→117µs、总 median 0.396→0.421（min 持平 0.369），回退。
> **三连排除后的结论**：down 的 ~30%/层 deficit（29.6 vs gateup 42.8 GB/s/NUMA，探针⑤ hop=1 复现 -27%）
> 既不是行短（kcp K=256 仅 -5%）、不是每行调用开销（nrc=2 减半调用→0 收益）、也不是 chunk 头 ramp
> （跨专家预取→0 收益；且 ARM PRFM 在 TLB miss 时会被静默丢弃，跨远地址预取可能根本没发出去）。
> 剩余假设指向**内存系统层**（chunk 边界 DRAM row-activate、页表走位、预取器流跟踪上限），
> 验证需 PMU 计数器（容器内大概率不可用），上限收益 ~1.5ms/token——投入产出比不成立。
> **D 线 kernel 侧三次试探（nrc=2、跨专家预取、+早前 nrc=2 短行）后正式收口。**

---

## 📐 终版 roofline 分解（2026-06-10 晚定稿，预取 kernel 后）

服务端口径：256 专家/层、32 常驻 NPU（prefix）、top-6 → CPU 平均承接 **5.25 专家/层/token** = 70.3MB/层 = **3.02GB/token**。

### cpu_moe_wall ≈ 27ms（median，load~150）的分解

```
 cpu_moe_wall 26.8ms median / 18.0 min
 ├─ on_cpu  ~1.8ms        host(D2H/routing/H2D)
 └─ off_cpu ~25ms median / ~16.5 min
     物理底  ~9.0ms        3.02GB ÷ 42 GB/s/NUMA(kernel形态上限,实验⑤)
     down 流重启 ~1.5ms    29.6 vs 42.8 GB/s/NUMA(-30%);三个廉价修法已证无效
     dispatch+skew ~4.6ms  每层 forward 外 106µs(fork-join 派发+max-of-8+sync)
     quant+merge  ~1.5ms   已最小化(merge 并行后 14µs/层)
     邻居噪声  +8.5ms       median−min;独占机器待验证(见下方测试方案)
```

### Phase 实测（实验③，n=4300/层，预取 kernel，@128t）

| 相位 | 每层/tp | 折算带宽 | 判定 |
|---|---|---|---|
| gateup | 156µs | 42.8 GB/s/NUMA | **≈探针形态上限 42.1（吻合 2%）—— 无泡沫** |
| down | 113µs | 29.6 GB/s/NUMA | -30%；探针⑤ hop=1 复现（31.3 vs 42.1，-27%），三连排除后归因内存系统层 |
| quant / merge | 21 / 15µs | — | 可忽略 |
| forward 外 | ~106µs | — | numa-job 派发 + 8-NUMA skew + sync，结构性 |

### 实验索引（复现工具全在 tools/）

| # | 实验 | 工具/命令 | 关键数 |
|---|---|---|---|
| ① | DIMM 实配 | `dmidecode -t 17` | 24/32 槽、3/4 通道、DDR4-3200 → 真 spec **614 GB/s** |
| ② | 加载延迟 | `bw_probe_mlp lat 0 512`（±16 线程自造邻居） | 147→195ns(+33%) → 单核带宽 ∝ MLP/延迟 |
| ③ | phase 分解 | 微基准 + `KT_MOE_PHASE_TIMING=1` iters≥4300 | 上表 |
| ④ | 时序对照 | v3 探针 ×6 + loadavg | load 91→151 带宽稳定 375-387 → **loadavg 不是噪声代理，看邻居打不打 DDR** |
| ⑤ | 形态探针 | `bw_probe_mlp gemv <cpus> <mb> <s> <v1/2/3> [K] [hop]` | K4096顺序 42.1 / K256顺序 42.8 / K256+hop 31.3 GB/s @16t |

### G1（邻居噪声）机理链

邻居 DDR 流量 → 控制器排队 → 延迟↑(实验②) → 单核带宽↓（outstanding miss 数硬件定死）→ **max-of-8 放大**（每层等最慢 NUMA，43 层累计上尾）→ median 被抬 8.5ms。历史佐证：load-400 时代全机只读上限 121-143 vs 安静窗口 375-442（3× 摆幅）。

---

## 🧪 独占机器测试方案（待窗口，照此执行）

**目的**：验证 8.5ms median−min 抖动归因（邻居 vs 自家 skew），并解锁更高线程数。

1. **窗口确认**（5min）：`bw_probe_mlp bw "0-191" 1 8 128 2` 应 ≥400 GB/s；`lat 0 512` 应 ~130-147ns；时序 ×3 稳定。
2. **微基准基线**（5min）：layer16 @128t ×3 次——共享机记录 median 0.38-0.42 / min 0.37；独占预期 min 不变、**median 收敛到 min+5%**。
3. **端到端主测**（30min）：8020 拉服务（mxfp4、cpuinfer 128、`KT_DECODE_TIMING=1 KT_MOE_PHASE_TIMING=1`），跑 ≥500 token 长生成，统计 cpu_moe_wall **median/p95/min**。
   - **判定**：median→17-18ms 且 p95−min <3ms ⇒ 8.5ms 归因邻居成立，回收兑现；
   - 若仍有 ±4-5ms 散布 ⇒ 自家 skew 占大头（fork-join/调度），邻居归因部分推翻——届时 dispatch+skew 那 4.6ms 的结构改造（任务编排重写）的赔率就值得重估。
4. **线程重扫**（20min）：同窗口 A/B `KT_CPUINFER` 128/144/160——共享机上 160 输在和 sglang 抢核；独占下抢核压力不同，160 可能翻盘（微基准 160 是 -13%）。
5. **顺手**：四 prompt 连贯性回归；若 NPU 侧也要量，KT_DECODE_TIMING 的 off_cpu/on_cpu 拆分已够。
   预期总收益（若归因成立）：cpu_moe_wall ~27→**17-18ms 稳定**，decode ~13→**15-16 tok/s**。

### ✅ 实测结果（2026-06-10 晚独占窗口，照单执行）

**窗口质量**：load 10.4、全机纯读 **521.7 GB/s**（新纪录，修正 spec 614 的 85%；共享窗最好 440）、延迟 144ns。

| 指标 | 共享机（load~150） | 独占 @128t (n=601) | 独占 @160t (n=401) | 预测 |
|---|---|---|---|---|
| cpu_moe_wall median | 26.8ms | **17.3ms** | **16.7ms** | 17-18 ✓✓ |
| min / p25 | 18.0 / — | 14.5 / 15.9 | 13.6 / 14.9 | — |
| p95 | — | 32.4 | 32.4 | <min+3 ✗ |
| decode 吞吐 | ~13 | **15.4-16.2** | 15.5-16.3 | 15-16 ✓✓ |

**判定**：
1. **8.5ms 邻居归因——主体实锤**：median 26.8→17.3（回收 **9.5ms**），精确落进预测区间；吞吐 13→16 兑现。
2. **p95 长尾是自家的**：独占下 p95 仍 32.4（分布：71% <20ms 很紧、20-25ms 次峰 18%、>60 仅 3 笔含一次 498ms hiccup）。
   即上尾 ~5% 来自自家（NPU sync/调度/GC 类 hiccup + fork-join skew），不是邻居——dispatch/skew 4.6ms 结构改造的赔率
   维持原判（中风险低赔率，且端到端已 NPU 主导）。
3. **160 线程独占下翻盘**：median 16.7 vs 17.3（-3.5%，共享机上 160 是输的→抢核假设证实），但吞吐持平
   （NPU ~45ms 主导，CPU 的 0.6ms 被稀释）。**生产维持 128**（160 多占 32 核收益≈0）。
4. 微基准独占 ×3（0.387-0.401ms）与共享机无差——隔离短窗本就常逃过邻居，长跑才暴露差异（方法论备忘）。
5. 四 prompt 连贯性过。

**终账（vs Q8_0 原始基线）**：cpu_moe_wall 55 → 17.3ms（**-69%**），decode 8.5 → **16 tok/s（+88%）**，DRAM 275→137GB。
**下一杠杆**：token 时间 ~62ms 中 NPU 侧 ~45ms 占 73%——B 线 CPU↔NPU overlap 是唯一大头。

---

> **历史状态**：开放（Session D 起点）｜**日期**：2026-06-10｜**隔离 worktree**：`/workspace/code/kt-D-mxfp4`
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
