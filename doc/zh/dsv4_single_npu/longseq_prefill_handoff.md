# Handoff — 长序列优化:prefill 逐层流式加载 + 热专家预加载

> **状态**:开放(Session C 起点)｜**日期**:2026-06-09｜**隔离 worktree**:`/workspace/code/kt-C-longseq`
> **场景**:序列长度**超过定义 context 长度**的长序列推理。
> **基线**:主干 `dsv4_one_card_dev` @ `22aac3d`(decode 已 `--kt-cpuinfer 128` + GEMV prefetch → ~9.5 tok/s;
> CPU 权重启动加载 ~47s;Q8_0 生产 + NPU graph 已闭合)。

---

## 启动提示词(开新 session 时整段贴)

> 你接手 **DeepSeek-V4-Flash 单卡 NPU 的"长序列(序列长度超过定义 context 长度)优化"**,两个子目标
> (2026-06-10 方向修正后):
> (1) **prefill 专家上 NPU 算 + 按层流式 DDR→HBM**——长 prefill M≫1,NPU GEMM 远快于 CPU;每层专家
> W8A8 ~6.4GB 按层 H2D 双缓冲流入 HBM(算 L 层时预取 L+1),MoE 部分预估 ~90×(§0/§3.2);
> (2) **prefill 激活分布 → decode 专家池**——prefill 记录专家命中,结束后按命中率把热专家留驻 HBM,
> decode 热走 NPU、冷走 CPU(复用 gpu_experts_mask/remap,改 post-prefill 动态)。
>
> **本文(这份 handoff)就是你的完整起点,从 §0 往下读。**
>
> 工作区:`/workspace/code/kt-C-longseq`(分支 `longseq-prefill`,独立 sglang 分支 `longseq-sglang`,
> kt-kernel 有 llama.cpp+llamafile 可重编 + 基线 `.so`)。启动脚本自动用本 worktree 的 sglang+kt-kernel,
> **不用 export PYTHONPATH;端口 8013**。
>
> ⚡ ~~开工第一步:验证子目标 1 前提~~ **已完成(2026-06-10)**:CPU baseline 见 §3.1,NPU 流式
> 量化依据见 §3.2;剩一个待实测前提:**910B3 W8A8 grouped matmul 每层耗时**(须 <271ms/层才是
> 纯 copy-bound 流水)。验完即进入流式 pipeline 设计实现。
>
> 边界:热专家标定已被 B 放弃 → **子目标 2 现在 C 独占**;B 现做 NPU/CPU 并行+MTP,会动 submit/sync/overlap
> 编排,合并时对齐(§2)。**实时 expert cache/evict 留作后续单独 session**——C 只把 load/evict 原语 +
> residency 钩子留干净、策略做简单版(§8)。
>
> 纪律:快参考/前提先实测验证(输出非零)再信;只杀自己 PID/端口、**绝不广播 `pkill -f sglang.launch_server`**;
> 拉服务前 `npu-smi info` 选空卡 + 查端口;改 C++ 重编只动自己 worktree 的 `.so`(§6 纪律)。

---

## 0. 任务(两个子目标,2026-06-10 方向修正)

> ⚠️ **方向修正(用户澄清)**:流式加载的目的地是 **NPU HBM**,不是 CPU/DDR。
> 旧表述(NVMe→DDR 的 CPU 侧流式)作废;§3.1 的 CPU 数据保留,作为**要打败的 baseline**。

1. **prefill 阶段专家在 NPU 上算 + 按层流式 DDR→HBM**:长 prefill 的 M=batch≫1,NPU GEMM 远快于
   CPU MoE(baseline:0.75ms/token/层)→ 每层专家权重(W8A8 ~6.4GB/层)按层 H2D 流入 HBM
   (算第 L 层时预取第 L+1 层,双缓冲 ~13GB HBM),算完即可让位 → (a) 长 prefill 的 MoE 部分
   预估 ~90×(32k:~12s vs CPU ~1058s,copy-bound);(b) 专家不必常驻 HBM。
   数字依据见 §3.2。
2. **prefill 激活分布 → decode 专家池**:prefill 期间记录 router 的专家命中分布(本来每层专家都流经
   HBM/NPU),prefill 结束后**按命中率把热专家留驻 HBM 形成 decode 专家池**(HBM 预算内 top-N/层),
   decode 热专家走 NPU、冷专家走 CPU(复用 `gpu_experts_mask`/remap 机制,改为 post-prefill 动态生成)。

---

## 1. 工作区 / 隔离(已建好,别碰别的目录)

| 项 | 值 |
|---|---|
| 仓库 | `/workspace/code/kt-C-longseq`(git worktree,父分支 `longseq-prefill`,从主干 `22aac3d`) |
| sglang | **独立 clone**:`third_party/sglang`(分支 `longseq-sglang`)→ 可改/commit,不影响主仓和别的 session |
| kt-kernel | llama.cpp + llamafile 齐(**改 C++ 自己重编**)+ 基线 `.so` `12c8c58` |
| 启动 | 脚本自动用本 worktree 的 sglang + kt-kernel,**不用** export PYTHONPATH |
| 提交 | 父仓 Python/C++ → `longseq-prefill`;sglang 改动 → `longseq-sglang` |
| 重编 | `cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace` |

---

## 2. 与 Session B 的边界(2026-06-09 更新)

B **已放弃热专家标定**,改做 **NPU↔CPU 并行计算效率提升 + MTP(多 token 预测)合入**。因此:

- ✅ **子目标 2(热专家预加载/不 evict)现在归 C 独占**,**不再和 B 冲突**——热专家/常驻这条线 C 全权。
- ⚠️ **新的(较弱)重叠**:B 的"并行计算效率"会动 **submit/sync/overlap 编排**(`kt_ep_wrapper` /
  `kt-kernel/python/experts_base.py` 的 `submit_forward`/`sync_forward`/host-callback/dual-stream);
  C 子目标 1 的 **prefill CPU MoE 路径**也碰这些文件 → **合并时在这些文件上对齐**(开发期 worktree
  隔离,无即时冲突)。B 的 **MTP** 改 decode/投机路径,与 C 基本正交(但 MTP 会改每步 token 数 → 影响
  M 大小 → 间接影响 prefill/decode 的 compute-vs-bandwidth,留意)。

---

## 3. ⚡ 开工第一步:先验证子目标 1 的前提(别跳)

测**长序列 prefill 的 CPU MoE 到底是不是计算密集**——只有 compute-bound,"用计算掩盖流式加载"才成立。

1. 拉起长 prefill 配置(`CHUNKED_PREFILL_SIZE=32768` 等,见 `tools/p27_launch_ds4flash_npu_longcontext.sh`
   + `doc/zh/DeepSeek-V4-Flash_NPU_decode_profiling_runbook.md` 的 seq32k 流程),发超长 prompt。
2. 量 prefill 阶段 CPU MoE 的耗时随 batch/seq 的变化(`KT_DECODE_TIMING` 的计时桩主要覆盖 decode 的
   `run_pinned_forward_sync`;prefill 走的是 `submit_forward`/`forward` 路径,可能要加 prefill 计时点)。
3. **判读**:
   - prefill CPU MoE **随 token 数线性涨、且算力占比高(compute-bound)** → 有掩盖加载的空间,做子目标 1。
   - prefill **仍是 bandwidth-bound**(和 decode 一样,加载量主导)→ 前提不成立,**回报换思路**。

> 纪律:之前有人没验前提、拿幻象 no-op 当"快参考",坑了数小时。**先实测前提,再动手。**

### 3.1 ✅ 验证结果(2026-06-10):前提成立,prefill CPU MoE 是 compute-bound

方法:`kt-kernel/python/experts_base.py` 加 `KT_PREFILL_TIMING=1` 计时桩(prefill 路径 submit→sync 每层墙钟,
env 门控零开销);服务器须 **`KT_FORCE_SYNC_SUBMIT=1`** 拉起才能量到真实 CPU 耗时(默认 async 模式
submit/sync 只入 NPU stream host-callback,host 不阻塞,桩量到的是 ~0.5ms/层的假数据)。
卡5 / 端口8013 / `CHUNKED_PREFILL_SIZE=32768` / `--kt-cpuinfer 128`,逐档发 prompt(`tools/p27_curl_long_prompt_sweep.sh`)。

| M (tokens) | CPU MoE 每层 | 相对上一档 |
|---|---|---|
| 1(decode 形态)| 32.9 ms | —(纯带宽底座)|
| 512 | 407.9 ms | — |
| 1024 | 761.0 ms | 1.87× |
| 2048 | 1504.6 ms | 1.98× |
| 4096 | 2979.9 ms | 1.98× |
| 8192 | 6284.4 ms | 2.11× |

判读:**~0.75 ms/token/层,完美线性 → compute-bound**(带宽项即 M=1 的 32.9ms 常数,M≥512 时占比 <8%)。
对照 NVMe 每层加载 ~2.5s(@2.7GB/s):**M≈3400 交叉;chunk ≥4096 tokens 时每层计算(2.98s/6.28s)
足以完全掩盖下一层权重的流式加载** → 子目标 1 可做。
旁证:默认 async overlap 模式端到端比 force-sync 快 ~30%(8192 prefill:188s vs 271s),
真实配置下"计算伞"同样存在。

原始日志:`tools/longseq_dbg/prefill_premise_server3.log`(sync 模式数据)、
`prefill_premise_server2.log`(async 模式端到端)、`sync_sweep3.log`。

### 3.2 ✅ NPU 流式方案的量化依据(2026-06-10,方向修正后)

模型形状(config.json):43 层全 MoE,256 routed experts/层,top-6,expert FFN 4096×2048×3
→ **每专家 W8A8 ≈ 25.2MB,每层 ≈ 6.4GB,全模型 ≈ 277GB**。

实测(卡5,torch_npu 1GiB copy ×5):**H2D pinned 23.6 GiB/s,pageable 7.9 GiB/s**(流式源必须 pinned)。
DDR 1.5TB(可用 ~1.46TB)→ GGUF Q8_0(CPU 用,~287GB)+ W8A8 专家副本(NPU 流式源,~277GB)可双份驻留。

| 量 | 值 |
|---|---|
| 每层专家 H2D(6.4GB int8 pinned)| ~273 ms |
| CPU MoE baseline(§3.1)| 0.75 ms/token/层(M=32768 → ~24.6 s/层)|
| 交叉点(NPU 流式 vs CPU)| **M ≈ 360 tokens**:流式每层固定 ~273ms(copy)vs CPU 0.75×M;chunk≥512 总赢 |
| 32k prefill 全程 MoE(43 层流水)| 流式 NPU ~11.7s(43×273ms,copy-bound)vs CPU ~1058s ≈ **90×** |
| HBM 双缓冲 | int8 ~12.8GB(2×6.4GB),须从 KV pool 预算里让出(`mem_fraction_static`)|

### 3.3 ✅ 真实算子实测:NPU MoE 是 **copy-bound,不是 compute-bound**(2026-06-10)

> ✅ **澄清(2026-06-10 订正,感谢用户指出)**:**生产脚本 `p27_launch_ds4flash_npu.sh`
> 用 `--kt-num-gpu-experts 32`** → 32/256 专家常驻 HBM 走 NPU grouped-matmul、224 走 CPU,
> **NPU W8A8 专家路径在生产里一直是活的、已验证的**(decode ~9.5 tok/s,精度保持)。
> 我前提验证误用了 `_num_expert_0` 变体(`--kt-num-gpu-experts 0`,全 CPU,且默认 MODEL_PATH
> 是错的——第一次启动失败的根因),所以 **§3.1 是纯 CPU baseline,不是生产形态**。
> 含义:(a) W8A8 NPU int8 grouped_matmul **不需要从零点亮**,已在生产跑;(b) 现有路径是
> **固定 32 常驻**,流式方案要把它从"32 常驻"扩成"prefill 期 256 个全部流经 HBM";
> (c) `longseq` 的 launcher 应改基于 **32-expert 生产脚本**,而非 `_num_expert_0`。

用**真实算子** `npu_grouped_matmul`(sglang `unquant.py:forward_npu` 同款:init_routing_v2 →
gmm1 → swiglu → gmm2)实测每层 routed-expert FFN(H=4096/I=2048/E=256/top6,bf16,卡5)。
脚本 `tools/longseq_dbg/npu_grouped_matmul_bench.py`:

| M | NPU MoE 每层 | TFLOPS | vs int8 H2D 273ms |
|---|---|---|---|
| 512 | 9.55 ms | 16 | copy 比它慢 28× |
| 4096 | 10.95 ms | 113 | copy 慢 25× |
| 8192 | 15.95 ms | 155 | copy 慢 17× |
| 16384 | 24.05 ms | 206 | copy 慢 11× |
| 32768 | 41.62 ms | 238 | copy 慢 6.6× |

**判读(前提成立,但重新定性)**:NPU 计算(@32k 仅 41.6ms/层)**远快于 H2D copy(273ms/层)**
→ 流式 prefill 是 **copy(H2D 带宽)-bound**,NPU 在等权重时 ~85% 空闲。结论:
1. **"用计算掩盖加载"反过来成立**:计算稳稳藏在 copy 伞下,流水节拍 = max(273ms copy, 42ms 计算)= **273ms/层**;
2. 全 32k prefill 的 MoE ≈ **11.7s**(43×273ms),vs CPU ~1058s → **~90× 站得住**;
3. **下一个优化瓶颈是 H2D 带宽本身**(已 pinned 23.6GiB/s)——要再快得提升有效 H2D(多 copy stream/
   并行 DMA),或 prefill chunk 太小才省得下专家(但 M≥512 时 256 专家基本全激活,省不了)。
4. **bf16 权重(12.9GB/层,H2D 546ms)更 copy-bound** → 流式源用 int8 才划算(273ms)。

注:本 bench 用 bf16 grouped_matmul 测**算子速度**;✅ **已查清生产 W8A8 NPU 专家路径
(2026-06-10 代码核实)**:`fused_moe_triton/layer.py:274` `quant_config.get_quant_method()`
为本 W8A8 模型选 **`NPUCompressedTensorsW8A8Int8DynamicMoE`**(`compressed_tensors_w8a8_int8_moe.py`),
权重 = **int8 `w13_weight`[E,...] / `w2_weight`[E,...] + bf16 `*_weight_scale`**,走 NPU int8
grouped_matmul(动态 per-token 量化)。compute 量级与 bench 相近,copy-bound 结论不变
(int8 H2D 273ms ≫ compute)。(`kt_ep_wrapper.py:315` 硬编码的 `CompressedTensorsWNA16MoE`/Marlin
是 docstring 示例默认,**非生产路径**;生产 gpu_method 由 caller 注入。)

**剩余待验**(实现期边做边量,不阻塞设计):(a) H2D copy 与 NPU compute 真并发时的互扰
(同一 HBM/总线);(b) ✅ **已验:277GB pinned 可行**——`/tmp/pin_probe.py` 在卡5 上一路 pin 到
**320 GiB+ 无报错**(DDR 1.5TB),∴ **全 277GB int8 常驻 pinned 做流式源可行,无需环形 staging**;
(pageable→pinned memcpy 19.1 GiB/s,备选不需要)。
(c) 把现有"32 固定常驻"扩成"prefill 期 256 全流经"——复用 `gpu_method.apply` + `gpu_experts_mask`,
新增按层 H2D 预取 + 双缓冲 weight slot。

### 3.4 ✅ 三带宽反推 + 交叉验证(2026-06-10,用户驱动)

用 **E-sweep**(固定 M=256 使 compute 极小,扫专家数 E)反推:NPU 每层耗时**完美正比于 E
(=权重字节)**,跨 E 反推的 HBM 带宽恒定 → 证明小 M 下算子是**读专家权重的 HBM 带宽 bound**,
不是固定开销。脚本 `/tmp/hbm_probe.py`(逻辑同 `npu_grouped_matmul_bench.py`)。

| E | 每层 ms @M=256 | 权重 GB(bf16)| 反推 HBM |
|---|---|---|---|
| 64 | 2.34 | 3.2 | 1378 GB/s |
| 128 | 4.71 | 6.4 | 1369 GB/s |
| 192 | 7.06 | 9.7 | 1369 GB/s |
| 256 | 9.46 | 12.9 | 1362 GB/s |

**三带宽(全部实测,落在 910B3 标称合理效率区间)**:

| 带宽 | 反推方法 | 值 | 对标 910B3 | 效率 |
|---|---|---|---|---|
| **HBM 读** | E-sweep,time∝E | **~1.37 TB/s** | 标称 ~1.6 TB/s HBM2e | 86% |
| **算力 bf16** | M-sweep 线性段斜率 1.15µs/token | **~260 TFLOPS** | 标称 ~376 TFLOPS | 70% |
| **PCIe H2D** | 6.45GB/273ms;直测 pinned 23.6 GiB/s | **~24–25 GB/s** | PCIe Gen4 x16(理论 31.5)| 80% |

**核心比值:HBM : PCIe ≈ 1370 : 25 ≈ 55×。** 同一层 6.45GB int8 专家:从 HBM 读进 cube 仅
**4.7ms**,经 PCIe 搬进 HBM 要 **~260ms**。NPU 消化权重比 PCIe 喂权重快 ~55 倍 → **这是流式
永远 copy-bound 的根因**。compute 追平 copy 需 M≈45 万 token,现实 prefill(≤32k)恒 copy-bound。

**架构结论**:
1. **全模型权重过一遍 PCIe = 277GB / 25GB/s ≈ 11.1s,是长 prefill MoE 的硬地板**(与序列长度无关);
   vs 纯 CPU ~1058s → ~90×,但本质被 **PCIe 带宽**锁死,不是算力。
2. NPU 流式期间 ~98% 空闲 → 最优调度 = **逐层流式 + 每层把整条序列全部 token 一次算完再换层**
   (layer-at-a-time over full sequence),把 260ms H2D 摊到尽量多 token(加 token 近乎免费);
   与"小 token-chunk × 多层"相反。
3. 再快只能动 PCIe 侧:多卡并行搬 / Gen5 / 或**热专家常驻 HBM 不重复搬**(= 子目标 2;
   decode 决不能每 token 付 260ms,必须靠常驻池)。

⚠️ 环境坑(容器重启后):`libhwloc.so.15` 会丢 → `apt-get install -y libhwloc15`
(kt_ep_wrapper 把 ImportError 吞成 "kt_kernel is not installed");拉服务须显式传
`MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8`(脚本默认路径错)。

---

## 3.5 ✅ 实施方案细化:prefill 模式选择(hybrid vs 流式)+ 热专家方案(2026-06-10)

### A. 两种 prefill MoE 执行模式的代价模型(每层)

| 模式 | 每层耗时 | 随 M | 说明 |
|---|---|---|---|
| **Hybrid**(现状:32 常驻 NPU + 224 CPU)| `≈ 0.67 ms/token × M` | **线性** | CPU 担 224/256≈87.5% 工作(0.767×0.875),NPU 32 个并行被掩盖;无 H2D |
| **流式**(256 全部按层 H2D→NPU)| `≈ 260 ms`(固定)| **常数** | int8 6.45GB/层 PCIe 搬运,copy-bound;compute(≤42ms)藏在伞下 |

> 流式 260ms/层只在**单遍扫层**(该层一次算完所有 token)时成立。若内存逼着小 chunk 多遍扫,
> H2D 要重复 → 流式代价 = `11.2s × ceil(S / M_chunk)`。**∴ 流式必须配大 chunk / layer-at-a-time。**

### B. 划算交叉点(单遍流式,全程 prefill = S tokens)

每层交叉:`0.67·M = 260` → **M* ≈ 388 token**。全程(43 层):

| S (prefill 长度) | Hybrid `0.67·S·43` | 流式 `43×260ms` | 赢家 |
|---|---|---|---|
| 256 | 7.4 s | 11.2 s | **hybrid** |
| 512 | 14.7 s | 11.2 s | 流式(1.3×)|
| 1024 | 29.5 s | 11.2 s | 流式 2.6× |
| 4096 | 118 s | 11.2 s | 流式 10× |
| 8192 | 236 s | 11.2 s | 流式 21× |
| 32768 | 944 s | 11.2 s | **流式 84×** |

**结论(prefill-only 策略)**:
- **S ≲ 500 token(短 prompt)→ hybrid**(现状 32 常驻 + CPU);流式的 11.2s 固定扫层不划算。
- **S ≳ 500 token(长序列,本项目目标场景)→ 流式 + 单遍 layer-at-a-time**;越长越赢。
- 阈值取整 **512 token** 作为切换点(也对齐 page_size)。
- **CPU 在流式期一起算?** 流式每层 260ms 内 CPU 至多再消化 ~350 个 assignment(0.26s/0.75ms),
  vs M×6=20 万 → 贡献 <0.2%,长序列不值得;保持流式=纯 NPU,实现简单。

### C. 热专家分布输出 + decode 加载方案(子目标 2,接 §8 后续 cache session)

**为什么必须有**:decode(M=1)若走流式要付 260ms/层 → 荒谬;必须**热专家常驻 HBM**。
prefill 流式时 256 专家本来就全过一遍 HBM,顺手统计命中 → 直接定 decode 常驻池。

1. **统计(prefill 期)**:每层维护 `count[layer][expert]`(router topk_ids 累加,zero-copy 计数,
   按 token 加权)。开销可忽略(一次 scatter-add)。env 门控 `KT_PREFILL_EXPERT_HIST=1`。
2. **定池(prefill 末)**:每层取 `count` 的 **top-K**(K = 现 HBM 预算的常驻数,当前 32)→
   写进现有 `gpu_experts_mask` / `logical_to_gpu_index`(本来 init 时按 prefix/frequency 静态定,
   现改为 **post-prefill 动态**,prompt 自适应)。
3. **加载**:top-K×43 个热专家从流式 pinned 源 H2D 进**常驻 HBM slot**(复用现 32-slot buffer)
   = 32×43×25.2MB = 34.6GB / 25GB/s ≈ **1.4s 一次性**,prefill→decode 切换时做。
4. **产物(为后续 session)**:导出 `[43 × K]` 热专家 id+count 表 + 分布偏度(top-32 占总激活的 %)。
   偏度高 → 常驻命中率高、residency 划算;偏度平 → 提示后续做更细的 LRU/LFU(§8)。
5. **验证(子目标 2 的前提,实现期测)**:用 prefill 定的常驻集,量 **decode 阶段命中率**
   (decode token 的 topk 落在常驻 K 内的比例)。命中率高 = prefill→decode 局部性成立 = 方案有效。
   留 `residency` 钩子干净,真·实时刷新/evict 策略交 §8 后续 session。

### C-bis. ✅ 直方图实测(2026-06-10,32-expert 生产配置 + 真实文本 ~89k token/层)

`KT_PREFILL_EXPERT_HIST=1` 已实现并跑通(sglang `kt_ep_wrapper.py` commit `d8c460d6b`,
guard 住 graph capture/decode);真实文本(repo 文档/源码,token 多样)prefill 后 dump
`[43×256]`,`analyze_expert_hist.py` 分析:

| K(常驻数)| 动态 top-K 占激活 | 静态 prefix[0:K] | 动态增益 |
|---|---|---|---|
| 16 | 25.7% | 6.5% | **3.9×** |
| 32 | **39.5%** | 12.8% | **3.1×** |
| 64 | 58.4% | 25.7% | 2.3× |
| 96 | 71.7% | 38.0% | 1.9× |
| 128 | 81.5% | 50.0% | 1.6× |

**三个关键结论**:
1. **冷专家/层 = 0** → 长 prefill **256 专家全激活**,坐实用户判断;流式必须每层搬全 256,省不掉。
2. **现生产的静态 prefix-K ≈ K/256(=随机水平)**:prefix-32 仅占 12.8%≈12.5%,prefix-128=50%。
   即**专家 0..K-1 根本不是热的,现静态放置等于没偏好** → NPU 常驻的 32 个只接到 ~13% 激活,87% 砸给 CPU。
3. **prefill 直方图定的动态 top-K 多接 2–4×**(top-32:39.5% vs 12.8%)。这就是子目标 2 的价值:
   **同样 32 个 HBM 常驻槽,动态放置让 NPU 命中率 ~3×**,decode 更多激活走快的 NPU、更少砸 CPU。
   偏度温和(~3×,负载均衡训练所致),要 80% 覆盖需 128/256 常驻 → HBM 预算 vs 命中率可调。

> ⚠️ 当前直方图**跨请求累加**(全局),真实现要**按请求复位**(prefill 开始清零、结束 dump 该请求池)。
> 测试时混入过退化 filler 数据,已用真实文本多发几次冲淡(冷专家 0、偏度稳定即代表性 OK)。
> **decode 命中率验证(下方步骤 5)仍待做** —— 这才最终确认 prefill→decode 局部性。

### D-2a. ✅ 子任务 2a 验证通过(2026-06-10):流式 round-trip 地基成立

`tools/longseq_dbg/stream_2a_roundtrip.py`(自包含,inline 生产算子避 sglang 循环 import)。
复刻 `NPUW8A8Int8DynamicMoEMethod` 的权重 layout(`w13` int8 [E,2I,H]→transpose→
`npu_format_cast` FRACTAL_NZ;scale squeeze→bf16)+ prefill 算子 `npu_fused_experts`,
全 256 专家单层实测(卡4):

| 验证项 | 结果 |
|---|---|
| **NZ 字节经 pinned DDR round-trip** | ✅ H2D 后 `npu_format=29`(NZ 存活),**无需每层 reformat** |
| **流式输出 vs 权重常驻 reference** | ✅ `max_abs_diff=0.000e+00`(bitwise 一致) |
| **整层 H2D** | 6.44GB = **308ms(20.9GB/s)** ≈ §3.4 预测 |
| **compute(M=4096)** | **6.7ms/层**(int8 比 bf16 bench 的 11ms 更快)|
| **节拍** | max(308,6.7)=308ms → **copy-bound 46×**,坐实 §3.4 |

结论:**流式地基全通**——DDR 源格式对、H2D 对、生产算子吃得下、数值 bitwise 一致、copy-bound 成立。
NZ 字节直存 DDR 是最快路径(Path-1);ND 存+H2D 后 NZcast 的兜底(Path-2)也对但 reformat ~61ms/层,不需要。
⚠️ 踩坑:E=256 权重 6.4GB,进程硬崩(ERR99999)会**泄漏 HBM 不释放**(卡5 被 dead PID 占 58GB),
换空卡跑;`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 减碎片。

**2a 余项(并入 2b)**:用真实 checkpoint 一层权重 + CPU fp32 dequant 参考对数值(确认 builder 读
checkpoint 正确);现 2a 用随机权重对"权重常驻 reference",已证流式机制忠实,语义正确性由生产精度背书。

### D-2b. ✅ 子任务 2b(2026-06-10):流水设计定型——**串行单 slot 即最优**(双缓冲≈持平,不值得)

`tools/longseq_dbg/stream_2b_prefetch.py`(复用 2a 函数,卡4,K=6 层 NZ pinned 池,**均带 warmup**):

| 方案 | 每层 | 43 层外推 |
|---|---|---|
| **serial 单 slot**(default stream,H2D→compute 串行)| **298.8ms** | **12.8s** |
| 双缓冲(H2D default + compute side stream)| 314.3ms | 13.5s |

| 多 copy stream 聚合 H2D | 带宽 |
|---|---|
| 1 stream | 21.2 GB/s |
| 2 stream | 23.6 GB/s |
| 4 stream | 23.5 GB/s(= PCIe Gen4 x16 墙)|

> ⚠️ **测量教训(2026-06-10,重要更正)**:本节首版报过"双缓冲 515ms、side-stream H2D 只 10GB/s、
> 多流 1 流 10GB/s",**全是 warmup/首次触碰假象**(无预热 + 迭代太少 + 新分配 buffer 首触)。
> 受控实验(`h2d_controlled.py`:同 buffer、预热、30 迭代,只变 stream)证实:**side stream 与
> default stream H2D 带宽基本一样(22–26 GB/s),无 2× 惩罚**;PCIe ~23.6 GB/s 才是真上限。
> 加 warmup 重测后,双缓冲 314ms ≈ serial 299ms(差 5%,在噪声内)。**结论不变但理由更正如下。**

**结论(流水设计定型)**:
1. **串行单 slot(default stream)≈ 最优:298.8ms/层 → 43 层 12.8s,vs CPU ~1058s = ~83×**。
2. **双缓冲 ≈ serial(不更快也不更慢)**:本质是 **copy-bound**——compute(6.7ms)只占 copy(~300ms)2%,
   overlap 最多省 2%,还被跨 stream 同步开销吃掉 → 与 serial 持平。∴ **选 serial 单 slot**(更简单、
   最省 HBM:1 slot 6.4GB),不是因为双缓冲"坏",而是**没必要**。
3. **多流打不破 PCIe 墙**:1 流 21.2,2 流即饱和 23.6,4 流 23.5。**PCIe ~23.6GB/s 是硬墙**
   (277GB 全模型扫一遍 ~11.7s 硬地板);要再快只能多卡/Gen5(§3.4 结论坐实)。

**2b 余项(并入 2c/checkpoint 加载器)**:真实 checkpoint 一层 256 专家 int8+scale 读取与
gate/up→w13 拼接顺序、CPU fp32 dequant 数值对照(本 2b 用随机权重验流水/带宽,机制已足)。

### D. 实现增量(建议顺序,worktree 隔离)

1. **[小、✅ 已做] prefill 专家命中直方图**:`kt_ep_wrapper.py` `KT_PREFILL_EXPERT_HIST=1`
   累加 `count[layer][expert]` + 导出 `[43×256]` + skew 摘要(commit `d8c460d6b`)。**待补:按请求复位**。
2. **[中] 流式权重池 + ~~双缓冲~~ 串行 H2D loop**(2a✅/2b✅ 已验流式地基与最优流水):
   DDR pinned NZ int8 专家池 + **串行单 slot**(default stream:H2D 第 L 层 → 跑算子 → 覆盖搬 L+1)。
   双缓冲经 2b 实测与 serial 持平(copy-bound,overlap 仅值 2%),**不做**取其简单(§D-2b)。
   扩 `gpu_experts_mask` 让 prefill 期 256 全上 NPU。

   > ⚠️ **代码核实后的关键发现(2026-06-10)——现状没有可直接流式的 NPU-layout DDR 源**:
   > - GPU 专家(`NPUCompressedTensorsW8A8Int8DynamicMoE`)的 int8 `w13/w2` 在 **load 时一次性进 HBM**,
   >   host 副本随后释放;buffer 形状**固定 `[num_gpu_experts=32, ...]`**(`create_weights` 时定死)。
   > - CPU 专家在 DDR 但是 **GGUF Q8_0 / AMX llamafile 布局,非 NPU grouped_matmul 可直接消费**。
   > - ∴ 流式需**新建一个 DDR pinned 池**,存全 256 专家/层的 **NPU int8 布局**(w13/w2 int8 + scale)。
   >   **好消息**:safetensors checkpoint(`...W8A8`)本就是 int8,转换量小(对齐
   >   `NPUCompressedTensorsW8A8Int8DynamicMoE.create_weights` 的 layout 即可)。
   > - **关键待验**:277GB pinned page-locked 上限(§3.3 待验 b);或退而用 pinned 环形 staging
   >   (只 pin 几层,后台从 pageable/mmap 填充)——但 pageable→pinned memcpy 会占 CPU/DDR 带宽。
   > - **HBM slot 形状**:串行单 slot 只需 **1 层**(6.4GB,2b 实测最优);现 `[32,...]` buffer 太小,
   >   需新建一个 `[256,...]` int8 NZ 流式 weight slot。
3. **[中] prefill 模式选择**:S≥512 走流式(单遍 layer-at-a-time)、<512 走现 hybrid;
   解耦 attention chunk 与 MoE 单遍(累全 S hidden 再一次 MoE)。
4. **[小] post-prefill 定池 + 1.4s 加载**:top-K 写 mask + 热专家进常驻 slot,切 decode。
5. 全程边做边量 §3.3 剩余待验(并发互扰、pinned 上限)。

---

## 4. 背景必读

- `doc/zh/dsv4_single_npu/DeepSeek-V4-Flash_CPU权重加载加速_P0-P1.md`:当前是**启动时全量加载**(P0 zero-copy +
  P1 并行重排 → ~47s);子目标 1 要改成 **prefill 期逐层流式**(更大的架构改动,与此相关:同一套 load/reshuffle 路径)。
- `doc/zh/dsv4_single_npu/graph_decode_bandwidth_findings.md`:decode 是 DDR 带宽瓶颈(M=1);prefill 与之对比(M=batch)。
- `doc/zh/DeepSeek-V4-Flash_NPU_decode_profiling_runbook.md`:长 context(seq32k)怎么拉起 / profiling。
- `doc/zh/dsv4_single_npu/DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md`:现行总纲(decode ~9.5 tok/s @ cpuinfer 128)。

## 5. 代码地图(起点)

- prefill / decode 分流:sglang `third_party/sglang/.../models/deepseek_v4.py`(`forward_normal_dual_stream` 等)。
- CPU MoE 路径:`kt-kernel/python/experts_base.py`:prefill 走 `submit_forward`/`forward`(M=batch);
  decode 走 `run_pinned_forward_sync`(M=1,graph host callback)。
- 权重加载:`kt-kernel/python/utils/llamafile.py` `load_weights()`(启动时全量)→ C++ `moe.hpp`
  `LLAMA_MOE_TP::load_weights`(P1 并行重排)。子目标 1 = 把它改成按层/按需流式。
- 专家放置:`experts_base.py` `generate_gpu_experts_masks` + kt_ep_wrapper(子目标 2,**与 B 共用**)。

## 6. 纪律(硬要求)

- 任何"快参考/前提"先实测验证(输出非零、premise 成立)再信。
- 杀进程只用自己 PID / 自己端口;**绝不** `pkill -f sglang.launch_server`(会杀别的 session;`pkill -f`
  内联还自杀执行 shell)。
- 拉服务前 `npu-smi info` 选空卡 + `ss -ltnp | grep :8013`;**端口 8013**;避开卡 2(别容器)和 B 在用的卡。
- 红线 R8:shared_experts / router gate 不 offload;改 C++ 重编只动自己 worktree 的 `.so`,不碰主分支/别 session。

## 7. 合并回主分支

- 父仓 Python/C++ → 主 checkout `git merge --no-ff longseq-prefill`(若动了 kt-kernel C++,合后主 checkout 重编 `.so`)。
- sglang 改动在独立 clone 的 `longseq-sglang` 分支 → 用 patch 或推到主的 sglang 子模块分支再合。
- 与 B 在 `kt_ep_wrapper`/`experts_base` 的 submit/sync/overlap 编排上**合并前对齐**(见 §2)。

---

## 8. 后续:实时 expert cache 刷新 / evict(建议**另开 session**,在 C 收口后)

规划中的"实时 expert cache 刷新 + 驱逐机制"是本线的**延伸/收口**,它直接建立在 C 的两块基础上:
- 子目标 1 的**流式加载** = "把某个专家调入 NPU HBM"的**原语**(以及反向的"调出/释放");
- 子目标 2 的**命中跟踪 + 常驻策略** = 一个基础的缓存决策。

**建议:C 先把基础收敛并合入主干**——即**流式 load/evict 原语 + 专家命中跟踪 + 一个简单常驻/驱逐策略 +
干净的 residency API(钩子)**;**然后另开一个 session,在这套已合入的原语之上做完整的实时缓存策略**
(LRU/LFU、HBM 预算管理、刷新节奏、驱逐触发条件、跨请求自适应)。理由:
1. cache 机制**依赖 C 的原语**,C 没收口前做不干净;
2. 全塞进 C 会让这条分支**无限膨胀、难合**;
3. 实时缓存策略本身是**独立且丰富的设计空间**,值得专注 session。

⇒ **C 阶段只需把 load/evict 原语 + residency 钩子留干净、策略做简单版即可**,把"实时刷新/驱逐策略"留给后续 session。
