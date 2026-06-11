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

实测(均带 warmup,卡4,NZ pinned 池),**两个 M 都测**(M=每层 token 数;长序列 layer-at-a-time 时 M=S):

| M | serial 单 slot/层 | 双缓冲/层 | 双缓冲 vs serial |
|---|---|---|---|
| 4096 | 298.8ms | 314.3ms | **+5.2%(更慢)** |
| 32768(真实长 prefill)| 323.7ms | 353.7ms | **+9.3%(更慢)** |

| 多 copy stream 聚合 H2D | 带宽 |
|---|---|
| 1 stream | 18–21 GB/s |
| 2 stream | 23.6 GB/s |
| 4 stream | 23.4 GB/s(= PCIe Gen4 x16 墙)|

> ⚠️ **测量教训**:本节首版报过"双缓冲 515ms、side-stream H2D 只 10GB/s",**全是 warmup/首次触碰假象**
> (无预热 + 迭代太少 + 新 buffer 首触)。受控实验 `h2d_controlled.py`(同 buffer、预热、30 迭代,只变
> stream)证实 side/default stream H2D 带宽一样(22–26GB/s),无 2× 惩罚;PCIe ~23.6GB/s 才是真上限。

#### 为什么双缓冲不值得做(原理 + 数据)

**核心原理:为什么 compute 时间相对 copy 特别少。** 一层 6.4GB int8 专家权重,三条路径搬运它:

| 路径 | 带宽 | 搬 6.4GB 耗时 |
|---|---|---|
| **PCIe H2D**(DDR→HBM,流式必经)| 23.6 GB/s | **~271ms** ← 瓶颈 |
| HBM 读(compute 时 cube 读权重)| 1370 GB/s | ~4.7ms |
| 实际 matmul(M=4096)| — | ~6.7ms |

**PCIe 比 HBM 慢 ~58×**(23.6 vs 1370 GB/s)→ 同一批权重,PCIe 喂进来要 271ms,NPU 读它+算它只要
~5–7ms。这就是 compute 相对 copy 特别少的根本原因:**weight 的 arithmetic intensity 固定**(每层固定
6.4GB 字节、FLOPs 随 M 线性),而**搬运它的两端带宽差 58×**。

compute 随 M 涨但始终 < copy:M=4096→6.7ms(占 copy 2%),M=32768→~31ms(占 ~10%),交叉点(compute=copy)
要 **M≈23 万 token**,现实 prefill(≤32k)永远 copy-bound。

**双缓冲能省的理论上限 = compute 占比(2%~10%)**;但 **2b 实测双缓冲在两个 M 都比 serial 更慢**
(+5%~+9%)——本 NPU 上**跨 stream overlap 的同步/调度开销(~30–60ms/层)超过了它能省的 compute**。
即:理论收益(≤10%)< 实现代价(跨 stream 开销)→ **净亏**。

**结论(流水定型)**:
1. **串行单 slot(default stream)就是最优**:M=32768 时 323.7ms/层 → 43 层 **~13.9s**,vs CPU ~1058s = **~76×**。
2. **双缓冲不做**——不是它"坏",是 copy-bound 下它能省的本就只有 ≤10%,而跨 stream 开销反吃掉更多 → 实测净亏。
   serial 还更省 HBM(1 slot 6.4GB)、更简单。
3. **多流打不破 PCIe 墙**:2 流即饱和 23.6GB/s。**PCIe ~23.6GB/s 是硬墙**,要再快只能多卡/Gen5(§3.4 坐实)。

### D-2c-i. ✅ 子任务 2c-i(2026-06-10):真实 checkpoint → pinned NZ 池加载器 + CPU 数值对照

`tools/longseq_dbg/stream_2c_ckpt_loader.py`。checkpoint 命名(已核实):
`layers.{L}.ffn.experts.{e}.{w1|w2|w3}.weight(+.weight_scale)`,**w1=gate int8[I,H] / w3=up int8[I,H]
/ w2=down int8[H,I]**,scale fp32 per-out-channel。NPU layout(`fused_moe_triton/layer.py:340` 核实):
**`w13=concat([w1(gate)上半, w3(up)下半], dim=0)` int8[E,2I,H]**,`w2`=down,scale 同理拼接;
再 `process_after_loading`(transpose+NZ+bf16 scale)。

实测(卡4,layer 21,全 256 专家):
- 读 checkpoint 一层 6.44GB **in 5.9s**(43 层全建池外推 ~4min 一次性,可缓存优化);
- **NPU 流式输出 vs CPU fp32 dequant 参考:cosine=0.99964,rel_err=0.0265**(int8 dynamic-quant 正常容差)。

⇒ 加载器读 checkpoint 正确(gate/up 拼接、转置、NZ、scale 全对)+ 生产算子数值忠实,**2a 的 CPU 数值
对照余项一并补齐**。**2c-i 完成**。

### D-2c-ii-a. ✅ 子任务 2c-ii-a(2026-06-10):流式接入 sglang,端到端跑通

实现:新模块 `third_party/sglang/.../layers/moe/kt_stream_prefill.py`(sglang commit `d8535b16e`)+
`kt_ep_wrapper.apply` 顶部 3 行分流(histogram 后,early-return,hybrid 路径 645-707 不动)。
`KT_PREFILL_STREAM=1` 门控;长 prefill(M≥`KT_PREFILL_STREAM_THRESHOLD`,默认 512)惰性建 277GB
pinned NZ 池(chunked NZ-cast 控峰值 HBM)→ 单 slot 串行 H2D 256 专家 → 跑生产 `npu_fused_experts`。
任何失败 `try/except` 回退 hybrid;env off = 零改变。

**端到端实测(卡4)**:
- 短 prompt(256<512)→ 走 hybrid,无 `[KT_STREAM]`,4s ✓(阈值门控对);
- 长 prompt(3712 token 真实文本)→ 建池(43 层 592s 一次性,无 OOM)+ 流式,**0 fallback**,
  **生成连贯中文**(续写 repo 文档,非乱码)→ 流式 prefill 数值正确;
- 第 2 个长 prompt(4096)→ **池复用不重建,17s,0 fallback**。**vs hybrid ~137s = ~8×**
  (32k 外推 ~13s vs ~940s ≈ 70×);坐实 §3.4 的 ~13s H2D 地板。

**回退验证**:首测 mem 0.85 时 KV 池占满 → slot OOM → **优雅回退 hybrid,请求仍 200**(安全机制对)。

⚠️ **2c-ii-b 待解(内存预算)**:流式 slot(6.4GB)与 KV 池争 HBM。生产 mem-fraction 0.85 下 KV 池占
57.7GB,slot OOM。测试用 `--max-total-tokens 49152` 压小 KV 池(留 15GB)绕过;**生产需在启动期把 slot
预留进内存预算**(KV 池按 61-6.4-model 算),而非惰性争抢。另:池建 ~10min 需落盘缓存;直方图按请求复位。

剩 2c-ii-b/c/d(内存预算正解、池缓存、模式参数化、post-prefill 定池→子任务 4)。

### D-2c-ii-b. ✅ 内存预算正解(2026-06-10):启动期预留 slot,KV 池绕着它配

`_profile_available_bytes`(`model_runner_kv_cache_mixin.py:65`)按**模型加载后实测的空闲 HBM**算 KV 池。
∴ 把 6.4GB 流式 slot **在模型加载期就 `reserve_slot`**(`kt_ep_wrapper.process_weights_after_loading`
调 `kt_stream_prefill.maybe_reserve_slot`),KV 池 profiling 时这 6.4GB 已不在空闲里 → **KV 自动绕开它配**,
不再惰性争抢/OOM。实测:**生产配置(默认 mem 0.85、context 65536、无任何 hack)直接 `ready to roll`**
+ 启动日志 `reserved streaming slot (6.44GB) at model-load time`。`--max-total-tokens` hack 弃用。
另:`KT_PREFILL_STREAM_POOL_CACHE=<dir>` 池落盘缓存(每层 .pt),重启从缓存加载免重建。**2c-ii-b 完成**。

### D-2c-ii-c. 设计定调(2026-06-10,用户):NPU 侧统一 NZ,CPU 侧用 GGUF —— 双引擎双布局

**结论**:NZ(FRACTAL_NZ)只有 NPU cube 能吃,GGUF/AMX 只有 CPU AMX/AVX 能吃,**collapse 不成一份**——
它们服务两个不同计算引擎。但 **NPU 侧应统一成一份 NZ DDR 池**:

| DDR 副本 | 布局 | 服务谁 | 何时建 |
|---|---|---|---|
| **NZ int8 池** | FRACTAL_NZ | **所有 NPU 侧**:prefill 流式 + decode 常驻热池(子目标 2) + 启动 32 常驻 | **模型加载时一次转**(下方) |
| GGUF/AMX | Q8_0/AMX | decode **冷专家的 CPU 计算**(M=1 带宽 bound,流式/全驻都不行) | 现状 |

**改进(替代当前"惰性建池 + 二次读盘")**:当前实现启动读一遍 checkpoint(转 32 HBM-NZ + 224 CPU-GGUF),
建池**又读第二遍**——冗余。正解:**在模型加载读 checkpoint 那一遍,顺手把全 256 专家转 NZ stash 进 DDR 池**
(钩 FusedMoE expert 加载 / `process_weights_after_loading`),一次读盘、一份 NZ、prefill+decode-resident
+32-resident 全复用,省第二次读盘 + ~22min 建池。GGUF 那份因 CPU 引擎需要而保留(DDR 共存 ~560GB,放得下)。

为什么 decode 冷专家不能也用 NZ:① 流式进 NPU = 273ms/层×43 ≈ 11.7s/token(荒谬);② 全 256 常驻
= 277GB 装不下 64GB HBM;③ CPU 跑不了 NZ(AMX 要自己的 packed 布局,逐帧转太贵)→ 只能 CPU+GGUF。

⇒ 2c-ii-c 实现 = 把池构建从"惰性二次读盘"挪到"模型加载一次转 NZ";`_build_pool`/`reserve_slot`/
`_streaming_forward` 原语不变,只换**填池的时机与数据来源**(从模型加载流复用,而非 standalone 重读)。

#### 2c-ii-c 实测要点(2026-06-10):**slot 分时复用**解决建池 HBM,生产满配零 hack 跑通

实现踩坑链(都验过):
1. **转 NZ 必须过 HBM**:`_nz_pinned` 每 chunk = DDR(CPU ND)→HBM→`npu_format_cast`(NZ,设备算子)
   →HBM(NZ)→DDR pinned。建池的 ~3GB **HBM 临时区**(chunk64)是 OOM 根源,不是 slot。
2. **build-at-load 失败**:模型加载峰值占 48.8GB HBM,此刻仅 333MB free → layer 9 OOM。**加载时是最差时机**(已回退)。
3. **mid-forward 生产满配也 OOM**:KV 池占满,没给建池留临时区 → 崩溃重启循环。
4. **✅ 解(slot 分时复用)**:slot(6.4GB)启动期 `reserve_slot`(KV 池绕开它);建池在前、流式在后**不重叠**
   → 建池前 `_free_slot()` 释放这 6.4GB 当 NZ 转换临时区,`_build_pool` 跑完 `_ensure_slot` 重建 slot。
   **流式特性净占 1 个 slot,在"建池临时区"与"流式 slot"间分时**。

实测(**生产满配:默认 mem 0.85、context 65536、零 hack**,卡1,sglang `60913fa6c`):
- 启动:slot 预留 + ready;第1个长 prompt(2048):建池 568s + 流式,**0 OOM/崩溃,scheduler 不重启**;
- 第2个长 prompt(4096):**14s,0 fallback**。(此前同配置必 OOM/崩溃循环。)

⇒ **2c-ii-b/c 完成**:流式 prefill 在生产满配下无 hack 即用。

#### 建池 568s 耗时拆解(2026-06-10,`/tmp/profile_build.py`,单层 warm)

| 子步骤 | 耗时/层 | 说明 |
|---|---|---|
| **safetensors 读** | **~3.0s** | 1536 个 per-tensor `get_tensor`(256 专家×3 权重×2)的**调用开销**(非带宽);server 上 page cache 被挤掉还要命中磁盘 |
| DDR→HBM `.to(dev)` | ~1.3s | pageable H2D |
| transpose+contiguous | ~0.9s | HBM 重排 |
| `npu_format_cast`(ND→NZ)| **~0.02s 稳态** | **首次 ~6s 是 TBE kernel 编译(一次性),非瓶颈** |
| HBM→pinned | ~0.6s | 回写 |

单机干净 ~5.7s/层,server ~13s/层(多出的是 page cache 被挤→磁盘读 + 已 pin 277GB 下再 pin 变慢)。
**结论:大头是"读 checkpoint",NZ 转换可忽略。** ∴ 把 NZ 转换搬进**模型加载流**(捕获加载时已读的 int8,
免二次读)是正解优化 → 见 2c-ii-c2。

#### 2c-ii-c2(进行中):NZ 转换搬进加载流,免二次读盘

模型加载的 load 循环本就读了全 256 专家 int8(224 个 CPU 专家的 safetensors 读后被丢弃)。
做法:load 循环里**捕获**每个专家 int8 进 per-layer DDR staging(memcpy,廉价),加载末(last-layer
`process_weights_after_loading`)从 staging **一次性建 NZ 池**(slot 分时复用做 HBM 临时区),免再读
safetensors。目标:**第一条 prompt 无额外建池耗时**(建在加载里)+ **加载流最优**(无二次读)。

**✅ 实测(2026-06-10,sglang `929295453`,生产满配 card0)**:
- `capture_expert`(`deepseek_v4.py` load 循环)捕获全 256 专家 int8+scale 引用;last-layer
  `process_weights` 里 `_build_pool_from_stage`(per-layer 从引用 assemble→NZ→pin)→ **免二次读**;
- 建池在**启动里**完成(`pool built from load-capture: 43 层 in 536~607s`)→ **第一条长 prompt 2048 = 14s
  零建池开销**(达成目标 #1),短 prompt 走 hybrid 2s,0 fallback。
- ⚠️ 初版(引用 stage)建池 ~536-607s 与重读一样慢——**根因(已定位)**:捕获的"引用"是 safetensors
  **懒 mmap 视图**,真实读发生在 assemble 时(冷缓存)→ "免二次读"只是把读挪了位置。

#### 2c-ii-c3 ✅ 增量建池(2026-06-10,sglang `31446b731`):根治——捕获即物化 + 就地 NZ

- `capture_expert` **物化** memcpy 进该层**最终 pinned ND 缓冲**(shard 页正热);每层 1536 张量凑齐
  **立即就地 NZ**(chunked ND→HBM→cast→写回**同一块** pinned 字节;单测 bitwise == 整体 cast);
  **零 stage、零额外缓冲、建池摊进加载循环**;in-loop cast 失败(HBM 紧)defer 到 last-layer 重试。
#### 2c-ii-c4 建池 roofline(2026-06-11):瓶颈是单线程 buffered 读 checkpoint,非 H2D

各链路实测带宽 + 读 277GB 耗时(`/tmp` 受控测量):

| 链路 | 实测带宽 | 277GB |
|---|---|---|
| H2D (PCIe) | 24 GB/s | **12s** |
| HBM 内 NZ 转换 | 1.37 TB/s | 0.2s |
| pinning(并行 / 预热后串行)| 44 / 6.8 GB/s | 6~40s |
| memcpy DDR→pinned | 6.9 GB/s | 40s |
| **safetensors 读(现状:单线程 buffered mmap + 1536 get_tensor/层)** | **0.8 GB/s** | **343s ← 瓶颈** |
| NVMe 裸读 direct 单线程 | 3.2 GB/s | 87s |
| NVMe 裸读 direct 并行(卷上限,8 路仅 3.57)| 3.5 GB/s | 79s |

**判读**:H2D 只占 12s(用户直觉对,完全不是瓶颈);~330s 是**读 + rearrange 277GB**。⚠️ pinning 首块
0.71 GB/s 是冷启动假象(warmup 教训),预热后 6.8-44 GB/s,非瓶颈;buffered 读受 page-cache 污染测不准。

**O_DIRECT 受控测量(cache-independent,可信)+ 深挖结论(`tools/longseq_dbg/odirect_reader.py`,正确性 ✓)**:
| 操作 | 带宽 | 说明 |
|---|---|---|
| O_DIRECT 读(纯读,不拷出)单线程 | 2.7 GB/s | = `dd iflag=direct` |
| O_DIRECT 读 并行 4-8 文件 | **3.5 GB/s** | = NVMe 卷上限(并行不再涨)|
| **O_DIRECT 读 + per-expert rearrange(单线程)** | **0.76 GB/s** | ← rearrange 是真瓶颈 |
| 同上 4 线程并行 | 1.2 GB/s | rearrange 的 GIL/dispatch 限制,仅 ~2× |
| memcpy DDR(rearrange 数据量)| 6.9 GB/s | 数据本身 277GB=40s,Python 768 拷/层 inflate 到 ~360s |

**根因**:checkpoint 专家布局是 **expert-major 但散布**(每专家 w1/w2/w3 连 24MB,但专家在文件里跳着排)→
建池要 **768 次/层 per-expert copy** 把专家摆进池 layout(w13=concat[gate,up])。**raw 读能到 3.5 GB/s,
但 Python per-expert rearrange 把整体压回 ~0.76-1.2 GB/s**——这是 Python 硬顶。

**结论**:① **H2D 完全不是瓶颈(12s),用户直觉对**;② 真瓶颈是 Python 读+rearrange 277GB(~1 GB/s);
③ Python 并行最多 ~2×(400s→~200s);④ **要到 NVMe 速度(3.5 GB/s,建池~150s)必须把读+rearrange 落 C++**
(像 kt-kernel GGUF loader 那样,6 GB/s)。落盘缓存(NZ 字节直读)也能绕开,但 torch.save 段错误待修。

#### 为什么 baseline 也转 NZ 却没这延迟?(2026-06-11,用户问)——数据拆解

baseline(32 GPU experts,不流式)**也做 NZ 转换**,但没这延迟,两个原因叠加(8× 数据 × ~6× 慢路径):

| | baseline(32-expert)| 流式池(全 256)|
|---|---|---|
| 从 safetensors 读 + NZ 的专家 | **仅 32 个**(loader 跳过 224 个 CPU 专家,拿 mmap 视图但不物化)| **全 256 个** |
| 读 + NZ 数据量 | 34.6GB(32×25.2MB×43)| **277GB(8×)** |
| 那 224 个 CPU 专家从哪加载 | **GGUF**(Q8_0,kt-kernel **C++ 并行 ~6 GB/s,~47s**)| 我又从 safetensors **重读(Python ~1 GB/s)** |

**判读**:① baseline 的 NZ 转换只 34.6GB(1/8 数据)+ 几秒,折进 ~150s 启动里,无感;② 流式池要读 8× 数据;
③ 我的 Python 读路径(~1 GB/s)比 baseline 的并行 safetensors loader 慢 ~3×。两因子相乘 = ~330s。

> ⚠️ **硬约束(2026-06-11,用户)**:**NPU 只能用 safetensors 的 int8,绝不用 GGUF**。后续 CPU 权重要切
> **MXFP4**(GGUF 里就没有 NPU 要的 int8 了)→ NPU 流式池**必须**从 safetensors 建。**这点不能违反。**
> ∴ 之前"像 GGUF loader"的说法要更正:**重点是"并行读 safetensors"这个技术,不是复用 GGUF 数据**。

#### 2c-ii-c6 ✅ 启动耗时精确分解 + pipelined 建池(2026-06-11,受控对照)

**`KT_LOAD_PROFILE=1`(kt_ep_wrapper.process_weights 插桩)+ baseline/warm 对照,把 519s 谜团钉死**:

| | Load weight | GGUF | construct+safetensors | parread | 总启动 |
|---|---|---|---|---|---|
| **baseline(关流式)** | 121s | **63.7s** | 55s | — | **152s** |
| 流式 warm(串行 parread)| 306s | 60.5s | 61s | 182s | 398s |
| 流式 cold(早先)| 454s | — | — | 172s | 519s |
| **流式 warm + pipelined parread** | — | — | — | **110s** | **338s** |

**结论(195s 谜团解开)**:① GGUF(~62s,287GB CPU 专家)和 safetensors(~58s)在 warm 下流式 vs baseline **几乎一样**——
流式没拖慢它们;② cold 519s 比 warm 398s 多的 ~120s = **纯 page-cache 冷读方差**,非真实成本;③ **流式真实
确定增量 = parread 建池**。**GGUF 63.7s 是 baseline 最大单项**(CPU 切 MXFP4 后会变)。

**pipelined parread ✅**:8 个 O_DIRECT 读 worker(NVMe/CPU)产出 buffer,主线程逐层 NZ-cast(NPU)消费;
read(~101s)与 NZ(~80s)用不同资源 → 重叠到 **110s(first-read 20s)**,vs 串行 182s = **省 72s(40%)**。

#### 2c-ii-c7 ✅ 建池↔load 重叠(2026-06-11):reads 后台化,藏进模型加载

`_start_bg_reads`(首个 process_weights 启动 8 后台 O_DIRECT 读 worker,host-only 无 HBM)+
`_finish_bg_build`(末层 drain done_q + 逐层 NZ)。reads 与 GGUF 加载/构造**并发**。实测(精度 ✓ 逐字一致):

| 版本 | 总启动 |
|---|---|
| capture(冷)| 553s |
| parread 串行(冷/暖)| 519 / 398s |
| pipelined parread | 338s |
| **+ load 重叠** | **308s** |
| baseline | 152s |

**收益有限(338→308,省 30s),根因 NVMe 争用**:parread reads(NVMe)和 GGUF 加载(NVMe)抢同一卷
→ GGUF 从 63.7s 拖到 76.3s;NZ-drain 95s 仍串行在 load 之后(NZ 要 HBM scratch,而 load 期 HBM 紧,
不能安全地在 load 中跑 NZ,否则 OOM——同 build-at-load 教训)。**流式启动从 553s 优化到 308s**,
vs baseline 152s,净增 ~156s(NZ 95 + GGUF 争用 13 + read 未全藏 + cache 方差)。

#### 2c-ii-c8 ✅ "加载到底"结论(2026-06-11):build 已贴硬件地板,没大肉可扣

NZ-cast 95s 子步骤拆解(`/tmp` 单机一层 w13):**transpose+contiguous 0.39s(~50%,最大)**、
H2D 0.18s、D2H 0.22s、**format_cast 0.01s(可忽略)**。chunk 64=35s < 128=40s < 256=46s →
**小 chunk 反而快,64 已最优**。

| build 部分 | 现状 | 能扣? |
|---|---|---|
| read 277GB | 79s(NVMe 3.5GB/s 上限)| ❌ 硬件顶,已藏进 load |
| NZ transpose | ~17s | ❌ NZ 格式必须转置(strided HBM ~3GB/s,非算法慢)|
| NZ PCIe 往返 | ~17s | ❌ 池在 DDR、format_cast 在 HBM,必往返 |
| format_cast | ~0 | 已是 0 |

⇒ **build 不可约 ≈ NZ 54s**(单机;server 95s,差 41s 是内存压力开销)。308s vs 理论 ~206s
(baseline 152 + NZ 54)的 ~100s 水分 = read 没全藏(GGUF 抢 NVMe)+ server NZ 开销 + cache 方差,
**有空间但难且边际**。**结论:这块到此为止,大头交 MXFP4 自然收益。**

**剩余 floor / 未来杠杆**:① 专家数据读两遍(safetensors int8 277GB + GGUF 287GB)= NVMe 硬底
~161s;② **CPU 切 MXFP4 后 GGUF 变小 → NVMe 争用减 → 流式 reads 更快**(自然改善);③ 复用 sglang
并行 loader 一次读 256 专家(避免和 GGUF 双读争用);④ NZ 也藏进 load(需解决 load 期 HBM 紧,风险高)。

#### 2c-ii-c5 ✅ 并行 O_DIRECT 建池实测(2026-06-11,生产满配 card0)

`_build_pool_parread`(8 worker O_DIRECT 读全 256 专家 safetensors → pinned ND → 就地 NZ)。capture 改 no-op。
- **建池 167s = read 98s(277GB → 2.8 GB/s,近 NVMe 上限!)+ NZ 69s**,vs capture 标准等效 ~400s = **2.4×**;
- **精度 ✓**:3637-token 流式 + 32 token,输出与之前已验版本**逐字一致**(权重 bitwise 没变);0 fallback,速度不变。
- ⚠️ **但启动总时只从 553s → 519s(省 34s)**,远小于建池本身的提速。原因:**旧 capture 与模型加载循环
  重叠**(边 load 边物化,load 本就在读),而 **parread 在 load 之后串行跑**。∴ 建池虽快 2.4×,因不重叠,
  总收益被吃掉。**下一步要害 = 让建池读与模型加载重叠**(后台读,趁 load 的 CPU/graph 阶段 NVMe 空闲),
  或直接**复用 sglang loader 的并行 safetensors 机制把 256 专家一起加载**(用户建议,见下)。

**正确的优化方向**:**baseline 本来就并行加载 safetensors 到 NPU**(sglang loader 的 executor 多 worker,
读 ~32 专家 + attn/shared 约 60-75GB 折进 150s,等效 ~3 GB/s)。所以 safetensors 读**能并行也快**——我的
~1 GB/s 是 **Python 实现问题**。⇒ 终极解 = **让 safetensors→NZ 的读达到 baseline 并行 loader 的 ~3 GB/s**
(C++ 或复用 sglang loader 的并行机制),配合干掉 Python per-expert rearrange,建池可压到 ~100s 量级。
**不碰 GGUF。**

- **实测(生产满配 card1)**:每层 NZ 仅 **1.5-1.6s**(旧 13-14s,~9×);43 层全部 in-loop 完成;
  **启动总 553s**(旧 ~750s);建池新增成本 600s→~400s,其中 NZ 65s,**剩余是必须的一次性 I/O**
  (240GB 冷专家 NVMe 读 ~90s + 277GB memcpy + 277GB pinning)≈ 物理地板,再快需线程化 capture 流水(边际)。
- **精度验证 ✓**:3637-token 流式 prefill + 32 token 生成,输出与此前已验证版本**逐字一致**
  (temp=0 确定性 → 权重 bitwise 没变);sweep 256→hybrid 2s / 2048→20s / 8192→14s,0 fallback。

剩 2c-ii-d(直方图按请求复位 → post-prefill 定 decode 热池 = 子任务 4);池落盘缓存(torch.save 段错误,待修)。

### D-目标2(2026-06-10→11,✅ 已修复并验证):动态 decode 常驻池——根因=常驻权重 gather 切的是 host NZ 池(host 切片 format-unaware→字节错乱);改为设备上切片即修复,real-topK decode 完全连贯

**✅ 根因坐实 + 修复验证(2026-06-11)。** `_apply_dynamic_residency` 把热专家从 DDR 池 gather 进 `layer.w13_weight`
时,用的是 **host 上**的 per-expert 切片 `stag13[s].copy_(h13[e])`。池是 **NZ(FRACTAL_NZ tiled)布局**,但 host 张量
**format-unaware**——host 切片按 ND 字节算,从 NZ-tiled 字节里抽出来的是**错位垃圾**。scale 没事(非 NZ)。

**定位链(全部实测,不靠读码猜)**:
1. 算子实测:decode 算子 == prefill 算子 == bf16 ref(cos 1.0)→ 算子无辜。
2. 关图复测:real-topK + `--disable-cuda-graph` 仍乱码 → graph staleness 排除。
3. 运行时 per-layer 差分(`KT_DYN_DIFF`):`cos(gpu,ref_res)≈0` 全层、`cos(cpu,ref_non)≈0.999` → **NPU 常驻分支算的是垃圾,CPU 分支对**。
4. 直接探针(`KT_DYN_PROBE`):`layer.w13_weight[slot] == pool[expert] (ND)? False, cos=0.0004;scale eq? True`
   → **权重错、scale 对**。
5. 单卡 mix 自洽:NPU 分支恒垃圾,但只在常驻=**热**专家(权重大)时显形;prefix(s==e 恒等)、scatter/antitop(冷专家≈0 权重)
   所以"看着干净"。旧 L0 自检 `readback==staging` 永远 True 是拿垃圾比同款垃圾,无效。

**修复**(sglang `kt_stream_prefill.py`):gather 改到**设备**上——整池 H2D 进 NZ slot(整张量拷贝 format 正确),再
`layer.w13_weight.data[s].copy_(slot13[e])`(NPU 切片 format-aware 正确)。验证:`KT_DYN_DIFF` 下 `cos(gpu,ref_res)=1.0000`
全层;生产配置(图开、无 diff)real-topK decode **完全连贯**(`_layers=1`/`num_layers=43`/`qk_rope_head_dim=64`…无重复)。

**意义**:Goal-2 动态热专家常驻现在**精度正确**。诊断开关 `KT_DYN_DIFF` / `KT_DYN_PROBE`(`maybe_dyn_diff`)、
算子测 `kernel_decode_vs_prefill.py` / `nz_gather_test.py` 全保留。

**decode 提速实测(2026-06-11,配对同负载,双服务交替长 prompt 120tok,读 gen-throughput 稳态窗口)**:
| 配置 | decode 稳态 tok/s(8 窗口) | 中位 |
|---|---|---|
| A prefix-32(13% 命中)| 3.40/3.89/3.22/3.96/2.35/3.20/2.44/3.62 | **3.31** |
| B real-topK(share 0.559)| 4.94/4.90/3.82/3.62/4.00/3.81/3.72/3.85 | **3.83** |

**提速 = ratio 中位 1.16× / 均值 1.25×**,远低于 CPU-bound 模型预测的 ~2×。原因(诚实):
1. **争用压缩**:两服务都在 ~3.3–4 tok/s(远低于安静时 ~9.7),DDR 被邻居容器抢,两条路都饿 → CPU/NPU 专家分割
   的差异被压扁,ratio 趋近 1。**故 1.2× 是下界**,安静/独占机上预计更高(1.2×~2× 之间),要专机才能定准。
2. **decode 实际命中率可能 < 0.56**:0.559 是 prefill share,decode(config 列举那种 token)路由未必全落常驻热集。
3. 固定开销(attention/routing/NPU/H2D)占比比模型设的大。

**复测(2026-06-11,安静窗口,全卡 5%/CPU 2%)**:同法配对。A prefix-32 稳态 ~5.2 tok/s(窗口 3.4–6.4),
B real-topK ~7.7 中位 / ~9–10 clean-state(窗口 4.3–10.5)。**安静下提速 ≈ 1.5–1.8×**(比争用下的 1.2× 高,
逼近但未到模型的 2×)。印证之前判断:**争用会把比值压扁,安静下 CPU/NPU 分割差异显出来**。仍短于 2× 因 decode
实际命中 <0.56 + 固定开销。B 实测 120 token **完全连贯**(`_layers=1`…`intermediate_size=12288`…`w8a8`,无重复)。

**精度实测(2026-06-11):real-topK 至少不差于 baseline,本测中明显更好。** 两个 instruction 式 prompt(可校验答案),
**A prefix-32 都掉进重复**,**B real-topK 都连贯且事实正确**(自然 prompt 正确答出"生产可用=3 种";另一 prompt 正确
提取"13 单元/节点、7 节点")。与"多卡全-NPU 是金标准、real-topK 更多专家走 NPU→更靠近参考"自洽。**故修复后
real-topK 精度站得住,甚至优于单卡 prefix-32。**

**成本回归(我的修复引入,确认非争用)**:常驻切换 ~21s 涨到 **~160s(安静下仍 ~160s)**——非 H2D 带宽,而是
device-gather 的 **1376 次(43 层×32 专家)逐专家 NZ 切片 copy** 的 launch/sync 开销。一次性/每长 prefill,可优化
(批量 gather / fancy-index slot13[top] 一次 / 复用 streaming 已 H2D 的 slot)。**净评估:Goal-2 现在精度正确、能跑,
安静下 decode ~1.5–1.8×;但切换变贵(~160s,可优化)。值不值得上 = 解码长度(摊薄切换)× 独占带宽(提速更大)。**

---
**(以下为修复前的调查记录,保留备查)**

实现(sglang `468ea4662`+`df220520f`,`KT_DYNAMIC_RESIDENT=1`):流式 prefill 期 device bincount
计每层激活;末层把静态 prefix-32 换成本请求每层 top-32:权重从 DDR NZ 池 host-gather 进 pinned
staging → 整张量 copy 进 `layer.w13_weight/w2_weight`(**per-slot NZ 切片 copy 字节错误,已验证;
staging 整张量路径 bitwise == fresh cast**),scale device 索引,三处路由结构**原地**改写
(KTEP device mask+l2g;kt_kernel pinned CPU mask——C++ 持指针 live 读)。切换 ~21s/请求(可优化)。

**关键重构(读 `_streaming_forward` 确认):动态常驻是 decode-only 量化混合效应,与 prefill 无关。**
流式 prefill(M≥512)每层走全 256 专家 streaming 路径(全 W8A8,精确),`_apply_dynamic_residency`
只在末层副作用改写 `layer.w13_weight`,**只影响后续 decode(M=1)的 hybrid 路径**。所以:
- baseline decode(prefix-32)= 32 专家 W8A8-NPU + 224 专家 Q8_0-CPU(后者扛 87% 流量);
- 真 top-K decode = 把扛 56% 流量的**热**专家提到 W8A8-NPU,冷专家落 Q8_0-CPU。
两者差异 = **哪 32 个专家走 W8A8 vs Q8_0**。退化 ⇔ 把高流量热专家放到更粗的 W8A8 路径上。

**证据链(判别实验,全部已提交)**:
| 实验 | 常驻集性质 | hit | 长 prompt decode | 结论 |
|---|---|---|---|---|
| `FORCE_PREFIX=1`(0..31)| 同基线/低 id | 13% | ✅ 干净 | **切换机制全对**(权重/双 mask/l2g/graph 原地可见性)|
| `FORCE_SET=shift1`(1..32)| 固定,最小扰动 | 13% | ✅ 干净(64 tok)| **decode graph 吃非 prefix 集没问题** |
| `FORCE_SET=permlayer`(每层 `[L*8..]%E`)| 每层异集,**固定**| 12.5% | ✅ 干净(64 tok)| 排除"每层异集"嫌疑 |
| `FORCE_SET=scatter`(每层 stride-7)| 每层**散布**,固定 | 12.7% | ✅ 干净(64 tok,`_layers=43`/W8A8 连贯)| 排除"层内散布常驻"嫌疑 |
| `FORCE_SET=antitop`(每层 bottom-K)| **数据相关**,低 hit | **0.6%** | ✅ 干净(64 tok,`_layers=1`/qk_rope 连贯)| **数据相关路径无辜** |
| 真 top-K(每层热集)| 数据相关,**高 hit 0.56**| 56% | ✗ ~15 tok 后 `_lame` 循环 | 单卡 hybrid 动态切换运行时 bug(算子已证无辜);误差随热专家权重放大,故只此例崩(见根因 v3)|

五个对照集全干净 → bug **不是**切换机制、不是非 prefix、不是每层异集、不是层内散布、**不是数据相关选择**
(`antitop` 数据相关但低 hit,干净)。real-topK 与所有干净集的唯一区别 = **NPU 命中率(0.6%/13% 干净
vs 56% 退化,单调)**。

**根因结论 v3(2026-06-11,作废 v2 的量化精度说;用户反例 + 算子实测共同推翻):real-topK 退化是
单卡 hybrid 动态切换的运行时状态 bug,不是精度、不是算子。MXFP4 救不了,但这是个能修的 bug。**

**作废 v2(量化精度/δ(E)/128× 粗/MXFP4 反转)**。推翻它的两条硬证据:
1. **用户反例**:多卡 TP 全专家 NPU-W8A8 int8 **不胡言乱语** → W8A8 精度本身没问题,"热专家上 W8A8 掉精度"不成立。
2. **算子实测**(`tools/longseq_dbg/kernel_decode_vs_prefill.py`,纯算子级无 server):同一输入喂 decode 算子
   `npu_fused_experts_w8a8_decode` vs prefill 算子 `npu_fused_experts`,**逐 token cos=1.00000**,两者都对 bf16
   参考 cos=0.99954(int8 量化噪声 ~3%,两边一样)。**decode 算子 = prefill 算子 = 参考,算子无辜。**

**已静态读码核对、确认正确的三处**(所以 bug 不是这些静态逻辑):
- CPU 侧(`kt-kernel/operators/llamafile/moe.hpp:815`):`should_skip_expert()` **每次 forward live 读**
  `gpu_experts_mask` 跳过常驻专家;按**逻辑 id** 索引权重(256 个全在);末段 `Σ weight_j×out_j` 只累非常驻、
  无归一化。无双算、无缺权重。
- NPU 侧(`mask_cpu_expert_routing`):常驻→真权重,非常驻→权重 0。
- 合并(`kt_ep_wrapper:743`):`gpu_out + cpu_out` = 全专家求和,精确。

**所以 bug 在动态切换的运行时状态**(三个结构 `gpu_experts_mask`/`logical_to_gpu_index`/常驻 NZ 权重 在 decode
时未真正一致),不是上面任何静态逻辑。自洽点:只有 real-topK 崩、随热专家权重放大;scatter/antitop 用同一套切换
机制"干净"只是因为它们的常驻专家权重小、把同一误差压住了(不是机制对)。**未定到具体层/具体结构——需运行时差分,
不靠读码猜(读码已证明显眼处全对)。**

**怎么拿到 decode 提速(修正"等 MXFP4"的错):修单卡 hybrid 动态切换这个运行时 bug**,让 real-topK 能连贯解码,
就拿到 ~2× decode。**跟 MXFP4 无关。** 这是 kt_ep_wrapper / `_apply_dynamic_residency` + kt_kernel mask 同步区域
(与 Session B 协调)。

**下一步(钉死的决定性实验)**:运行时 per-layer 差分——real-topK decode 时,逐层比 hybrid 输出 vs 全-NPU 流式
输出(已知对)对同一输入,同时 dump 三个 mask 在 decode 时的实际值/一致性。哪层先发散、哪个结构不一致,就钉死。

**⚠️ decode 提速实测——作废重测(2026-06-11)**:这台机器是**共享**的,decode 是 K920 DDR 带宽 bound,
邻居容器一忙就把本服务 decode 从 ~9 tok/s 饿到 0.3(单测绝对值随邻居负载摆 ~24×)。下面这版用了**不同
负载窗口**测的两个数(A 在 card 0/3/5 都 86–95% 的拥挤时刻测出 0.31,B 另一时刻 1.98),**两个数不在同一
争用下,不可比**,故 6.4×/公平 1.9× 的结论**作废**。

~~A prefix-32 0.31 / B real-topK 1.98 / CPU-bound 回填公平 1.9×~~ ——baseline 被邻居争用污染,不算数。
(B 单请求内 per-window 0.08→4.53→6.07 的**加速**仍是真的:重复循环每步命中同批常驻热专家→命中率→~100%
→几乎全上 NPU。这说明 decode 确被 CPU-offload 专家数 bound,但**倍数要在同负载配对下重测**。)

**流式 prefill 对 decode 无影响(已配对坐实,2026-06-11)**:同条件交替 A=流式(8013)vs B=非流式(8015),
短 prompt 隔离 decode,6 轮:ratio A/B = **1.00**(round1–5:1.01/0.95/1.05/1.00/1.00;round0 0.72 是首请求 graph
warmup)。**277GB pinned pool 不拖慢 decode。** 之前担心的"流式拖垮 decode"是邻居争用假象,非流式本身。
脚本 `tools/longseq_dbg/{measure_decode_speed,profile_decode,paired_decode}.py`。

**待重测(同负载配对)**:prefix-32 vs real-topK decode tok/s,用 `paired_decode.py` 交替两服务同一争用窗口测,
才能得到 Goal-2 真实提速倍数(预期仍 ~2×,但需坐实;且 real-topK 重复循环会把表观倍数冲高,须用连贯解码或
per-token 命中率校正)。**教训**:共享机器单测 decode 绝对值无意义,必须同窗口配对 / 控 CPU busy%。

**重要方法论结论**:
1. `readback==False` 但 force-prefix 干净 → param `copy_` 非对称(H2D raw 字节、D2H 格式转换),
   **权重落位正确**,raw 回读比对不是有效校验。
2. **短 prompt 输出不可作判据**:CPU(GGUF Q8_0)与 NPU(W8A8)是同一权重两种量化,换常驻集=换量化
   混合 → 欠定 prompt 贪心天然漂移(force-prefix 的短答案同样答非所问)。判质量须强上下文或定量指标。
3. prefill top-K share:真 top-K 0.559(单请求局部性强于聚合分布的 0.395!),prefix 0.133 ✓ 自洽。

**诊断开关(已落 commit)**:`KT_DYN_FORCE_PREFIX` / `KT_DYN_FORCE_SET=shift1|permlayer|scatter|antitop`
/ `KT_DYN_SKIP_WEIGHTS`。判别链已收敛,无需再加变体。

**已关闭(用户定调,不追)**:
- ~~H1/H2 根因追查、δ(E) 量化、teacher-forced logprob~~ —— 不做。精度基线是多卡全-NPU-int8(已验证连贯),
  单卡 hybrid 的 prefix↔real-topK 漂移不是缺陷,不值得管。
- ~~量化粒度验证~~ ✅ 已做(W8A8 per-channel vs Q8_0 per-block-32,128×),保留为旁证,但不构成阻塞。

**后续验收点(真正该做的)**:把"单卡流式 prefill + 动态热专家常驻"的输出**对齐多卡全-NPU-int8 参考**
(而非对齐单卡 Q8_0-heavy)。MXFP4-CPU 落地后同法验收——预期热专家上 NPU 既对齐参考又提速。

### D-阈值. ✅ "长度阈值:短走 hybrid / 长走纯流式"——有价值,但定位是短 prompt 保护(2026-06-10)

**问题**:设一个 prefill 长度阈值 `T`,`S < T` 用现 hybrid、`S ≥ T` 用纯流式 NPU。值不值得?

**数据/原理**:两条路径对 prefill(S token)的 MoE 耗时:
- **流式**:≈ `43 × 271ms(H2D 固定)+ 43×compute(S)` ≈ **11.7s + 小量**,**几乎与 S 无关**(单遍扫全 256 专家,
  H2D 占死)。S=512→~11.8s,S=32k→~13.9s。
- **hybrid**:≈ `0.7ms/token/层 × 43 × S` = **随 S 线性**(实测生产 hybrid prefill ~0.7–0.78ms/token/层,
  因静态 prefix-32 只接 13% 激活、87% 砸 CPU,≈ 全 CPU)。

**交叉点**:`0.7×43×S = 11700ms` → **S* ≈ 390 token**(0.67→406,0.78→350)。
- `S < ~400`:hybrid 更快(且越短优势越大——50 token:hybrid ~1.5s vs 流式 11.7s = **8×**);
- `S > ~400`:hybrid > 11.7s,流式(固定 ~12s)赢,越长赢越多(32k:hybrid ~940s vs 流式 14s = **67×**)。

**结论:有价值,但要看清定位**:
1. **核心价值 = 防短 prompt 踩流式的 11.7s 固定地板**。流式不管多短都先扫一遍全模型权重(11.7s),
   对 100-token 的请求是灾难。阈值把短请求路由回 hybrid(亚秒级)→ **混合流量服务器必须有**。
2. **交叉点 ~400 token 不算小**——真实短请求(单轮对话、短查询)常 < 400 token,这个区间真实存在
   → 阈值**不是没意义**(用户担心的"值很小就没意义"不成立:400 不小,且越短 hybrid 优势越大)。
3. **但对本项目目标场景(长序列 > 定义 context,动辄 32k+)阈值几乎不触发**——长 prefill 永远走流式。
   ∴ 阈值的定位是**通用/混合流量的安全兜底**,不是长序列本身的核心优化。
4. **实现极便宜**:hybrid 路径就是现生产默认,"S<T 走 hybrid" = 啥都不用改;只在 MoE forward 入口加
   一个 `if prefill_len >= T: 流式 else: 现状`。`T` 设服务器参数,默认 **512**(略高于交叉点 + page 对齐)。
5. 附加考量(可选精化):流式还顺带产出 decode 动态热专家池(子目标 2,命中率 ~3× 静态)。若请求要
   decode 很多 token,即便 prefill 长度略低于 400,流式的更好 decode 池也可能整体更优 → 后续可把 `T`
   按"预期 decode 长度"微调,但**先按纯 prefill 速度的 ~512 默认**即可。

⇒ **建议实现**(便宜、防灾、参数化),但文档清楚标注:长序列场景下它基本不触发,主要服务混合流量。

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
3. **[中] prefill 模式选择(长度阈值 `T`,见 §D-阈值)**:`S≥T` 走流式(单遍 layer-at-a-time)、
   `S<T` 走现 hybrid;`T` 设服务器参数默认 512(交叉点 ~400)。短走 hybrid = 现状不改,只在 MoE
   forward 入口加长度分支。解耦 attention chunk 与 MoE 单遍(累全 S hidden 再一次 MoE)。
   定位:主要防短 prompt 踩流式 11.7s 地板(混合流量);长序列目标场景几乎总走流式。
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

### D-MXFP4合入测试(2026-06-11,✅ 通过):CPU MXFP4 + Goal-2 动态常驻 合一版本

**合入**:`mxfp4-cpu-moe` 干净 merge 进 `longseq-prefill` → 分支 `longseq-mxfp4`(commit 213bf34,无冲突;
两者正交:MXFP4=kt-kernel `moe.hpp`+GGUF,Goal-2=sglang)。sglang submodule 指针 bump 到 c850eea7e(real-topK fix)。
MXFP4 .so 复用 kt-D 已构建件(本分支 merge-base 后无 kt-kernel C++ 改动,二进制兼容;手册 §2.4 认可 cp .so 换格式)。
43 层 MXFP4 GGUF 已就绪。

**组合配置**:`KT_GGUF_TEMPLATE=...dsv4_layer{L}_mxfp4.gguf`(CPU 4-bit MXFP4)+ `KT_PREFILL_STREAM=1`(NPU W8A8 流式)
+ `KT_DYNAMIC_RESIDENT=1`(热专家动态常驻,带修复)。

**测试结果(端到端通过)**:
- ✅ **连贯**:dynres prompt 解码结构化连贯无重复(`_layers=43`/`qk_rope_head_dim=64`/`qk_norm="l2"`…)。
- ✅ **事实正确**:acc2 prompt 正确复述"Aurora 官方支持**四种**后端"(=4,正确),连贯续写文档。
- 切换 share=0.559 fired,decode ~6.86 tok/s(单窗口,与 Q8_0 real-topK ~7–9 同量级)。

**结论:MXFP4-CPU + Goal-2 动态常驻 合一可用、精度正确。** 待办:MXFP4 vs Q8_0 decode 同负载配对实测(MXFP4 半字节,
预期 CPU MoE decode 更快~30%);切换 ~157s 成本同前(可优化:批量化 gather)。

### D-MXFP4收益确认(2026-06-11):decode 不提速(vs 优化版 Q8_0),收益在内存砍半
配对实测(prefix-32,同 .so,仅 GGUF 不同,card6/7):decode M(MXFP4) settled ~11 vs Q(Q8_0 优化版) settled ~12
tok/s → **MXFP4 不快,Q8_0 略快**。因 mxfp4 分支也给 Q8_0 上了同款 2.38× 优化(行内预取),MXFP4 半字节带宽收益
被 4-bit 反量化开销抵消(那个 -28~37% 是 vs 旧 Q8_0)。**真收益=内存:GGUF 3.2GB/层 vs 6.4GB/层(总 137 vs 277GB)。**
推论:**decode 提速来自 Goal-2 热专家常驻(~1.5–1.8×),非 MXFP4;MXFP4 是内存账。** 砍 152s 切换仍值得(降 Goal-2 成本)。

### D-砍切换开销(2026-06-11):165→124s,瓶颈是 NZ 设备切片 copy 带宽病态
**Profile(KT_DYN_SWITCH_PROF=1)**:H2D 整池=12.6s(正常),**gather 逐专家=152.8s(瓶颈)**。根因:NZ 设备
切片 copy `slot13[e]→w13[s]` 跑 ~0.3GB/s(HBM 峰值的 ~1/3000),病态。
**修复:ND 往返 gather**(`format_cast NZ→ND` 全带宽 → ND fancy-index → `ND→NZ`)。算子级实测 cos=1.0(与逐专家
bitwise 等价,`nz_batched_gather_test.py`)、空卡 12.5× 快(716→57ms)。**但服务内 format_cast 受 HBM 占用拖累**
(~2.8GB/s vs 空卡 28GB/s,仅 7.9GB free,且 NPU 路径 mem-fraction 不释放 HBM)→ gather 152.8→106.3s,**切换 165→124s**。
decode 保持连贯,OOM 安全(失败回退静态集)。
**要砍到 ~15-25s 需更大改动**:host 端存 ND 池(+277GB host 内存,总 554GB)→ switch 在 host 做 ND fancy-index(对、快)
→ 只 H2D 32 个常驻(34GB,~1.5s)→ NZ-cast 32 个 → 免掉每层整池 6.4GB 设备 format_cast。待定。
