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
| 每层专家 H2D(6.4GB pinned)| ~271 ms |
| CPU MoE baseline(§3.1)| 0.75 ms/token/层(M=32768 → ~24.6 s/层)|
| 交叉点 | **M ≈ 360 tokens,prefill chunk 几乎总是赢** |
| 32k prefill 全程 MoE(43 层流水)| 流式 NPU ~12s vs CPU ~1058s ≈ **90×** |
| NPU 计算(掩盖在 copy 下的条件)| 每层 FLOPs = M×6×2×25.2M;@32k ~10 TFLOP/层,需 >37 TFLOPS 有效(910B3 int8 量级足够,**待实测 grouped matmul**)|
| HBM 双缓冲 | ~12.8GB(2×6.4GB),须从 KV pool 预算里让出(`mem_fraction_static` 调整)|

**待实测前提(下一步)**:910B3 上 W8A8 grouped matmul 每层耗时 @ M=4096/8192/32768
(须 <271ms 才是纯 copy-bound 流水);以及 H2D 与 NPU 计算并发时的互相干扰。

⚠️ 环境坑(容器重启后):`libhwloc.so.15` 会丢 → `apt-get install -y libhwloc15`
(kt_ep_wrapper 把 ImportError 吞成 "kt_kernel is not installed");拉服务须显式传
`MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8`(脚本默认路径错)。

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
