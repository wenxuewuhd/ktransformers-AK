# Handoff — 长序列优化:prefill 逐层流式加载 + 热专家预加载

> **状态**:开放(Session C 起点)｜**日期**:2026-06-09｜**隔离 worktree**:`/workspace/code/kt-C-longseq`
> **场景**:序列长度**超过定义 context 长度**的长序列推理。
> **基线**:主干 `dsv4_one_card_dev` @ `22aac3d`(decode 已 `--kt-cpuinfer 128` + GEMV prefetch → ~9.5 tok/s;
> CPU 权重启动加载 ~47s;Q8_0 生产 + NPU graph 已闭合)。

---

## 启动提示词(开新 session 时整段贴)

> 你接手 **DeepSeek-V4-Flash 单卡 NPU 的"长序列(序列长度超过定义 context 长度)优化"**,两个子目标:
> (1) **prefill 逐层流式加载权重**——长序列 prefill 计算密集(M=batch≫1,不像 decode 的 M=1 纯带宽瓶颈)→
> 逐层流式加载专家权重、用计算掩盖加载延迟,(a) 加速长 prefill、(b) 不必全部 ~275GB 常驻、给超长 context 腾内存;
> (2) **热专家预加载/不 evict**——用 prefill 命中保留热专家给 decode。
>
> **本文(这份 handoff)就是你的完整起点,从 §0 往下读。**
>
> 工作区:`/workspace/code/kt-C-longseq`(分支 `longseq-prefill`,独立 sglang 分支 `longseq-sglang`,
> kt-kernel 有 llama.cpp+llamafile 可重编 + 基线 `.so`)。启动脚本自动用本 worktree 的 sglang+kt-kernel,
> **不用 export PYTHONPATH;端口 8013**。
>
> ⚡ **开工第一步(别跳)**:验证子目标 1 前提——测长序列 prefill 的 CPU MoE 是不是计算密集(§3)。
> 是 → 做流式;仍是带宽瓶颈 → 回报换思路。
>
> 边界:热专家标定已被 B 放弃 → **子目标 2 现在 C 独占**;B 现做 NPU/CPU 并行+MTP,会动 submit/sync/overlap
> 编排,合并时对齐(§2)。**实时 expert cache/evict 留作后续单独 session**——C 只把 load/evict 原语 +
> residency 钩子留干净、策略做简单版(§8)。
>
> 纪律:快参考/前提先实测验证(输出非零)再信;只杀自己 PID/端口、**绝不广播 `pkill -f sglang.launch_server`**;
> 拉服务前 `npu-smi info` 选空卡 + 查端口;改 C++ 重编只动自己 worktree 的 `.so`(§6 纪律)。

---

## 0. 任务(两个子目标)

1. **prefill 逐层流式加载权重**:长序列 prefill 是**计算密集**(M=batch≫1 的 GEMM,不像 decode 的 M=1
   纯 DDR 带宽瓶颈)→ 可**逐层流式加载专家权重、用计算掩盖加载延迟**,从而 (a) 加速长 prefill、
   (b) 不必把全部 ~275GB 权重常驻,给超长 context(KV cache)腾内存。
2. **热专家预加载 / 不 evict**:用 prefill 阶段观察到的专家命中,**保留(不驱逐)这些热专家**,
   让后续 decode 直接命中 NPU。

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
