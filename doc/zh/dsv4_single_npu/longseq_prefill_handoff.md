# Handoff — 长序列优化:prefill 逐层流式加载 + 热专家预加载

> **状态**:开放(Session C 起点)｜**日期**:2026-06-09｜**隔离 worktree**:`/workspace/code/kt-C-longseq`
> **场景**:序列长度**超过定义 context 长度**的长序列推理。
> **基线**:主干 `dsv4_one_card_dev` @ `22aac3d`(decode 已 `--kt-cpuinfer 128` + GEMV prefetch → ~9.5 tok/s;
> CPU 权重启动加载 ~47s;Q8_0 生产 + NPU graph 已闭合)。

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

## 2. ⚠️ 与 Session B 的边界(开工前必看)

B 正在做**全局频次静态把热专家放 NPU**(改 `kt-kernel/python/experts_base.py` 的
`generate_gpu_experts_masks` / sglang `kt_ep_wrapper` 的 placement)。
**本任务子目标 2 与 B 重叠**——都是"哪些热专家常驻 NPU",只不过 B 是**静态全局频次**,你是**prefill 动态保留**。

- **别和 B 各写一套打架的放置逻辑**。子目标 2 应:**复用 B 的 `gpu_experts_mask` 接口**(B 定静态基线,
  你在其上做长序列动态保留),或**先专注子目标 1**(与 B 正交),子目标 2 等 B 收口后再接。
- 开工前跟协调人确认 B 的进度与接口。
- 子目标 1(prefill 流式加载)与 B 基本正交,可独立推进。

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
- 子目标 2 与 B 的放置改动**合并前先对齐**,避免两套策略打架。
