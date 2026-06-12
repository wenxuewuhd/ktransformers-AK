# Handoff — worker 线程池 park 阈值优化（Session F）：消除 decode 冷启/抖动掉速

> **状态**：开放（Session F 起点）｜**日期**：2026-06-11｜**隔离 worktree**：`/workspace/code/kt-F-workerpark`
> **目标**：消除 decode 「冷启 ~8 → 连发升 16 → 空闲/抖动又掉回 8」的反复，让 serving 稳定在高 tps。
> **基线**：主干 `dsv4_one_card_dev`（MXFP4，graph-on，稳态 ~16 tok/s）。

---

## 启动提示词（开新 session 时整段贴）

> 你接手 **DeepSeek-V4-Flash 单卡 NPU 的「worker 线程池 park 阈值优化」**。现象：MXFP4 服务 decode
> 冷启只 ~8 tok/s，连发几个 prompt 升到 ~16，但只要空闲一下或单 token 抖动就掉回 8、再发又升。
> 根因已定位（见 §1）：kt-kernel 线程池 worker 空闲 **>50ms 就 park**，而 decode 正常 token 间空闲 ~40ms，
> 余量太薄 → 任何抖动越过 50ms 就 park、下个 token 付唤醒代价又慢 → 抖动自维持。
>
> **本文（这份 handoff）是完整起点，从 §0 往下读。** 工作区 `/workspace/code/kt-F-workerpark`
> （分支 `workerpark-tune`，独立 sglang 分支 `workerpark-sglang`，本任务**只改 kt-kernel C++**，sglang 不动）。
> 启动脚本自动用本 worktree，**不用 export PYTHONPATH；端口 8022**。**改 C++ 必重编 .so**。
>
> ⚡ 开工第一步：先用 `KT_DECODE_TIMING=1` 拉服务、复现并量化现象（冷启头几 token vs 稳态的 cpu_moe_wall
> ramp），坐实 park 假说再动手（§3）。别凭读码下结论。
>
> 纪律：A/B 只改阈值这一个变量、真实权重、同窗口对比；端口 8022，只杀自己 PID，绝不广播
> `pkill -f sglang.launch_server`；拉服务前 `npu-smi info` 选空卡；长跑服务自己终端前台拉。

---

## 0. 工作区

| 项 | 值 |
|---|---|
| 仓库 | `/workspace/code/kt-F-workerpark`（worktree，父分支 `workerpark-tune`，自 `dsv4_one_card_dev`） |
| sglang | 独立 clone `workerpark-sglang` @ `456687a0f`（**本任务不改**） |
| kt-kernel | llama.cpp+llamafile 齐（**改 C++ 自己重编**），基线 `.so` 已就位 |
| 端口 | **8022**（A8000/B8012/C8013/D8020/E8021） |
| 重编 | `cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace` |

## 1. 根因（代码铁证，两处）

`kt-kernel/cpu_backend/worker_pool.cpp` 的线程是**混合 spin-park**：空闲 **<50ms 继续 busy-spin**
（保持热、唤醒延迟 0），**>50ms 才 `cv.wait()` park**（futex sleep，让出核）。**两层都有同一阈值**：

| 落点 | 线程 | 行 |
|---|---|---|
| `InNumaPool::worker_thread` | NUMA 内 16 个 compute worker | `worker_pool.cpp:226` `if (duration > 50)` |
| `NumaJobDistributor`（dispense loop） | 跨 8 NUMA 的 distributor | `worker_pool.cpp:402` `if (duration > 50)` |

⚠️ **两处必须一起改**——只改一层，另一层仍 park、唤醒延迟仍在。

**机制**（解释用户现象）：
- 稳态 16 tps：TPOT~62ms，CPU MoE 占~22ms，token 间 worker 空闲 ~40ms **< 50ms** → 全程 spin、热（正反馈）。
- 冷启/空闲后首发：worker 已 park → 唤醒 128 线程（8 NUMA×16）+ OS 重调度回核（共享机争抢下 ms 级）+
  cache 被邻居污染需重 warm → 慢；头几 token 间隔 >50ms 又 park → 恶性循环卡 ~8 tps。
- **关键洞察**：50ms 离 ~40ms 正常空闲只 ~10ms 余量 → 任何单 token 抖动（NPU 慢、邻居抢核、page cache
  refault 单层几百 ms）越过 50ms 就 park 一次 → 抖动**自维持** → 连发也会无故掉回 8。
- park ≠ 权重冷启（138GB 权重一直在 DRAM/page cache）；"冷"是**线程调度延迟 + cache locality**。

## 2. 方案

把 50ms 阈值**拉长**（如 500ms~2s）：正常 decode 的 token 抖动都不越线，只有真正长空闲才 park。

**权衡（务必处理，不是不做）**：
- **空闲 spin 烧 CPU**：阈值拉长后空闲那段 128 核 busy-spin 占 100%，共享机和邻居抢核/吃配额。
  ⇒ **别设无限**，设个"够覆盖最坏 token 抖动、用户停下来仍能让出核"的值（建议先扫 200ms/500ms/1s/2s）。
- **首次冷启躲不掉**：刚拉服务那一次 park→唤醒仍在（除非开机常驻 spin），但只一次，不影响连续体验。

**实现选项（开工前可定）**：
1. 最小：两处 `50` 改成一个常量（如 `kPARK_IDLE_MS=500`）。简单，但写死。
2. 可配：从 `WorkerPoolConfig` / cpuinfer 配置传入阈值（serving 大、礼貌小）。**注意**与开源 clean code（E 线，减
   env）目标对齐——优先走**构造参数/配置**而非新 env；若必须 env，先和用户确认（E 线只保留计时类 env）。
3. 进阶：两段式（idle 短→spin、中→低频 yield poll、长→park），兼顾热与 CPU 礼貌。先做 1/2 量化收益再决定。

## 3. 验证（先复现量化，再改）

1. **复现现象**：`KT_DECODE_TIMING=1 KT_CPUINFER=128 NPU_DEVICE_ID=<空卡> PORT=8022
   KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf'
   MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 bash tools/p27_launch_ds4flash_npu.sh`。
   冷启发一发、停 1–2s 再发、连发——看 `cpu_moe_wall`（sync/on_cpu/off_cpu）：
   - **park 假说证据**：冷启/停顿后头几 token `on_cpu`（host 侧，含唤醒/submit）偏高，几 token 内 ramp 到
     ~1.5–2ms 稳态；`off_cpu` 同步 ramp。连发稳态后保持低。
   - 区分：单层孤立 spike 几百 ms = page cache refault（次因，非本任务）；普遍平高 = 带宽/DVFS。
2. **A/B 改阈值**：两处 `50` → 500/1000/2000ms 各重编一版，同窗口同 prompt 序列对比：连发是否不再掉回 8、
   停顿后再发是否直接 16、cpu_moe_wall 方差（median/p95−min）是否收窄。
3. **CPU 礼貌代价**：测空闲期 `top`/loadavg——确认拉长阈值后空闲 spin 占核多少，定一个收益/礼貌平衡点。
4. **回归**：`PORT=8022 bash tools/p27_curl_f2_prompts.sh` 连贯 + mxfp4 layer16 cosine 对账不变（纯调度改动，
   数值必须 bit 不变）。

## 4. 量化目标

- 连续发 prompt **不再无故掉回 8**，稳定 ~16 tok/s；停顿后再发**直接回 16**（首次冷启那一下除外）。
- cpu_moe_wall 方差收窄（p95−min 明显下降）。
- 空闲 spin 的 CPU 占用在可接受范围（给出数字 + 推荐阈值）。

## 5. 纪律

- 先 `KT_DECODE_TIMING` 复现量化、坐实 park 假说，再改（别凭读码下结论）。
- A/B 单变量（只改阈值）、真实权重、同负载窗口；共享机挑清净窗口或标注 load。
- ISA 红线无关（纯调度）；改 C++ 重编只动本 worktree `.so`。
- 端口 8022；只杀自己 PID；绝不 `pkill -f sglang.launch_server`；长跑服务自己终端前台拉（后台会被回收）。

## 6. 合并

- 纯 kt-kernel C++（`worker_pool.cpp`）→ 可并回 `dsv4_one_card_dev`，合后主 checkout 重编 `.so`。
- 若做成可配，**与 E 线（开源 clean code，减 env）对齐**实现形态（优先构造参数，非新 env）。
- 相关：[mxfp4_cpu_moe_handoff.md](mxfp4_cpu_moe_handoff.md)（带宽/瓶颈背景）、E 线
  `opensource_release_handoff.md`（已把本项列为 roadmap 待办）。
