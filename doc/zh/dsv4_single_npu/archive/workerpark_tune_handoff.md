# Handoff — worker 线程池 park 阈值优化（Session F）：消除 decode 冷启/抖动掉速

> **状态**：**已关闭 — park 假说被受控 A/B 证伪（2026-06-12）。不要再走调阈值这条路。** 见文末「Closeout」。
> ｜**日期**：2026-06-11 起 / 2026-06-12 结｜**隔离 worktree**：`/workspace/code/kt-F-workerpark`
> **目标**：消除 decode 「冷启 ~8 → 连发升 16 → 空闲/抖动又掉回 8」的反复，让 serving 稳定在高 tps。
> **基线**：主干 `dsv4_one_card_dev`（MXFP4，graph-on，稳态 ~16 tok/s）。

> ⚠️ **结论先行**：把 `worker_pool.cpp:226/:402` 的 50ms 阈值拉到 2000ms（env 可配、两处都改、重编 .so），
> 单变量同协议 A/B 显示**对 decode 吞吐/首 token 延迟/停顿后重发/稳态抖动均无可测差异**，且有**礼貌代价**
> （sub-2s 间隔活动把 128 核钉死）。§1 的 park 机制（"稳态每 token 55ms gap 越线 park"）**实测为假**。
> 代码已 **revert 回基线、.so 重编、git 干净**。下文 §1–§5 是原始假说（保留作记录），以文末 Closeout 为准。

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

---

## Closeout（2026-06-12）— park 假说被受控 A/B 证伪，未合入任何改动

**做法**：`worker_pool.cpp:226` 与 `:402` 两处 `duration > 50` 改成 `kt_park_idle_ms()`（env `KT_PARK_IDLE_MS`
可配，默认 50 保持基线 bit 不变），重编 `.so`。单变量、真实权重（DSv4-Flash MXFP4，cpuinfer128，单卡 910B3）、
同一 prompt/协议，A/B 对比 50ms vs 2000ms。env 生效已验证（见下）。

**复现的现象（真实存在）**：`KT_DECODE_TIMING=1` 下，冷启/停顿后首 token `cpu_moe_wall` 尖峰 300–650ms，
其后多 token ramp 回稳态 ~20–25ms；连续 decode ~12–15 tok/s。**关键**：`on_cpu`（host 侧 submit/wake）全程平 ~2ms，
冷启成本全部落在 `off_cpu`/`sync`（worker 计算墙）——与 handoff §3 预测的"on_cpu 偏高"**相反**。

**A/B 结果（50ms vs 2000ms，同协议）**：

| 指标 | 50ms | 2000ms | 判定 |
|---|---|---|---|
| 连发 20-tok ×5（1s 停顿） | 2.78/2.13/2.00/2.05/1.87s | 2.75/2.14/1.95/1.92/2.25s | 噪声内相同 |
| 背靠背 20-tok ×3 | 1.79/2.05/1.97s | 1.90/1.83/1.80s | 相同 |
| 连续 150-tok 稳态 median | 24.1ms | 20.0ms / 另一次 25.0ms | 两次 2000ms 互差 ≥ 50↔2000 差 |
| active-decode CPU | ~129 核 | ~129 核 | 相同 → 连续 decode 根本不 park |

**为什么 park 机制是错的**：
1. **连续 decode 根本不 park**：active CPU 两档都 ~129 核（满）。CPU MoE 每 token 按 43 层逐层 submit，
   worker 两次 submit 间空闲 <50ms，永远到不了 park 阈值。§1 的"稳态每 token 55ms gap 越线 park"不成立。
2. **首 token 尖峰不是 park**：300–650ms 尖峰在 50ms 和 2000ms **都在**（2000ms 下 worker 明明没 park）。
   它是 prefill→decode 的 graph/transition 成本，阈值动不了。
3. **env 确实生效**（排除"改了个寂寞"）：2000ms 下单 token 后 1s 窗口 scheduler 仍烧 **129.8 核**（worker 还在 spin 没 park）；
   50ms 同窗口 ~1 核（已 park）。即阈值确实改变了 park 行为——只是这行为对吞吐无影响。

**raise 阈值的唯一净效果 = 礼貌代价**：任何 sub-2s 间隔的活动会让 128 核全程 100%（共享 192 核机不友好），零吞吐收益。
真正长空闲（>2s）两档都 park、都回到 ~1 核，所以拉阈值连"省冷启"都不省。

**冷启 8↔16 的真因**（都不是 park）：① per-request 首 token transition；② 会话级 cache/TLB/NPU-graph 暖机
（第一发慢、整会话渐热）；③ 共享机邻居争抢（实测 loadavg ~40/192、邻居 NPU 卡 AICore 100%）→ 大 run-to-run 方差。
要治冷启应往 **keep-warm**（首 token transition / graph 常驻 / 会话级预热）方向，不是 park 阈值。

**落地**：worker_pool.cpp 已 `git checkout` 回基线（两处 `> 50`、无 env）、`.so` 重编、`git status` 干净，无残留。
建议 E 线 `opensource_release_handoff.md` 把"park 阈值可配"从 roadmap **删除**。原始数据 `/tmp/wp/results.md`。
（教训印证启动提示词的"别凭读码下结论"——读码看着像 park，实测不是。）

---

## Follow-up（2026-06-12）— 去掉 `--skip-server-warmup` 实测救开机冷启 −37%（已采纳）

park 证伪后顺着 keep-warm 方向查：当前 `tools/p27_launch_ds4flash_npu.sh` 一直传 `--skip-server-warmup`，
即**跳过 sglang 开机预热**。去掉它 → 开机时多跑一次 dummy decode pass，把 NPU graph + cache 提前暖好。

**A/B（20-tok 请求，每臂 boot 两次，card 4 vs 6，同协议）**：

| 请求 | skip（基线）run1/run2 | warmup 开启 run1/run2 |
|---|---|---|
| **req1（开机第一发）** | 3.43 / 3.31s（~3.37） | **2.28 / 1.98s（~2.13）** |
| req2 | 3.28 / 2.13 | 1.86 / 1.74 |
| req3 | 2.63 / 1.70 | 1.84 / 1.59 |
| 60-tok 吞吐 | 11.0 tok/s | 13.9 tok/s |

**结论**：开机第一发 **−37%（~1.2s）**，两次 boot 区间不重叠；到稳态从 req4 提前到 req2。开机冷启的慢**不在
cpu_moe_wall**（skip req1 的 cpu_moe_wall 仍只 25–65ms），在 NPU 侧 graph/prefill 一次性建立——正是 warmup 救的。

**边界**：server warmup 只在**开机跑一次**，所以只治**开机冷启**；对会话中"空闲一下又掉回 8"的复发冷启**无效**
（那需要周期性 keep-warm ping，另做）。代价：开机多几秒一次性预热，无运行时代价。

**落地**：脚本 `--skip-server-warmup` 已改为 `${SKIP_WARMUP:-1}` 门控——**默认 1 保持基线**，serving 建议设
`SKIP_WARMUP=0` 开预热。这是本任务唯一保留的改动（worker_pool.cpp 已 revert）。数据见 `/tmp/wp/results.md`。
