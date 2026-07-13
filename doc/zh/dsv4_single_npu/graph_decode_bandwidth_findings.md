# Findings — CPU MoE 带宽利用率(graph decode 子问题)

> ## ⚠️ 历史档案 —— **端到端数字已过时,勿对外引用**
> 本文写于 **2026-06-09,aclgraph 驱动 bug 修复之前**。文中的端到端 **6.84 / 8.52 tok/s**、
> **cpu_moe_wall 55-67ms** 反映的是 graph 未真正生效时的状态。
>
> **当前实测(2026-07,910C/A3)**:decode **~20-21 tok/s**、**cpu_moe_wall ~16ms**(长 prompt 热专家暖后)。
> **现行数字见总纲 §6.10 / §7.1.1。**
>
> **仍然成立的**:CPU MoE 是**内存带宽 bound**(decode 是 batch-1 GEMV,算术强度 ~3.8 OP/byte,
> 远低于 roofline 拐点)、线程/NUMA 扫描的方法论、TP 对 decode 近乎最优。这些定性结论未被推翻。

> **状态**:阶段性收口｜**日期**:2026-06-09｜**分支**:`cpu-compute-opt`(worktree `/workspace/code/kt-A-cpuopt`)
> **承接**:[graph_decode_bandwidth_handoff.md](graph_decode_bandwidth_handoff.md)(本文证伪了它的核心假设)
> **复现工具**:`tools/p27_cpu_moe_bw_bench.py`(隔离微基准,PATH_B,真实 layer3 权重,输出签名校验)

---

## 0. TL;DR

1. **handoff 的"≥128 线程 thrash 崩"是错的**(是在线服务争抢假象)。隔离实测 128/160 不崩、反而更快;**只有 192(占满 24 核/NUMA)崩溃**(无空闲核给 NumaJobDistributor 自旋线程 + NPU host + python/OS → 过订)。
2. **有效带宽 vs 核数是"128 拐点 + 噪声 plateau"**,不是单峰:96=88, 112=96, **128=114**, 144=109, 160=110, 176=116 GB/s(中位,机器 load~400)。增益主要在 96→128,之后持平。
3. **默认值 96 → 128**(已改 `tools/p27_launch_ds4flash_npu.sh`)。端到端服务(单卡,32 GPU experts):decode **6.84 → 8.52 tok/s(+24%)**,F2 连贯,精度不变。128 与 160 **decode 吞吐相同**(过 128 后 CPU MoE 已和 NPU 重叠/旗鼓相当),128 多留 8 核/NUMA 余量更稳。
4. **当前不是"MoE 自身开销受限",而是被邻居挤占的内存上限**:同 128 线程部署、同负载窗口,**MoE@128 ≈ 纯流式 probe@128 ≈ 120 GB/s** —— MoE 的计算已被内存流式完全掩盖。
5. 因此 **dot-ILP/blocked-GEMV(#3b)、减屏障(#2)对带宽无效**(它们不是 128 下的限制因素)。**prefetch(#3a)+2–3%**(免费,保留待定)。**hugepage(#4)≈0%**(顺序流式 TLB 友好;且容器禁用 THP)。
6. 突破内存上限只剩两条,都在本优化线之外:**Q4 砍字节** / **热专家放置(更多专家上 NPU)**。

---

## 1. 线程数扫描(隔离微基准,真实权重,输出签名逐字节一致 = 全对)

| cpuinfer | 每 NUMA | 中位 ms | 中位 GB/s | 最快 GB/s | 空闲核/NUMA |
|---|---|---|---|---|---|
| 96 | 12 | 1.83 | 88 | 94 | 12 |
| 112 | 14 | 1.67 | 96 | 104 | 10 |
| **128** | **16** | **1.41** | **114** | 118 | **8** |
| 144 | 18 | 1.47 | 109 | 113 | 6 |
| 160 | 20 | 1.46 | 110 | 116 | 4 |
| 176 | 22 | 1.38 | 116 | 129 | 2 |
| 192 | 24 | 80+ | **1.7(崩)** | — | 0 |

**knee 在 128;128–176 是噪声 plateau。192 占满核必崩。**

## 2. 端到端服务验证(NPU_DEVICE_ID=6, port 8011, `--kt-num-gpu-experts 32`)

| cpuinfer | cpu_moe_wall/token | gen 吞吐 | client tok/s | F2 |
|---|---|---|---|---|
| 96 | 67.7 ms | 8.63 | 6.84 | — |
| **128** | 55.1 ms | ~9.6 | **8.52(+24%)** | 通过 |
| 160 | 51.5 ms | ~9.6 | 8.50 | 通过 |

128↔160 decode 相同 → 过 128 后 CPU MoE 不再是唯一瓶颈(和 NPU 重叠)。**纯运行时改 `KT_CPUINFER`,不重编、不动 C++、精度不变。**

## 3. 192 崩溃机理

24 线程/NUMA 时全部物理核被 STRICT 绑给 InNumaPool 计算线程(core 0–23),8 个 NumaJobDistributor 自旋线程 + cpu_infer TaskQueue + NPU runtime/host + python + OS 抢不到核 → 过订;`worker_pool.cpp` 的忙等(`wait()`:119 / `do_numa_job`:378)饿死真正 worker。160(20/NUMA)留 4 核即可避免。

## 4. latency 与带宽对账(自洽,差 ~4%)

| | 每层 ms | 字节/层 | 反推带宽 |
|---|---|---|---|
| 微基准 @128 | 1.411 | 160 MB(top-6 全 CPU,最坏) | 113.7 GB/s |
| 服务端 @128 | 55.1/43=1.281 | ~140 MB(6×224/256=5.25 专家,平均) | ~109.5 GB/s |

服务端每层更快正因平均只搬 140MB(1.28/1.41=0.91 ≈ 140/160=0.875)。两点外推 `lat=bytes/BW+F` → 边际带宽 ~157 GB/s、**固定开销 F ~0.39ms/层(占 ~30%)**(两点外推+140MB 是估计,仅作"固定开销非零"的提示;要 per-phase timing 才靠谱)。

## 5. 内存上限对账(决定 #2/#3b 是否值得)

同 128 线程、同负载窗口(load~420):

| | 带宽 |
|---|---|
| 纯流式 probe @128(`/tmp/bw_probe2.c`,16-累加器 ILP) | 121.4 GB/s |
| MoE @128 | ~120(中位)/127(最快) |

**相等** → MoE 计算完全被内存掩盖,卡在和 probe 同一个**被邻居(load~420)挤占的内存上限**。⇒ **#3b(blocked GEMV/dot-ILP)、#2(减屏障)对带宽 ~0 收益**(不是限制因素)。

> 注:probe@96 spread=221、probe@128 spread=121 的非单调是 spread 绑定+清净窗口假象,非硬件性质;之前"2.4× MoE 开销"即源于拿 221 这个清净峰值做不公平对比,已纠正。

## 6. 已验证的优化项

- **#3a prefetch(+2–3%,保留待定)**:`moe.hpp` Q8_0 GEMV 内对下一权重行预取 4 条 cache line,`KT_NO_MOE_PREFETCH=1` 门控。A/B(128):ON 120.9 vs OFF 118.3(均值)。免费、低风险、纯 kt-kernel;吵机器上收益被内存上限淹没,清净机器上可能更大。**尚未 commit。**
- **#4 hugepage(排除)**:THP 全量生效(prctl 重开 + MADV_HUGEPAGE,AnonHugePages=4094MB)下 42.0→42.1 GB/s,**~0%**。顺序流式 TLB 友好(1 miss/512 行)。且本容器默认禁 THP(`THP_enabled:0`,`HugePages_Total:0`)。
- **#1 NUMA-local 权重(已闭合)**:`set_memory_to_numa` 用 `hwloc_set_membind(BIND|STRICT|THREAD)`,load 经 `do_numa_job` 跑在各 NUMA worker 上 → 权重严格绑本地。架构是 TP 切 intermediate 维(每 NUMA 存所有专家的 1/8 列),非"每专家钉一节点"。

## 7. 机器环境(关键背景)

4-socket Kunpeng-920,192 核(无 SMT),8 NUMA × 24 核;ISA:`asimd+asimddp+fp16`,**无 SVE/i8mm/AVX/AMX**。int8 GEMV 因 iqk/tinyBLAS 在 dotprod-only 上 NaN → 回退 ggml `vec_dot_q8_0_q8_0`(NEON+vdotq_s32,competent)。**机器长期 load ~400(共享机,多租户),所有绝对带宽是被邻居挤占后的下限。** 理论 DDR4 spec 751 GB/s;清净单 NUMA 实测 ~45 GB/s → 干净聚合上限 ~360 GB/s(spec 的 ~48%)。

## 8. 工具(复现)

- `tools/p27_cpu_moe_bw_bench.py` — 隔离 decode 带宽/线程扫描(已在仓库)。
- `/tmp/bw_probe2.c`(16-累加器纯流式)、`/tmp/bw_probe4.c`(+THP 开关) — STREAM 类带宽探针(临时,待决定是否收进 tools/)。

## 9. per-NUMA 带宽不均(跨-NUMA 屏障的税),且 non-stationary

单-NUMA probe 分别钉到 8 个节点(24 核,best-of):

| 轮次 | N0 | N1 | N2 | N3 | N4 | N5 | N6 | N7 |
|---|---|---|---|---|---|---|---|---|
| 首测 | 44.6 | 45.4 | 43.5 | **28.3** | 43.8 | **29.0** | 43.8 | **25.3** |
| A | 44.6 | 45.2 | **29.0** | 27.8 | 36.4 | 44.0 | 41.3 | 26.2 |
| B | 42.4 | 45.8 | 41.9 | 27.5 | 42.8 | 39.8 | 34.1 | 24.4 |
| C | 45.3 | 46.2 | 39.7 | 40.0 | 44.3 | 38.4 | 34.6 | 25.5 |

慢节点(~25-29 GB/s vs ~44)**逐轮漂移**(N2/N3 时慢时快;只有 N7 持续慢)。MoE 的 `do_numa_job` 是**屏障,每层等最慢的 NUMA** → 等 1/8 列均分时,被最慢 socket gate,瞬时税 ~1.3-1.5×。但因为慢节点**非平稳**(邻居驱动),**静态按速度切列不可行**(快照几秒后就过时);动态方案被"权重按 NUMA 物理钉死 + decode 顺序依赖"卡住(快节点不能廉价地算慢节点的列)。⇒ **该税真实存在但在本共享机上不可鲁棒回收。**

## 10. 与 STREAM 基准对账(读 vs 读+写,关键)

外部 STREAM(`-mcpu=native`, 100M elem):192 线程 Triad **208 GB/s**、Copy 225;taskset 48 核(1 socket)Triad **95.5 GB/s**。
**关键:STREAM 计的是读+写流量**(Copy=2×、Triad=3× 元素流量);**MoE 权重流是只读**。折算 STREAM 的"只读分量"(Triad 读占 2/3 ≈ 139,Copy 读占 1/2 ≈ 112)→ **~112-139 GB/s @192**。MoE@128 只读 ~120,probe@192 只读实测 143 —— **完全吻合**。即:208/380 是读+写口径,只读工作负载够不着;**MoE 已经在真实的"只读"上限上**。STREAM 也佐证 NUMA 绑定的重要(208 unbound vs 380 ideal),而 **MoE 本来就是 NUMA-local 严格绑核的"好案例",不吃 unbound 惩罚**。

## 11. EP(专家并行)across-NUMA 可行性评估

把不同 expert 整份放到不同 NUMA、做 EP(而非现在的 TP 切 intermediate 维):
- **对 decode 不利、大概率更差**:decode 每 token 只激活 ~5.25 个 CPU 专家、随机散落 8 NUMA → 多数 NUMA 空转 → **浪费 8 通道的内存并行度**(而 TP 每个 token 都让全部 8 个内存控制器满负荷);且 per-token 路由不均(最忙 NUMA gate)。**TP 对 decode 近乎最优**(每 NUMA 算每个专家的 1/8,完美均衡 + 全通道)。
- EP 唯一上风(每个活跃 NUMA 用满 24 核、免 TP 归约)被"空闲节点 + 路由不均 + 高方差"抵消。
- **EP 利好 prefill**(批量、专家全命中、均衡、免归约)—— 若将来 prefill 成瓶颈,可走 "prefill-EP + decode-TP" 混合(DeepEP 式);但当前瓶颈是 decode。

## 12. 下一步

软件利用率track 在本共享机上**基本到顶**(只读上限 ~120-140 已达)。真正能动的:
1. **Q4 砍字节**(只读流量腰斩 → ~2× 上限,撞墙也有效);
2. **热专家放置 / 调大 `--kt-num-gpu-experts`**(把字节挪去 NPU);
3. **环境**:独占/低负载机器可恢复到 ~读上限更高处(但 §1 plateau 显示 128-176 已平,纯加核也有限)。
