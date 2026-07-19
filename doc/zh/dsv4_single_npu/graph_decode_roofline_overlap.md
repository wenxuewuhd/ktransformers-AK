# Graph decode — roofline 自洽 + side-stream 重叠分解(当前权威)

> **状态**:现行结论（取代两份 2026-06-09 历史档案的定量部分）
> **日期**:2026-07-19 ｜ **硬件**:单卡 Ascend 910C（A3），Kunpeng CPU 40 核 / 1 socket / **1 NUMA**
> **配置**:graph-on 生产态（`KT_SIDE_STREAM=1`、`KT_HOT_TAIL_TOKENS=0`、`MEM_FRACTION=0.82`、
> `KT_CPUINFER=32`、`KT_NUM_GPU_EXPERTS=32`、depool 开、streaming-prefill 阈值 64）
> **前置历史档案**:[graph_decode_profiling_report.md](graph_decode_profiling_report.md)、
> [graph_decode_bandwidth_findings.md](graph_decode_bandwidth_findings.md)（方法论仍有效，端到端数字已过时）

---

## 0. TL;DR

在同一条固定 16k 自然 prompt（temp=0）上，用 A/B/C 三次运行 + decode 算子 profile，验证了两条自洽关系并把 side-stream 的收益拆清楚：

1. **自洽①（roofline）**:CPU-MoE 总墙钟 ↔ 命中率 + 带宽 **对得上** —— 反推带宽 **142 GB/s**，落在 [100, 155 GB/s] 区间、贴微基准峰值。
2. **自洽②（overlap）**:`NPU 时间 + 露出的 CPU = TPOT` 的符号/量级 sanity **通过**（hidden、exposed 均为正且 ∈[0, cpu_moe_wall]）。
3. **side-stream 净收益 9.43ms/token 的真实分解**:
   - **~4.8ms** 是 NPU 真正盖住的 CPU（= resident GroupedMatmul 窗口，和 sparse-expert 算子对得上）；
   - **~4.6ms** 是避开了 `SIDE=0` 的 host 串行惩罚（host 回调挂主流堵流水，×43 层），**不是**"盖住计算"。
4. **attention/NSA 盖不进 side-stream 是数据依赖硬约束**（`self_attn` 在 `mlp` 之前跑完，fork 在 topk 之后），不是调度问题。

---

## 1. 实验设置

同一条**固定** 16k 自然 prompt（License 正文截断 68000 字符 + 提问），`temperature=0`、`ignore_eos=True`、
`max_new_tokens=200`，`KT_HOT_TAIL_TOKENS=0`。三次运行只改一个维度：

| 运行 | 模式 | 关键开关 | 目的 |
|---|---|---|---|
| A | eager | `--disable-cuda-graph` + `KT_HITRATE_PROBE=1` + `KT_DECODE_TIMING=1` | 拿 decode resident 命中率 H |
| B | graph | `KT_SIDE_STREAM=1`（生产）+ `KT_DECODE_TIMING=1` | 拿 TPOT_重叠、cpu_moe_wall |
| C | graph | `KT_SIDE_STREAM=0`（串行）+ `KT_DECODE_TIMING=1` | 拿 TPOT_串行 |

> 命中率与 graph 无关（temp=0 路由固定、resident 集来自 eager 的 prefill），所以 H 用 eager 量、迁移到 graph 口径成立。
> `KT_HITRATE_PROBE` 在 graph capture 下会因 D2H 非法被跳过，故必须 eager 量。

---

## 2. 直接实测量

| 量 | 值 | 来源 / 说明 |
|---|---|---|
| decode resident 命中率 **H** | **26.2%**（52890 slots / 200 步） | A |
| **cpu_moe_wall**（纯 CPU-MoE 计算墙钟，submit→sync） | **18.3ms** | B、C **一致** → 证明与 side-stream 调度无关（side-stream 只影响能否被掩盖，不影响 CPU 本身算多久） |
| **TPOT_重叠**（SIDE=1） | **49.85ms**（20.06 tok/s） | B，服务端 gen throughput 中位 |
| **TPOT_串行**（SIDE=0） | **59.28ms**（16.87 tok/s） | C |

**常量**（本盒实测/推导）:
- MXFP4 单专家 = **13.37 MB**（25.17M 参数 × 0.53125 B/参数）
- 每 token top-k slots = **264.4**（52890/200，≈ 6×43 + shared 项）
- DRAM 带宽硬顶（32 核 / 1 NUMA，`bw_probe_mlp.c` 实测）= **155 GB/s**
- CPU-MoE 微基准实测达成带宽（`p27_cpu_moe_bw_bench.py`，MXFP4 口径）= **108–137 GB/s**

---

## 3. 自洽 ①：CPU 总时间 ↔ 命中率 + 带宽 ✓

CPU 只处理 miss 掉 resident 的那部分专家:

```
miss→CPU 的 slots/token = 264.4 × (1 − 0.262)      = 195.1
CPU 每 token 读的权重     = 195.1 × 13.37 MB          = 2.61 GB
反推带宽 = 2.61 GB / 18.3 ms                          = 142 GB/s
```

**142 GB/s ∈ [100, 155]**，且贴微基准峰值 137、低于硬顶 155 → **CPU 墙钟、命中率、带宽三者自洽**。
（这也再次印证 decode 的 CPU-MoE 是**内存带宽 bound**:batch-1 GEMV，权重只用一次，算术强度 ~3.8 OP/byte，远低于 roofline 拐点 ~33。）

---

## 4. 自洽 ②：NPU 时间 + 露出的 CPU = TPOT ✓

以 SIDE=0/SIDE=1 两个 TPOT 定义 hidden / exposed:

```
hidden_cpu  = TPOT_串行 − TPOT_重叠 = 59.28 − 49.85 = 9.43 ms   （side-stream 相对串行省下的）
exposed_cpu = cpu_moe_wall − hidden = 18.3  − 9.43  = 8.87 ms   （没被掩盖、露在关键路径上的 CPU）
```

**sanity 检查**:hidden、exposed **均为正**且 ∈ [0, cpu_moe_wall]。若 side-stream 有害，hidden 会为负；
若掩盖超过 CPU 本身，exposed 会为负。两者都成立 → **符号/量级自洽,side-stream 确实有净收益**。

> ⚠️ 注意:`hidden + exposed = cpu_moe_wall` 是**定义上的恒等式**（由 hidden、exposed 的定义直接推出），
> 它本身不是独立证据；独立的证据是"两者都落在 [0, cpu_moe_wall] 内且为正"。真正需要挖的是
> **这 9.43ms 到底是不是都被 NPU 计算盖住的** —— 见 §5、§6。

---

## 5. 代码层面:side-stream 到底盖住哪一段（决定性证据）

`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py::apply`（**每个 MoE 层跑一次**）:

```python
fork_ev.record(comp_stream)              # fork 点:进 apply 时 topk 已算完（topk_output 是入参）
side_stream.wait_event(fork_ev)
with npu.stream(side_stream):
    self._submit_cpu_npu_graph(...)      # 侧流:CPU-MoE（D2H 拷入 + host 回调算 miss 专家）
# —— 主计算流同时往下 ——
gpu_combine_input = self.gpu_method.apply(...)   # 主流:resident 专家 GroupedMatmul + MoE 路由
join_ev.record(side_stream)
comp_stream.wait_event(join_ev)          # join 点:combine 之前,等 CPU 回调返回
cpu_output = self.sync(...); output = output + cpu_output   # gather / 合并
```

- **重叠窗口 = `[topk 之后 → combine 之前]`**，主流在窗口内只有 `gpu_method.apply`
  = **resident GroupedMatmul（3.6ms）+ MoE 路由辅助（~1.2ms）≈ 4.8ms**。
- **attention/NSA 结构上盖不进**:decoder layer 里 `self_attn`（`deepseek_v2.py:1791`）先跑完、
  `mlp`（`:1823`）才进 apply、fork 才发生。attention 全在 fork 之前，是**数据依赖**，不是调度没写好。
- 对比 `SIDE=0`:`_submit_cpu_npu_graph` 直接挂在**主计算流**（`kt_ep_wrapper.py:614`），host 回调把它
  后面整条流（D2H、下一层启动）都堵住，逐层 ×43 累积 host 气泡。

---

## 6. Profiler 旁证 + 诚实的局限

decode 算子 profile（`/start_profile`，GPU-only，手动 start→跑完→stop；**不要用 num_steps 自动停**——
graph 模式下它会停在第 1 步并卡死 decode）。

**单算子（eager 短 prompt，上下文无关部分，稳定跨 eager/graph/SIDE0/SIDE1）**:

| NPU 算子（每 decode 步） | 时间 | 能否进 side-stream 窗口 |
|---|---|---|
| **resident sparse-expert `GroupedMatmulV5`** | **3.49–3.62ms** | ✅ 窗口内 |
| MoE 路由（InitRouting+ComputeExpertTokens+DequantSwiglu+FinalizeRouting） | 1.29ms | ✅ 窗口内 |
| MLA/attn 投影 `QuantMatmulWeightNz` | 4.87ms | ❌ fork 之前 |
| NSA `Compressor` | 2.86ms | ❌ fork 之前 |
| NSA `TransposeBatchMatMul` | 2.82ms | ❌ fork 之前 |

**SIDE=1 vs SIDE=0 device 对比（同短 prompt、手动 start/stop）**:

| 每步（profiling 下数值被放大 ~1.6×） | SIDE=1 | SIDE=0 |
|---|---|---|
| device span | 81.1ms | 82.0ms |
| GroupedMatmul（恒定） | 3.62ms | 3.59ms |

> **关键局限**:profiling 下两模式 **span 几乎相等（差 ~1ms），9.43ms 收益在 trace 里复现不出来**。
> 原因:torch_npu profiler 给每个 host 回调加了大量 host 开销，恰好把"side-stream 让 host 回调与主流重叠"
> 这个机制**扰动没了**。（两条 trace 的 sum-of-dur 也不同质:167 vs 121ms/步，说明 profiler 在重扰动，
> union/sum 不能拿来硬算重叠量。）
>
> **方法论坑（务必记住）**:**side-stream 的收益要用无 profiler 的 B/C TPOT 差来量,不能靠 profiler trace。**
> profiler 会破坏 host-callback 的重叠，trace 只能用来量**单算子时长**（那个不受扰动）。

---

## 7. 时间分解:9.43ms 到底是什么

综合"代码给的重叠上界 ≈4.8ms"和"profiler 无法在 trace 里复现 9.43ms"两条:

```
NPU 真正盖住的 CPU  ≈ 4.8ms   = resident GroupedMatmul + MoE 路由窗口(和 sparse-expert 算子对得上)
SIDE=0 host 串行惩罚 ≈ 4.6ms   = host func 挂主流、逐层×43 堵住 host 启动流水(避开它 ≠ 盖住计算)
──────────────────────────────────────
side-stream 净收益 9.43ms/token = 盖住 4.8 + 省掉 host 惩罚 4.6
```

**回答"side-stream 时间能不能和 NPU sparse-expert 算子对上":能——但对上的是那 ~4.8ms（GroupedMatmul + 路由），
不是 9.43ms。** attention/NSA 盖不进是硬约束；side-stream 剩下 ~4.6ms 的收益来自避开 host 串行，
属于"少了一次 host 阻塞"，不属于"NPU 掩盖 CPU 计算"。

> **一处被更正的旧解读**（留档警示）:曾用数字凑出 "hidden 9.43 ≈ MoE block 4.99 + attn 投影 4.87 = 9.86" —— 
> **是巧合，错的**。代码物理上盖不了 attention 投影（在 fork 之前）。正解见本节。

---

## 8. 复现命令 / 工具

- 启动（B 生产态）:见 `tools/p27_launch_ds4flash_npu.sh`，`KT_SIDE_STREAM=1 KT_DECODE_TIMING=1`。
- A（命中率）:加 `EXTRA_FLAGS="--disable-cuda-graph" KT_HITRATE_PROBE=1`，读服务端 `[KT_HITRATE] DECODE`。
- C（串行）:同 B 但 `KT_SIDE_STREAM=0`。
- TPOT:发固定 prompt 驱动 decode，读服务端 `gen throughput (token/s)` 中位 → TPOT=1000/tps。
- cpu_moe_wall:读服务端 `[KT_DECODE_TIMING] cpu_moe_wall=...` 中位。
- decode 算子 profile:`POST /start_profile {activities:["GPU"], record_shapes:false}` → 发短 prompt 跑完 →
  `POST /stop_profile`（**手动 stop，不用 num_steps**），解析 `ASCEND_PROFILER_OUTPUT/trace_view.json`
  的 `Ascend Hardware` pid 事件（用**区间并集**算 device 占用，不能对多流 duration 直接求和）。
- 带宽硬顶:`tools/bw_probe_mlp.c`（`streams=1` ≈ GEMV 口径）。
- CPU-MoE 微基准:`tools/p27_cpu_moe_bw_bench.py`（MXFP4 必须传 `--bytes-per-elem 0.53125`）。

---

## 9. 提高命中率的收益空间（decode 天花板）

用 §4/§7 的生产态（SIDE=1）模型，在这条 16k prompt 上外推：

```
NPU 临界路径 ≈ 36.4ms         (H 无关:attention+NSA+resident GEMM,不被 CPU 掩盖那部分)
cpu_moe_wall(H) = 24.8×(1−H)  (H=26.2% 时 = 18.3ms,线性随 miss 比例缩)
NPU 可掩盖窗口 ≈ 4.8ms         (resident GroupedMatmul + 路由)
TPOT(H) = 36.4 + max(0, cpu_moe_wall(H) − 4.8)
```

| 命中率 H | cpu_moe_wall | 露出的 CPU | TPOT | tok/s | 相对当前 |
|---|---|---|---|---|---|
| **26.2%（当前）** | 18.3ms | 13.5ms | 49.85ms | **20.1** | — |
| 40% | 14.9ms | 10.1ms | 46.4ms | 21.5 | +7% |
| 50% | 12.4ms | 7.6ms | 44.0ms | 22.7 | +13% |
| 60% | 9.9ms | 5.1ms | 41.5ms | 24.1 | +20% |
| 70% | 7.4ms | 2.6ms | 39.0ms | 25.6 | +28% |
| **≈80%（CPU 打平窗口）** | 4.8ms | **0** | **36.4ms** | **~27.5** | **+37%** |
| >80% | <4.8ms | 0 | 36.4ms | ~27.5 | 封顶不动 |

**天花板 ≈ 27–28 tok/s（TPOT ~36ms），在 H≈80% 时到达。** 之后 CPU-MoE 被 NPU 完全掩盖，
再提命中率对 decode **无收益** —— 转为纯 NPU-bound，瓶颈变成 attention/NSA/resident GEMM 本身。

**三个前提**:
1. **H≈80% 不一定物理可达**:只有 32/256 常驻,命中率被路由集中度卡上界。当前动态常驻 ~26–31%,
   hot-tail 长上下文 ~43%。现实近期靠更聪明热池能到 ~50–60% → **22–24 tok/s(+13~20%)**;
   冲 80% 基本得加 `KT_NUM_GPU_EXPERTS`(吃 HBM,32→48 约再十几 GB)。
2. **临界路径随 H 微涨**:H↑ → 更多 slot 落 resident GEMM → GroupedMatmul 变大,36.4ms 会往上飘几 ms,
   真实封顶更可能 **~25–27 tok/s**,不是硬 27.5。
3. **这是 16k 口径**:短上下文 NPU 临界路径更小 → 天花板更高;长上下文更低。

**要突破 36ms 那条线**(而不是逼近它)得动 NPU 侧:NSA compressor/attention 提速、或 resident GEMM 更省。
即:**命中率优化的收益空间是 20→27 tok/s(~+37%),封顶后转战 NPU decode 算子。**
