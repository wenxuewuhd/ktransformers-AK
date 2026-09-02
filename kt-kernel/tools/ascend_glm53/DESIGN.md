# 设计说明：为什么是这个形状

GLM-5.3-Flash 在**一张昇腾 910C die** 上服务，路由专家 offload 到主机 DDR。
本文解释这个系统为什么长成现在的样子——容量怎么算出来的、时间花在哪、
哪些约束不是选择而是被逼的。怎么用看 [`README.md`](./README.md)，
精度怎么建立的看 [`ACCURACY.md`](./ACCURACY.md)。

---

## 1. 问题的形状

GLM-5.3-Flash：45 层（0–2 dense，3–44 MoE），**288 个路由专家 + 1 个 shared**，
top-8，`moe_intermediate_size` 2048，`hidden_size` 4096，34 层 KDA + 11 层 DSA，
外加 1 层 MTP（第 45 层，不服务）。

一张 910C die 有 64 GiB HBM，**约 61.3 GiB 可用**。INT8 W8A8 checkpoint 是 306 GiB，
其中 **290.6 GiB 是路由专家**。所以：注意力、3 个 dense 层和一小部分专家留在 die 上，
其余专家以 MXFP4 常驻主机 DDR，由 kt-kernel 的 `LLAMAFILE` MoE 在 CPU 上算。

```
routed_experts   290.63 GiB   ← 43 个带专家的层（3..44 服务 + 45 是 MTP）
attn_norm_other   10.81
embed + lm_head    2.36
dense_mlp(0-2)     1.08
shared_experts     1.01
mtp(非专家部分)     0.20
─────────────────────────
TOTAL            306.09 GiB
```

## 2. 容量账：K=32 是算出来的，不是试出来的

实测拟合（三次加载，残差 0.13 GiB）：

```
die 上权重(K, 是否有流式槽) = 15.60 + 6.75·[流式槽] + 0.9925·K   GiB
```

**每个专家序号跨 42 个服务层 = 0.9925 GiB。**
⚠ 早期文档写的 1.009 GiB 是错的——那把不服务的第 45 层（MTP）也算进去了。

| K（常驻） | 16 | 24 | **32** | 40 |
|---|---:|---:|---:|---:|
| die 上权重 GiB（hybrid，无流式槽） | 31.6 | 39.7 | **47.8** | 55.8 |
| 61.3 可用中剩余 | 29.7 | 21.6 | **13.5** | 5.5 |

K=32 留 13.5 GiB 给 KV、KDA conv/mamba state 和图 buffer；K=40 只剩 5.5 GiB，
放不下有用的上下文。

**流式配置下这笔账更紧**：流式 prefill 要预留一个能装下一层全部 288 个专家的
W8A8-NZ 转换槽（6.75 GiB），而且是**在 KV 定容之前**预留。K=32 + 流式槽实测
54.11 GiB，图捕获后剩约 5.5 GiB 动态余量。

⚠ **K 和 chunk 大小花的是同一份余量。** KDA 层在多千 token 的 prefill 里要向 Triton
要一块 ~3 GiB workspace，所以流式路径的 `--chunked-prefill-size` 不能是 8192
（实测 OOM，见 README）。**两者不能分开读。**

**CPU 侧**：逐层 GGUF 每层 3.586 GiB × 42 = **151 GiB 磁盘**；
运行时只有非常驻的 256/288 进 NUMA buffer ≈ **134 GiB RAM**。

## 3. 时间花在哪：CPU MoE 已经在带宽 roofline 上

### 先说度量口径，因为它能把结论整个弄反

⚠ **`wall / generated_tokens` 不是 decode 速率。** 它把 prefill 摊了进去，而这条路上
每个 prefill chunk 都付全额 CPU MoE。630 token 提示下，它把 **52.5 ms/token 的真实
decode 报成 89 ms/token**，让一个健康的系统看起来慢 40%。

用两个生成长度做减法，同一个提示，prefill 精确抵消：

```
wall(n) = prefill + n·decode   ⇒   decode = (wall(256) − wall(64)) / 192
```

（warmup 是前提：一次冷 prefill 会整个落进被减项。）两次独立测量差 0.8%，
在 load 30 的机器上依然稳。

### Roofline 对账

```
CPU MoE 字节/token = 42 层 × 8 × (256/288) 非常驻 × 12.75 MiB = 3.99 GB
  在实测的 150 GB/s 下                                        = 26.6 ms
NPU decode                                                    = 33.3 ms
─────────────────────────────────────────────────────────────
完全串行                                    59.9 ms → 16.7 tok/s
完全重叠                                    33.3 ms → 30.0 tok/s
实测                                        52.5 ms → 19.0 tok/s
```

实测比串行界快 7.4 ms —— side stream 掩盖了约 12%。

**和 DeepSeek-V4 交叉验证**（同一公式，两个模型）：DSV4 算得 18.7 tok/s、文档实测
18–20；GLM 算得 16.7、实测 19.0。⚠ GLM 每 token 比 DSV4 多读 **32%** 的字节
（top-8 vs top-6，288 vs 256 专家），理论上就该更慢。

### Profile：42 次停顿，一个位置

`--prefill-settle 20` 保证窗口里没有 prefill 批次。device 31.10 ms/step、
wall 52.50、host 21.40（40.8%）。间隙直方图（阈值 50 µs）：

```
间隙 ≥50µs：44.1 个/step，合计 21038 µs
  其中 42.0 个、合计 20567.8 µs 全在同一处：
      after aclnnMoeFinalizeRoutingV2 → before aclnnAdd
```

**每个 MoE 层恰好一次，每次 489.7 µs**，位置就是 CPU 专家输出加回 hidden states 的地方
——`sync()` 在等 CPU。**不是几处大停顿，是 42 次固定的每层代价。**

两边独立算，吻合到 30 µs：

```
CPU 每层字节 95.0 MB ÷ 150 GB/s = 633.3 µs   ← roofline
实测暴露的停顿                   = 489.7 µs
  ⇒ 被掩盖                      = 143.6 µs
NPU 侧 MoE 块（可掩盖的全部）    = 113.8 µs
```

两条结论：

1. **CPU MoE 已经跑在带宽 roofline 上** —— 633 µs 就是 95 MB ÷ 150 GB/s，没有浪费。
2. **重叠也已用满结构上限** —— CPU 任务在同一层内 submit 再 join，能掩盖的只有那一层
   常驻专家的活（114 µs），占 CPU 的 18%。

⚠ 这解释了**为什么 side stream 只值约 12%**：它没有失效，可掩盖的量本来就只有这么多。

### 天花板

**device 的 31.1 ms（≈32 tok/s）是硬上限。** 低于它再优化 CPU 侧没有意义。
唯一的一阶杠杆是**减字节**——CPU 侧 26.6 ms 里全是字节。常驻命中率从 prefix 的
32/288 ≈ 11% 提上去能线性减字节：

| 常驻命中率 | CPU/层 | 暴露/层 | decode | tok/s |
|---:|---:|---:|---:|---:|
| **11.1%（prefix）** | **633 µs** | **490 µs** | **52.5 ms** | **19.0** |
| 30% | 499 | 355 | 46.8 ms | 21.4 |
| 50% | 356 | 212 | 40.9 ms | 24.4 |

⚠ **能不能达到取决于路由有多偏，这是可测量的，不要外推。** 均匀路由下命中率就是
11.1%，那么热专家放置一分钱也赚不到。动态热专家的实测结果见 `ACCURACY.md`。

## 4. 一个会让所有测量作废的陷阱：NUMA 页放置

⚠ **这是开发机独有的，目标机不会有——但它会污染在开发机上做的每一次测量。**

独立实测（24 GiB 缓冲、`mbind(MPOL_BIND)` 验证过页确实落位、NEON 读核、每点 5 次）：

| 线程在 | 缓冲在 | 32 线程读带宽 | 惩罚 |
|---|---|---:|---:|
| node0 | node0（本地） | **150.8 GB/s** | 1.00× |
| node0 | **node1** | **154.7 GB/s** | **无惩罚** |
| node0 | node2 / 3 / 4 / 7 | **~20.3 GB/s** | **7.4×** |
| node0 | 全 8 节点交织 | 61.7 GB/s | 2.4× |

⛔ **8 个节点两两配对成 4 个快速域：(0,1) (2,3) (4,5) (6,7)，跨对砍到 ~20 GB/s。**
⚠ **`/sys/devices/system/node/*/distance` 报均匀的 `10 20 20 ...`——它是错的，且不警告你。**

⚠ **跨对读长得和"带宽饱和"一模一样**：4–8 线程就饱和，加到 40 线程完全不动
（1t 11.9 → 8t 20.9 → 40t 20.1）。一条漂亮的饱和曲线可能只是页放错了地方。
**先查页放置，再谈带宽。**

排除项（都实测过，都不是问题）：随机 12.75 MiB 块读 157 GB/s，不比顺序慢——
专家散射的访问模式不要钱；4 KiB 页 vs 2 MiB 大页只差 2%。

**我们踩的就是这个。** `--kt-threadpool-count 1` 时 kt-kernel 走"一个 subpool 独占整个
张量就别名 mmap"的优化（省 154 GiB 内存），于是权重留在 page cache 当初被谁 fault
就在谁那儿——而 GGUF 是 `--jobs 32` 跨全节点转出来的，页散在 8 个节点上，
**只有 27% 在快速对内**。subpool 数 > 1 时不满足"独占"，改成拷进节点本地缓冲：

| 构型 | 256 token |
|---|---|
| `threadpool=1`（别名 mmap，页散布） | 3.99 tok/s |
| **`threadpool=2`（node 0+1，节点本地拷贝）** | **11.26 tok/s** |

同样 32 线程、同样常驻数，只改了页落在哪：**2.8×**。

⚠ **所以在这台机器上用 `threadpool=1` 模拟目标机是无效的**——它复现的是一个目标机
根本不会有的跨节点病态访问。正确的代理构型是 `threadpool=2` 绑到一个快速对。
⚠ 反过来，**别把开发机的 `threadpool=2` 搬到目标机**：目标机单 NUMA 节点、
229 GB 内存，别名 mmap 在那里既正确又省 154 GiB。

## 5. 流式 prefill：为什么是"整层流进一个复用槽"

超过 `KT_PREFILL_STREAM_THRESHOLD` 的 prefill chunk 不走 hybrid：整层 288 个专家从
GGUF 流进一个**复用的** HBM 槽，MoE 全部在 NPU 上算，不做 CPU 往返。

几个不是选择而是被逼的约束：

- **单槽复用，不做双缓冲。** 双缓冲实测只值 ≤2%，却要两倍 HBM——在第 2 节那笔账下
  没有第二个 6.75 GiB。
- **逐 chunk H2D，不整层暂存。** 原先把整层 288 个专家的 GGUF 块一次 H2D 当暂存区
  （3.586 GiB），改成在块循环里逐块搬，峰值降到约 `3.6 × chunk/E`。这在一个同时还要
  找 ~514 MiB 转换 transient 的 prefill 上是决定性的。
- ⛔ **常驻掩码是权重写入的 COMMIT，不是记账。** 权重在该层被流式处理时立刻改写；
  如果掩码推迟到最后一层才提交，一次在第 L 层的中断（转换 OOM——正是超预算时会发生的）
  会让每个 < L 的层拿着专家 `top[i]` 的权重、而掩码还说槽 `i` 是专家 `i`。
  **算错专家、专家 i 没人算、什么都不抛。** 所以流式过程的**每一个出口**都提交掩码。
  回归测试 `test/registered/moe/test_kt_stream_resident_commit.py` 守住这一条。
- ⚠ **每个异常都被吞掉并回退 hybrid。** 这是当前设计：坏掉的流式路径从外面看和好的
  一模一样，唯一的证据是日志里的 `inline resident` 计数，`verify.sh` 会查。

**TTFT ≈ chunks × 18.0 s + 0.49 ms/token** —— 主项是 **chunk 数**，不是 token 数，
因为每个 chunk 都要重流全部 288 个专家。所以它对短 prompt 是负收益（交叉点实测
约 1100 token），阈值交由业务按自己的 prompt 分布配置。

## 6. 和 DeepSeek-V4 配方的差异

同源，但这几条不能照抄：

| | DeepSeek-V4-Flash | GLM-5.3-Flash |
|---|---|---|
| `--page-size` | 128 | **64**（DSA pool 有 `assert page_size == 64`）|
| `--attention-backend` | `ascend` | **不传**，让 GLM 自己选 KDA / DSA |
| MoE 层 | 0..42（`first_k_dense_replace=0`）| **3..44**，另有层 45 是 MTP，不转 |
| 专家 / top-k | 256 / 6 | **288 / 8** |
| shared expert | 本就不融合 | **必须显式关掉**，否则模型加载即失败 |
