# DeepSeek-V4-Flash 单卡 NPU — graph 模式 decode profiling 报告

> **日期**:2026-06-09 ｜ **分支**:graph_acc(== dsv4_one_card_dev @ afa4666)
> **目标**:graph 模式 decode 还有没有加速空间,用扎实(且**带正确性校验**的)profiling 定位。
> **结论**:有。瓶颈是 **CPU MoE 受内存带宽限,且只用了 192 核里的 24 核**。
> 把 `--kt-cpuinfer` 从 24 提到 **96**,**真实权重端到端实测 3.6 → 6.12 tok/s(~1.7×)**,
> CPU MoE 215→115ms,F2 四 prompt 连贯(精度无损),**纯配置改动、零代码**。
> (cpuinfer=128 真实权重下**会崩**——dummy 扫描的"128 峰值"是 dummy 假象,见 §6。)
> 本报告含**两段诚实纠错记录**(幻象测量 + dummy 不可信于崩溃点)。

---

## 0. 一句话

decode ~280ms/token,其中 ~70% 是 43 层 CPU MoE。CPU MoE 是**内存带宽瓶颈**(每层读 ~144MB int8 专家权重),
线程越多带宽越大、越快;生产 `--kt-cpuinfer 24` **只用了 24/192 核**。提到 ~96 核 → CPU MoE 整 token **194ms → 64ms**。

---

## 1. ⚠️ 纠错:一个幻象测量(诚实记录)

调查中段我犯了一个错误并已纠正,写在最前面警示:

- 我曾用 `wrapper.forward()`(记作 PATH_A)当"快参考",测出 CPU MoE「只要 0.6ms/层」。
- **真相**:隔离微基准里 `forward()` 走 `cudaLaunchHostFunc` 但**没有 ACL callback 订阅者**,host 回调
  **永不触发 → 计算从没执行 → 输出全 0**。"0.6ms" 是在测「什么都不干」。
- 因此一批基于它的结论**全部作废并撤回**:GIL 争用假设、"服务器内慢 9×"、NUMA 绑定救得了、
  "0.6ms 可达"、"协调等待为主"——都错。
- **教训(已纳入流程)**:任何"快参考"先做**正确性校验**(输出非零/对账)再信。下文所有数字均
  用**会真正计算的 `run_pinned_forward_sync`(PATH_B)** 测,且每条都带 `correct=True`(输出 norm≈29.2)。

---

## 2. 关键数据(全部输出校验过)

### 2.1 decode 拆解(graph-on, bs=1)
- token 墙钟 ~280ms(3.6 tok/s);CPU MoE `submit→sync` 实测 ~215ms(占 ~70%,服务器内 `KT_DECODE_TIMING`)。

### 2.2 CPU MoE 单层延迟 vs 线程数(真实 layer-3 权重,变化路由输入,PATH_B,均 correct=True)
| cpuinfer | 每 NUMA 线程 | 单层 median | ×43 (token) |
|---|---|---|---|
| 1 | (单线程) | 76 ms | — |
| 24 | 3 | **4.5 ms** | **194 ms**(= 生产,与服务器实测 ~215ms 吻合) |
| 48 | 6 | 2.67 ms | 115 ms |
| **96** | **12** | **1.48 ms** | **64 ms(3× ↓,甜点)** |
| 192 | 24(满核) | 31.9 ms ⚠️ | 崩(无余量给 NPU host 线程→争抢) |

- 1→2→3→6 线程近似**线性减半** → **内存带宽瓶颈**(读 ~144MB/层 int8 权重),线程=聚合带宽。
- 192(占满 192 核)**崩溃**:不留余量给主线程/NPU host 线程 → 争抢。**必须留头**。

---

## 3. 修复:纯配置,`--kt-cpuinfer=96`(已端到端验证)

- 改动:launch 脚本 `--kt-cpuinfer` 改为可被 `KT_CPUINFER` 覆盖(默认仍 24)。**零业务代码改动。**
- **真实权重端到端实测(NPU 4)**:

  | cpuinfer | gen throughput | CPU MoE/token | F2 连贯 |
  |---|---|---|---|
  | 24(基线) | 3.6 tok/s | ~215 ms | ✅ |
  | **96(推荐)** | **6.12 tok/s(~1.7×)** | **115 ms** | ✅ 四 prompt 全过 |
  | 128 | **崩(1149 ms/token)** | — | ❌ |

- **崩溃悬崖在 96~128 之间**:128 线程(16/NUMA)在真实内存压力 + NPU host 线程争抢下 thrash。
  **96 留 8 核/NUMA 余量,是稳妥甜点**。生产用 `KT_CPUINFER=96`。
- 正确性:只改线程数,**不改数值**;F2 四 prompt 连贯,精度无损。
- ⚠️ 经验:**dummy 权重扫线程数不可信于"崩溃点"**——dummy 路由退化、访存少,128 不崩;真实权重会崩。
  扫最优线程数**必须用真实权重**。

---

## 4. 被证伪 / 撤回的方向(省得再走)

❌ **GIL 争用**(加 `gil_scoped_release` 重编后 tok/s 3.65 vs 3.6,无效,已回退)。
❌ **NUMA 绑定派发线程**(`KT_TQ_NUMA` 扫 + 共置都没用)——基于幻象 PATH_A 的误判。
❌ "CPU 计算只要 0.6ms" / "9× 协调开销" —— **幻象,撤回**。
✅ **真账**:CPU MoE 是内存带宽瓶颈,真实单层(24 线程)~4.5ms,加线程能降。

---

## 5. 产物清单

| 文件 | 性质 |
|---|---|
| `tools/p27_launch_ds4flash_npu.sh` | `--kt-cpuinfer` 支持 `KT_CPUINFER` 覆盖(本次唯一业务改动,极小) |
| `kt-kernel/python/experts_base.py` | env 门控 `KT_DECODE_TIMING` 计时(未提交) |
| `tools/_prof_tmp/*.py` | 输出校验过的 PATH_B 线程扫描微基准(临时,可删/可留) |
| 本报告 | 诊断 + 纠错记录 |
| `ext_bindings.cpp` / `task_queue.*` / `worker_pool.cpp` / `.so` | **已全部回退原状**(GIL/NUMA 实验证伪) |

> 数字均为本会话实测且带正确性校验;文档仅作参考。机器:只用空卡、按 PID/端口管自己的进程、未碰他人。
