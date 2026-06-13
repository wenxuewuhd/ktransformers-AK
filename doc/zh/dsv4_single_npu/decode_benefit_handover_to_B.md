# Handover → Session B：把 decode 热专家收益变成 wall-clock 提速

> **来源**：Session C（depool + 动态常驻 + prefill 流式 owner）｜**日期**：2026-06-13
> **给谁**：Session B（NPU↔CPU 并行计算效率 + MTP owner）
> **一句话**：热专家机制 C 已做完;受控 A/B 钉死——decode off_cpu **地板已 -45%**,但净提速被
> **每层 CPU↔CPU dispatch/fork-join 固定开销**盖住,**这块是你的域**。把它藏进 overlap,-45% 才变 wall-clock。

---

## 0. 你接什么 / 不接什么

**接(你的活)**：把 hybrid decode 里 **CPU MoE 的 per-layer dispatch/fork-join 固定开销藏进 CPU↔NPU
overlap**,让热专家砍掉的那块带宽收益(地板 -45%)真正变成 tokens/s。这正是你在做的 submit/sync/overlap + MTP。

**不接(已交付,别重做)**：
- **热专家机制**：C 做完(动态常驻,见 §4)。
- **算子**：G 做完(MXFP4→W8A8 fused,convert 230ms/层,cos 0.99999976)。
- **prefill 流式 / depool / DDR**：C 做完(省 137GB,prefill 137s→~15-20s)。

---

## 1. 受控 A/B 实测(别重测,这是判据)

`KT_DYN_FORCE_PREFIX=1`(静态 prefix-32)vs 正常(动态 top-32),**同池 / 同 prompt / 同内存态**,
只换常驻集,304 个 decode token：

| 常驻集 | 命中 share | off_cpu floor | p10 | p25 | median | p75 |
|---|---|---|---|---|---|---|
| static prefix-32 | 0.134 | 18.8 | 29.1 | 39.0 | 63.1 | 103 |
| **dynamic top-32** | 0.586 | **10.3** | 28.0 | 38.6 | 57.0 | 102 |

**读法**：热专家选择有效(命中 4.4×),off_cpu **地板砍 45%(18.8→10.3)**,但 **p10/p25/median/p75 全不动**。

---

## 2. 瓶颈拆解（off_cpu 三段，你打中间那段）

`off_cpu`（KT_DECODE_TIMING 打的 CPU MoE wall）≈
1. **专家带宽分量** —— 随上 CPU 的专家数。热专家把它砍了 → **地板 -45%**。✅ 已拿。
2. **每层 dispatch / fork-join 固定开销** —— 43 层每层 submit→128 线程 fork-join→sync。**不随专家数缩**,
   所以热专家碰不到。**p10 以上被它顶住。← 这是你的靶子。** （mxfp4 handoff 估 ~4.6ms dispatch + 每层 fork-join）
3. **共享机邻居噪声** —— median 是地板的 2-5×,偶发 GC/争用尖峰(见过 2068ms / 6276ms 单 token)。
   **不是代码问题,是测量条件**(见 §5)。

**你的杠杆 = ②**：把 CPU MoE 的 submit/sync 跟 NPU 侧计算 overlap 起来,让 dispatch 不在关键路径上。
②藏住后,①的 -45% 地板才会从 p10 显出来 → wall-clock 提速。

---

## 3. 热专家机制怎么用（**你在 int8 路,不必碰 MXFP4**）

**关键：dispatch 开销 path-agnostic**——在 kt-cpuinfer 的 submit/sync,跟常驻权重来自 W8A8 还是 MXFP4 无关。
所以：
- **你在 int8/W8A8 路**：用**现有的 W8A8 动态常驻**(`KT_DYNAMIC_RESIDENT=1`,已修复 device-slice,见 memory
  `dyn-resident-mechanics-proven`)。real-topK 常驻、解码连贯。**不需要 C 的 MXFP4 depool。**
- C 的 MXFP4 depool(`KT_MXFP4_DEPOOL=1`)是另一条 **gated** 路(默认 off,W8A8 逐字节不变)。你做的 overlap
  对两条路都生效,所以**在 int8 路做就行**。
- 想验热专家地板收益:`KT_DECODE_TIMING=1` + `KT_DYNAMIC_RESIDENT=1`,对照 `KT_DYN_FORCE_PREFIX=1`(静态)。

---

## 4. 仪器 + 判据

- `KT_DECODE_TIMING=1` → 每 token 打 `cpu_moe_wall=X (sync=.. on_cpu=.. off_cpu=..)`(`kt-kernel/python/experts_base.py`)。
  - `off_cpu` = CPU MoE 算非常驻专家的 wall（你要藏的）。`on_cpu` = NPU 侧那部分（~2ms,小）。
- **判据看 floor / p10,不看 median**（共享机 median 是噪声）。你的 overlap 成不成,看 **p10/p25 是否往地板(~10ms)收**。
- 目标方向：让 decode 的 off_cpu **p10 从 ~29ms 往 ~10ms 收**(把 dispatch 藏掉),那就是把 -45% 地板兑现了。

---

## 5. 测量纪律（共享机，硬要求）

- **floor/p10 robust,median 是噪声**；要稳定 median 需 **≥500 token + 尽量独占/空载窗口**。
  -45% 的地板收益**只有在独占机上才显成 wall-clock**——安排一次空载窗口复测是值得的。
- 选 HBM<10% 的空卡(`npu-smi`,~3200/65536)；**不碰 card 2**(别的容器)。
- **绝不 `pkill -f sglang.launch_server`**（打到别的 session）;只杀自己的 PID/PGID。
- 停服务 **SIGTERM 不 SIGKILL**,重启前等 HBM 落基线。
- 长跑服务 `setsid` 或自己前台拉(后台会被回收)。
- readiness 盯 log "fired up",别盯 launcher PID(会假死)。

---

## 6. 代码地图 + 合并对齐

| 文件 | 域 |
|---|---|
| `kt_ep_wrapper.py`（apply / combine `gpu_out+cpu_out`）、`kt-kernel/python/experts_base.py`（submit_forward/sync_forward/dual-stream/host-callback） | **你（B）** —— dispatch/overlap 在这 |
| `kt_stream_prefill.py`（depool / 动态常驻 / 切换） | C |
| `kt-kernel/operators/llamafile/moe.hpp`（`should_skip_expert` live 读 mask、final reduce） | 共享,合并点对齐 |

C 的改动在分支 `mxfp4-dequant-kernel`(父)/ `mxfp4-dequant-kernel-sglang`(sglang),**全 gated 默认 off**,
不影响你的 int8 默认路。**合并 defer 到协调点**(你 overlap 告一段落或要推 depool 生产时一次性整合)。

---

## 7. 验收（你这条线算成了）

- decode **off_cpu p10 从 ~29ms 往 ~10ms 收**(dispatch 藏进 overlap),且在**独占窗口**复测 dynamic vs
  static(`KT_DYN_FORCE_PREFIX`)能看到 wall-clock decode tok/s 提升(热专家地板 -45% 兑现)。
- 判据按 component(off_cpu floor/p10),不按共享机 median。

---

## 附：相关 memory
`dyn-resident-mechanics-proven`（W8A8 动态常驻已修复+机制）、`depool-dynresident-offcpu-floor-only`
（本 A/B 结论）、`static-prefix-placement-is-random-level`、`mxfp4-cpu-moe-validated`、
`npu-bandwidth-bench-needs-warmup`、`dsv4-server-launch-pitfalls`。
