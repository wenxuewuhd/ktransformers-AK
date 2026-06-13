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

---

# MXFP4(depool)合并工作流 + C 分支清单（备查，2026-06-13）

## 何时合 —— 两个触发,谁先到合谁
1. **depool 的 DDR/prefill 收益要上生产** → 立刻合 C(gated 默认 off,低风险,不碰 int8 默认路)。
2. **B 的 overlap 落地、想 MXFP4 路也吃 decode wall-clock** → B 先落,C 再 rebase 上去合（不急走这条更干净，
   冲突面在 B 重改的 `kt_ep_wrapper`/`experts_base` 的 §D 钩子，让 C 适配 B 的最终版）。

## 分阶段
- **Phase 1（现在）**：C 收口。改动 commit 在分支、gated。B 交接完成。
- **Phase 2**：B 在 int8 默认路做 overlap+MTP（用现有 W8A8 动态常驻），与 MXFP4 解耦、可独立推进。
- **Phase 3（B 落地后，一次集成，建议单开集成 session 带双边上下文）**：
  1. C 的 `mxfp4-dequant-kernel-sglang` rebase 到 B 落地后的 mainline；
  2. 解冲突（只在 `kt_ep_wrapper`/`experts_base` 的 §D 钩子 vs B 的 overlap；`kt_stream_prefill` B 几乎不碰）；
  3. **双路验收（硬动作）**：gate **off** → int8 路逐字节不变；gate **on**(`KT_MXFP4_DEPOOL=1`) →
     cos 0.99999976 + DDR 省 137GB + prefill ~15-20s + §D 解码连贯 + 切换 ~8s；**独占窗口**复测
     dynamic vs `KT_DYN_FORCE_PREFIX` 确认热专家地板 -45% 在两路兑现成 tokens/s。
  4. 落 `longseq-mxfp4`。
- **Phase 4**：默认决策——G 优化后 depool prefill ~15-20s 已接近 W8A8 预建池 14s 且省 137GB，验收过后
  **depool 有资格转默认**（生产决策，Phase 3 数据齐了再拍，别提前）。

## C 分支清单（按图索骥）
**父仓 `mxfp4-dequant-kernel`**（从 `longseq-mxfp4` 切）— C 的 sglang 子模块指针 + 文档。
**sglang `mxfp4-dequant-kernel-sglang`**（从 `longseq-sglang` 切）— depool/§D 主体，关键 commit：
| commit | 作者 | 内容 |
|---|---|---|
| `7fc757933` | G | gated MXFP4 depool 路（存 MXFP4、现转 W8A8-NZ）|
| `710bb2161` | C | §D 动态常驻接 MXFP4 池（热-K 现转进常驻槽，设备安全切片）|
| `4a222a568` | C | `KT_MXFP4_POOL_NO_PIN` 标志（unpinned 池，opt-in）|
| `e14203b2f` | C | 切换 H2D pinned-staging（`_stage_pin_h2d`，17.5→7s）|
| 基线 `c850eea7e`/`9c8e0e70f` | C(旧) | W8A8 动态常驻 device-slice 修复 + ND-round-trip（已在 longseq-sglang）|

## ⚠️ 合并前置（重要）
- **G 的 230ms convert 优化当前 *未提交***（`tools/ascendc_mxfp4/mxfp4_fused_op.py` 工作区 `M`，
  STATUS.md 已记其内容）。**合并前 G 必须先 commit 这个**，否则 prefill 收益（137→~15s）丢失、回到 3077ms。
- kt-kernel ext + llama.cpp MXFP4 patch 容器重启会丢（见 `HANDOVER_SESSION_C.md` §C.0），集成机上须先补。

## 🔜 G 可能还有更高效的算子（待 G 通知）
G 那边可能再出**更高效的算子更新**（如 path-3 核子直写 FRACTAL_NZ，STATUS 估 ~113ms，省 transpose+format_cast）。
**届时有了 G 会告知**，再评估怎么合回来。合的要点不变：**只要 `mxfp4_layer_to_nz_slots` / `convert_proj`
的输出契约不变**（NZ-tagged `[E,IN,OUT]` int8 + bf16 `[E,OUT]`），**C 的 wiring（`kt_stream_prefill`）零改动**，
直接 drop-in 受益；G 若改了输出格式，才需 C 同步改。所以新算子合并 = 替 `mxfp4_fused_op.py`/`mxfp4_*_kernel.cpp`
+ 跑 §A(cos) + §C(端到端) 验收，不动 §D。
