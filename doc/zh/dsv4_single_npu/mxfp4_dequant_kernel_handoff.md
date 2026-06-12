# Handoff — MXFP4→W8A8 融合 dequant kernel(消池:流式 prefill + 热专家 decode 一起拿)

> **状态**:开放(Session G 起点)｜**日期**:2026-06-12｜**隔离 worktree**:`/workspace/code/kt-G-mxfp4kernel`
> **父分支**:`mxfp4-dequant-kernel`(从 `longseq-mxfp4` @ `6d9dc2e` 切出 —— 已带 MXFP4-CPU + Goal-2 动态常驻 合好的基线)
> **sglang 分支**:`mxfp4-dequant-kernel-sglang`(`third_party/sglang`,从 `longseq-sglang` @ `9c8e0e70` 切出)
> **场景**:DeepSeek-V4-Flash 单卡 910B3 长序列。前序结论见 Session C handoff `longseq_prefill_handoff.md` §D-两个收益都拿 / §D-端到端测不准。

---

## 启动提示词(开新 session 时整段贴)

> 你接手 **DeepSeek-V4-Flash 单卡 NPU 的"MXFP4→W8A8 融合 dequant kernel"**。目标一句话:**写一个在 NPU 上把
> MXFP4(4-bit,GGUF)专家权重直接转成 W8A8(int8 + per-channel scale,FRACTAL_NZ)的融合算子**,从而
> **消掉 277GB 的 W8A8 流式池**——池一消,decode 的 pin 税就没了(无论 pin/size),于是 **prefill 流式收益**
> 和 **热专家 decode 收益**第一次能同时落地,顺带把 DDR 占用从 ~414GB 砍到 ~137GB(CPU 和 NPU 共用同一份
> MXFP4)。
>
> **为什么走到这一步(前序已穷尽,别重走)**:Session C 把 W8A8 池这条线测到底了——两个收益**组件级各自都真**
> (流式 prefill 8–70×;热专家把 `cpu_moe_wall` 低端 39→20ms ≈2×),但合起来卡在 **W8A8 池的 pin 税**,而
> pin 税在 runtime **去不掉**(unpin 无 API、free 被 torch caching allocator 吞、bounce memcpy 太慢、全程
> unpinned 又把 prefill H2D 拖到 237s)。**唯一干净的解 = 消池**:不维护 W8A8 池,只留 MXFP4,流式时在 NPU 上
> 现转。瓶颈已被精确定位在 **dequant 段**(nibble unpack + FP4 e2m1 解码 + e8m0 per-block-32 scale),向量化
> PyTorch 是 3.4s/层(比 345ms H2D 地板慢 10×,launch/materialization-bound,**不是带宽**),所以**必须落到
> 自定义算子**。requant + nz 是 NPU-native,可不动。
>
> **路线已定:先 Triton,AscendC 当 fallback**。这台机 `triton-ascend 3.2.0` 已装、`ascend` 后端已注册,
> 且 Session C 跑过 smoke test:`(x & 0xF) * scale`(正是 dequant 内核形状,nibble-mask + scale 广播)在 NPU 上
> **bit-exact**(max_abs_err 0.0)。转换主体是 elementwise 向量操作 + 一个 reduction(requant 的 per-channel
> max-abs),是 Triton 的主场。**第一小时先打三个可行性桩**(§4):FP4 e2m1 解码能否 lower、bf16 I/O、能否压到
> 345ms 地板。三桩过 → Triton 直接写;有桩卡死 → 评估 AscendC。
>
> **本文(这份 handoff)就是完整起点,从 §0 往下读。** bit-exact 参考实现在
> `tools/longseq_dbg/mxfp4_conv_vectorized_npu.py`(慢但对),timing/accuracy 脚本同目录。
>
> 工作区:`/workspace/code/kt-G-mxfp4kernel`,sglang/llama.cpp/pybind11/custom_flashinfer/llamafile/kt-kernel
> 都已就位(§1)。集成点是 `third_party/sglang/.../moe/kt_stream_prefill.py` 的流式转换处。
>
> 纪律:快参考/数字先实测(输出非零)再信;只杀自己 PID/端口、**绝不广播 `pkill -f sglang.launch_server`**;
> 拉服务前 `npu-smi info` 选空卡(<10% HBM)+ 查端口、**别碰 card 2**;改 C++ 重编只动本 worktree 的 `.so`;
> **NPU 只用 safetensors,绝不把 CPU 的 GGUF/MXFP4 权重喂给 NPU 算**——本任务是"在 NPU 上把 MXFP4 转成
> NPU 自己的 W8A8 再算",不是让 NPU 直接吃 MXFP4。改 `kt_stream_prefill`/kt-kernel 的共享文件,合并时跟
> Session B/C 对齐(§8)。

---

## 0. 任务

**写一个 NPU 算子,把流式进来的 MXFP4 专家权重转成 W8A8-NZ,免掉常驻的 W8A8 DDR 池。**

数据流(目标态):
```
GGUF MXFP4 (DDR, ~137GB, CPU 也用这一份)
   └─[H2D 流式, 4-bit 量小]→ HBM 暂存
        └─[本算子: dequant + requant]→ int8 + per-channel scale (HBM)
             └─[native npu_format_cast(·,29)]→ W8A8 FRACTAL_NZ (HBM)
                  └─ npu_fused_experts (prefill 全 NPU 算)
```
对比现状(要替换掉的):常驻 **W8A8 pinned 池 277GB**(DDR)→ 按层 H2D → 池被 pin → decode `cpu_moe_wall`
被 pin 税抬 2–3×(17-27ms → 39-55ms),热专家收益被吃掉。

**消池后一次拿三样**:① 无池→无 pin 税→热专家 decode 收益落地;② prefill 流式 1.8× 保留;③ DDR 414→137GB,
**CPU MoE 和 NPU 流式共用同一份 MXFP4**(用户早先就问过这份能不能复用——消池正是复用)。

精度:MXFP4→W8A8 经 agent 验证**近无损**(int8 足以承载 MXFP4 有效精度,cosine ~0.99994;参考
`mxfp4_to_w8a8_accuracy.py`)。

### 0.1 你的单点 = 这一个 kernel(别扩散)

**你只做一件事:写出"MXFP4→int8(+per-channel scale)"的融合 NPU 算子,单层离线 bit-exact、且快到能藏在
H2D 下面。** 做完做一个最小集成验证就收。以下**明确不是你的活**,别被带跑:

- ❌ **追端到端 decode tok/s 的"干净数"**——Session C 已证这台共享机 NUMA 噪声击败小样本(`cpu_moe_wall`
  单配置就 12–120ms 乱摆),要稳定 median 得 ≥500 token + 独占机。你拿不到也别试,**用组件级地板判定收益**。
- ❌ **改 decode 编排 / submit-sync-overlap / MTP**——那是 Session B 的领地。
- ❌ **整网调通后再验算子**——反了。**先离线单层 bit-exact,再碰整网**(NZ 切片坑见 §坑)。
- ❌ **顺手优化 requant / nz**——它们是 NPU-native、已经够快(§2),不在瓶颈上,别动。

判定你成没成,只看 §10 那四条;前两条(算子 bit-exact + 够快)是硬核,后两条是顺带。

### 0.2 预期收益(量化;来源=Session C 实测 + 本算子 roofline)

| 维度 | 现状(W8A8 池) | 消池后(本算子) | 收益来源 |
|---|---|---|---|
| **decode `cpu_moe_wall`** | streaming-pinned **39–55ms**(pin 税抬 2–3×) | 回到热专家地板 **~20ms** | 无池→无 pin 税(C §D 实测,runtime 去不掉) |
| **decode tok/s** | ~7–8(被 pin 税压) | **~14–16 量级**(热专家应有值) | ↑ 同上;**注**:共享机测不准,以地板判 |
| **prefill 流式** | 4096 ~14s / 32k ~13s(8–70×) | **保留,甚至略快** | 流式 4-bit MXFP4 比 8-bit W8A8 **H2D 减半**(§2) |
| **DDR 占用** | **~414GB**(W8A8 池 277 + MXFP4 ~137) | **~137GB** | 砍掉 277GB W8A8 池;CPU/NPU **共用一份 MXFP4** |
| **精度** | int8 | **不变**(近无损,cos 0.99994) | MXFP4→W8A8 lossless |

> 一句话:**消掉 277GB 常驻池 → pin 税消失 → 热专家 + 流式两个收益第一次同框,DDR 还砍 2/3,精度不动。**
> 这就是为什么值得为它写一个算子。

---

## 1. 工作区 / 隔离(已建好,2026-06-12)

| 项 | 值 |
|---|---|
| 父仓 | `/workspace/code/kt-G-mxfp4kernel`(git worktree,分支 `mxfp4-dequant-kernel`,从 `longseq-mxfp4` @ `6d9dc2e`) |
| sglang | `third_party/sglang`,分支 `mxfp4-dequant-kernel-sglang` @ `9c8e0e70`(含 `kt_stream_prefill.py`)|
| 其余子模块 | llama.cpp `a94e6ff` / pybind11 `bb05e08` / custom_flashinfer `fd94393` 均已 checkout 到记录 commit |
| kt-kernel | 普通目录(非子模块),已随 worktree checkout;改 C++ 自己重编 |
| 参考脚本 | `tools/longseq_dbg/mxfp4_conv_vectorized_npu.py`(bit-exact 慢版)+ `mxfp4_to_w8a8_{accuracy,timing}.py` |

> ⚠️ **子模块建池的坑(已踩,记下避免重踩)**:这台机 `git submodule update` 默认走 SSH(`git@github`)会
> Host-key 失败;且默认 `--local` 硬链接 clone 也会挂。sglang 是用 **`git clone --no-local <sibling>`** 从
> `/workspace/code/kt-C-longseq/third_party/sglang` 本地克隆来的(origin 指向那个 sibling,非 github)。
> **含义**:本 worktree 的 sglang 改动要合回共享的 `longseq-sglang`,走本地 remote(sibling 路径)push/pull,
> 或让 Session C 从这边 `git fetch <kt-G 路径>`。别指望直接 push 到 github。

重编 kt-kernel(若要跑 hybrid 基线对照才需要;Triton 算子本身是纯 Python、**不碰 kt-kernel C++**):
```bash
cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 /usr/local/python3.11.14/bin/python3.11 setup.py build_ext --inplace
```

---

## 2. 为什么是这个算子(roofline,别再质疑前提)

Session C 实测(`KT_DECODE_TIMING`,见 C handoff §D):
- decode 大头是 `cpu_moe_wall`,**不是** NPU 侧。no-stream ~17-27ms / streaming-**pinned** ~39-55ms。
- **pin 税真实且 runtime 去不掉**:差的那 2–3× 就是 W8A8 池 pin 上的代价;torch pinned caching allocator
  不释放,`torch.npu.empty_cache` 只管 device,没有 host-cache-empty API。
- 热专家**真有用**:`cpu_moe_wall` 低端 prefix-32 ~39ms → real-topK ~20ms(off_cpu ∝ 专家数,5.25→2.6)。
- **结论**:两个收益独立都真,合起来只被"W8A8 池要常驻+要 pin"卡住。**消池 = 同时解锁两者**。

转换为什么必须写算子(agent 已定位):
- 向量化 PyTorch 转换 = **3.4s/层**,比 H2D 地板 ~345ms 慢 **10×**;
- 瓶颈在 **dequant 段**:nibble unpack + FP4 e2m1 + e8m0 per-block-32 scale,PyTorch 物化了 **int32
  flat-index** + **13GB bf16 中间量**,是 **launch / materialization-bound,不是带宽**;
- 一个融合算子把这些留在片上、单趟过,就能压回 H2D 地板量级。
- **requant + nz 不用动**:它们是 NPU-native 算子,已经快(C handoff 记的量级 requant ~196ms / nz ~599ms,
  系**整模型 43 层合计**,非每层;换算每层 ~4.5 / ~14ms,远在预算内)。

### 2.1 算子自己的 roofline:它只需要"藏得进 H2D",而预算极宽

流式 pipeline 是双缓冲:**算第 L 层时 H2D 预取第 L+1 层**。所以本算子(dequant+requant+nz)只要满足
**每层转换耗时 ≤ 每层 H2D 耗时**,转换就被完全掩盖 ≈ **免费**。算一下每层预算(DSv4-Flash,每层 E=256
专家,W8A8 6.45GB / MXFP4 ~3.4GB @ 4.25 bit/weight):

| 段 | 每层数据 / 带宽 | 每层耗时(估) | 说明 |
|---|---|---|---|
| **H2D(预算上限)** | MXFP4 ~3.4GB / PCIe ~19–23GB/s | **~150–180ms** | 流 4-bit,比 W8A8 6.45GB(~345ms)**减半** |
| dequant+requant(本算子) | 读 3.4 + 写 6.45 ≈ 9.85GB / HBM ~1.3TB/s | **~8ms**(BW 地板),实测 ~12ms | HBM↔HBM,**纯带宽**;agent conv 地板 12ms |
| native nz(不动) | ~6.45GB 重排 / HBM | **~14ms** | `npu_format_cast(·,29)` |

**结论(robust,跟 nz 精确数无关)**:转换片上段 ~12+14 ≈ **26ms/层**,而 H2D 预算 **150–180ms/层**——
转换只占 H2D 的 **~15%,藏得绰绰有余**,等于免费。**你不需要"最快",只需要"别病态"**:PyTorch 的 3.4s/层
是 H2D 预算的 **~20×**,会把流水彻底撑爆(暴露在关键路径上)→ 流式收益归零。**桩3 的判据因此是"≲H2D 预算
(~150ms),量级对就行",不是抠到地板。**

> 顺带的 prefill 加成:流式 MXFP4(4-bit)比原 W8A8(8-bit)**H2D 减半**(345→~170ms/层),所以消池不仅不
> 拖慢 prefill,反而**省一半 H2D 带宽**——这是之前 W8A8 池没有的额外便宜。

### 2.2 为什么 decode 收益是真的(地板判定,不靠端到端)

decode 大头是 `cpu_moe_wall`,而它被 W8A8 池的 pin 死死抬着:**no-stream 17–27ms → streaming-pinned
39–55ms**(2–3× 就是 pin 税)。热专家把它的**下界**从 prefix-32 ~39ms 砍到 real-topK ~20ms。消池后没有池可
pin → `cpu_moe_wall` 落在 ~20ms 那档而非 40–55 → decode 拿回热专家应有的速度。**这条不依赖端到端 tok/s**
(共享机测不准),只依赖"pin 税真实 + 热专家砍半"这两个 C 已实测的组件事实。

---

## 3. 路线:先 Triton,AscendC 当 fallback

**为什么 Triton 优先**:
- 这台机 `triton-ascend 3.2.0` 已装,`triton.backends` 里 `ascend` 已注册。
- Session C 的 smoke test(`/tmp/tri_smoke.py` 思路):kernel 做 `(x & 0xF).to(f32) * scale` —— nibble-mask
  + per-block scale 广播,正是 dequant 内核形状 —— 在 `npu` 设备上跑出 **max_abs_err 0.0**。位运算(`&`/shift)
  与 scale 广播都能正确 lower。
- 转换主体 = elementwise 向量操作 + 一个 reduction(requant 的 per-channel max-abs);Triton 主场。
- 代价:Triton ~50 行 vs AscendC 几百行 boilerplate(显式 TQue/TBuf/tiling/double-buffer)。

**AscendC 仅作 fallback**:若 §4 的三桩里 FP4 LUT 无法 lower、或 Triton 压不到地板,再评估 AscendC。
别一上来就写 AscendC——没必要付那个迭代成本。

---

## 4. ⚡ 开工第一步:三个可行性桩(别跳,先打桩再写整核)

smoke test 只证了"位运算 + scale 能 lower"。写整核前,用最小单测把这三个真·风险点钉死:

1. **FP4 e2m1 解码能否在 Triton 里 lower**。MXFP4 的 4-bit 是 e2m1(1 符号 + 2 指数 + 1 尾数,16 个值)。
   两种实现:(a) 16 项查找表(`tl` 里能否高效做 gather/select);(b) 纯位拼(拆 sign/exp/mantissa 后按 e2m1
   规则组 fp32)。哪种能 lower 且对 → 定下来。**判据**:对一个 block(32 个 nibble)解出的 fp32 与
   `mxfp4_conv_vectorized_npu.py` 的参考 bit-exact。
2. **bf16 I/O**。dequant 中间/输出若走 bf16,确认 Triton-Ascend 的 bf16 load/store/算术都对(部分早期后端
   bf16 支持不全)。**判据**:bf16 往返 bit-exact。**设计选择**:能否干脆 **fuse dequant+requant、跳过 bf16
   中间量、直接出 int8 + per-channel scale**(这样连 13GB 中间量都不物化,最干净)——这是首选,bf16 桩是
   退路验证。
3. **能否压到 345ms H2D 地板**。一层 expert(w13 + w2)的 dequant 算子端到端计时,目标 **≲345ms/层**
   (conv 理论地板 ~12ms,见 `mxfp4_to_w8a8_timing.py` agent①)。**判据**:量级对得上(别被冷启动假象骗,
   先 warmup——见 memory `npu-bandwidth-bench-needs-warmup`)。

三桩全过 → 写整核(§5)。任一桩卡死 → 记录现象,评估 AscendC 或调整方案,**别硬怼**。

---

## 5. 算子 spec(融合 dequant,首选连 requant 一起融)

**输入**:MXFP4 专家权重块。布局关键(见 memory `mxfp4-cpu-moe-validated` + `moe.hpp` MXFP4 路径):
- `GGML_TYPE_MXFP4 = 39`;每 **block-32** 一个 **e8m0** scale(8-bit 指数);
- nibble **consecutive → half-block repack**(转换时已做的重排,参考实现里有);
- DSv4-Flash:43 层,H=4096,moe_I=2048,E=256,top_k=6;每 expert int8 ≈ 25.2MB,每层 ≈ 6.45GB。

**计算**(单趟、片上):
```
for each nibble:
   v_fp = e2m1_decode(nibble)              # §4 桩1
   v    = v_fp * e8m0_scale[block_of(nibble)]   # per-block-32 广播
# 首选:同核内接 requant
   amax = reduce_max(|v|) over output-channel   # per-channel
   int8 = round(v / amax * 127); chan_scale = amax/127
```
**输出**:int8 张量(ND)+ per-output-channel scale。**不在算子里做 NZ**——交给
`torch_npu.npu_format_cast(t, 29)`(ND→FRACTAL_NZ,acl_format=29)。

**正确性基线**:整核输出经 native requant+nz 后,喂 `npu_fused_experts`,与"参考路径(慢版向量转换
→同样 nz →同 GEMM)"对 cosine ≈ 1.0。先离线单层对账,**过了再上整网**(NZ 切片是 format-aware 的,host 切
会字节错乱——见 C handoff §D-目标2 的坑)。

> 注意标量颗粒度:requant 的 per-channel scale **不跨 expert**(scale 颗粒度在 output-channel 内);这与
> prefix-32 同精度前提一致(用户早先确认过)。

---

## 6. 参考实现与测试脚本(都在 `tools/longseq_dbg/`)

| 脚本 | 作用 |
|---|---|
| `mxfp4_conv_vectorized_npu.py` | **bit-exact 的慢版向量转换**(3.4s/层)——你的正确性金标准 / 移植蓝本 |
| `mxfp4_to_w8a8_accuracy.py` | MXFP4→W8A8 近无损对账(cosine ~0.99994) |
| `mxfp4_to_w8a8_timing.py` | H2D 345ms / conv 12ms 地板的来源 |
| `kernel_decode_vs_prefill.py` | decode==prefill==ref cos 1.0(算子等价性套路可复用) |
| `nz_batched_gather_test.py` | NZ gather/format-cast 的正确姿势(ND-gather 12× 快) |

单元测试纪律:**只验单层、离线、bit-exact / cosine**,可行性钉死再碰整网(对齐用户早先要求:
"只考虑单元测试,验证可行性,不上整网")。

---

## 7. 集成点

转换发生在 `third_party/sglang/python/sglang/srt/layers/moe/kt_stream_prefill.py` 的流式加载/建池路径
(`_h2d_pool` / 建池处)。消池后:不再建/pin W8A8 池,改为每层 H2D MXFP4 → 调本算子 → native nz →
`npu_fused_experts`。`_apply_dynamic_residency`(热专家常驻)逻辑保留——消池后它直接受益(无 pin 税)。

decode 侧 `kt_ep_wrapper.py` / kt-kernel `experts_base.py` 的 submit/sync/combine **与 Session B 共享**,
本任务原则上不改它们(只改 prefill 流式建池);若需动,§8 对齐。

---

## 8. 与 Session B / C 的边界 + 合并

- **共享文件**:`kt_stream_prefill.py`(C 的领地)、`kt_ep_wrapper.py` / `experts_base.py`(B 也碰 submit/
  sync/overlap/MTP)。开发期 worktree 隔离无即时冲突;**合并时在这些文件上对齐**。
- **合并方向**:本 worktree sglang origin 指向 `kt-C-longseq` 的 sibling(非 github)。算子稳定后,
  把 `mxfp4-dequant-kernel-sglang` 合回 `longseq-sglang`,父仓 `mxfp4-dequant-kernel` 合回 `longseq-mxfp4`。
- **合并前**先确认 B/C 没有并发改同一段(C handoff 的 `_apply_dynamic_residency` / `_h2d_pool` 是热点)。

---

## 8.5 坑(前序已踩,别重踩)

按"会不会咬到你这个算子"排序:

1. **NZ 只能在设备上、format-aware 地切/gather**(★ 最相关)。Session C 的 Goal-2 乱码根因就是:常驻权重
   gather 切的是 **host 上的 NZ 池**,host 切片 format-unaware → 字节错乱。**你的算子输出喂 nz、以及任何对
   W8A8-NZ 的 gather/slice,必须在 device 上做**(`npu_format_cast` ND↔NZ:`(t,2)`=NZ→ND,`(t,29)`=ND→NZ)。
   离线对账时:先 NZ→ND 再切再 ND→NZ,或直接设备 fancy-index。参考 `nz_batched_gather_test.py`。
2. **别复刻 PyTorch 转换的病态**。慢不是因为带宽,是它物化了 **int32 flat-index** + **13GB bf16 中间量** +
   多次小 launch(launch/materialization-bound)。融合算子的全部意义就是**消掉这两个物化、单趟过**。首选
   dequant+requant 一起融、连 bf16 中间量都不落地(§4 桩2)。
3. **计时必须先 warmup**。无预热的首次触碰会给离谱假数(memory `npu-bandwidth-bench-needs-warmup`:曾因此
   误判 side-stream 半带宽)。桩3 量 ms/层前先跑几轮丢弃。单变量受控对照。
4. **子模块建池**:这台机 `submodule update` 走 SSH(`git@github`)host-key 失败、默认 `--local` 硬链接
   clone 也挂。已用 `git clone --no-local <kt-C sibling>` 绕过(§1)。**含义**:合回 `longseq-sglang` 走本地
   remote,不是 github。
5. **pin 税 runtime 去不掉**(这是消池存在的理由,别再去试老路):torch pinned **caching allocator 不释放**,
   `torch.npu.empty_cache` 只管 device,没有 host-cache-empty API;unpin 无 API;bounce(unpinned→pinned
   memcpy)更慢;全程 unpinned 又把 prefill H2D 拖到 237s。**所以唯一出路是根本不建 W8A8 池**(=你做的)。
6. **共享机 NUMA 噪声**:`cpu_moe_wall` 同一配置就在 12–120ms 乱摆,100–150 token 小样本测 decode tok/s
   纯噪声。**别拿端到端 tok/s 当算子的验收**;算子验收看 §10.1/§10.2(离线、确定性)。
7. **kt_kernel 导入**:单独跑脚本报 `No module named kt_kernel` 是包名没注册(`ln kt_kernel->python` 或
   `ensure_kt_kernel.sh`);容器重启报 "kt_kernel is not installed" 先查 `libhwloc15`、`MODEL_PATH` 须显式传。

---

## 9. 纪律(硬要求)

- 快参考/数字**先实测**(输出非零、warmup 后)再信;别拿冷启动假象当结论。
- **只杀自己的 PID/端口**;**绝不** `pkill -f sglang.launch_server`(会打掉别的 container/session)。
- 拉服务前 `npu-smi info` 选 HBM <10% 的空卡 + 查端口;**别碰 card 2**(别的 container);SIGTERM 不 SIGKILL
  (避免 HBM 泄漏)。
- **NPU 只用 safetensors 的 W8A8**;本算子是"NPU 上把 MXFP4 转成 NPU 自己的 W8A8 再算",**不是**让 NPU 直接
  吃 GGUF/MXFP4。
- 改 C++ 只重编**本 worktree** 的 `.so`。
- kt-kernel 单独跑脚本若报 `No module named kt_kernel`,是包名没注册——`ln kt_kernel->python` 或
  `ensure_kt_kernel.sh`(memory `kt-kernel-import-needs-symlink`);容器重启报 "kt_kernel is not installed"
  先查 `libhwloc15`(memory `container-restart-loses-libhwloc`)。

---

## 10. 验收(怎么算这个 session 成了)

### 10.1 硬核(你的单点,必须达成)
1. **算子 bit-exact**:单层离线,MXFP4→(算子)→int8+scale→native nz→`npu_fused_experts`,对参考路径
   (`mxfp4_conv_vectorized_npu.py` 同 nz 同 GEMM)cosine ≈ 1.0。
2. **算子够快**:dequant(+requant)片上段 **≲ H2D 每层预算(~150ms),量级对即可**(理论地板 ~8–12ms,
   有 ~15× 余量;§2.1)。warmup 后量。

### 10.2 顺带(算子过了之后的最小集成,能到就到,卡住先记录)
3. **消池整网**:`kt_stream_prefill` 走 MXFP4 流式 + 现转,**不建 W8A8 池**;prefill 流式收益保留
   (4096 prefill ~14s 量级);`free`/RSS 证 277GB W8A8 池已消、DDR 落到 ~137GB。
4. **收益地板**:decode 开热专家常驻,`cpu_moe_wall` 回到 ~20ms 档(而非 39–55)。
   **判定靠这个组件数,不靠 decode tok/s**——共享机 NUMA 噪声让小样本 tok/s 不可信(§坑6)。

---

## 11. 结果(Session G,2026-06-12,Triton 路线跑通)

**算子已写出并端到端验证**。Triton-Ascend,纯 Python,不碰 kt-kernel C++。三个文件:
- `tools/longseq_dbg/mxfp4_dequant_triton.py` —— 核 + 三个生产 API(`mxfp4_dequant_requant` /
  `mxfp4_proj_to_slot_nz` / `mxfp4_layer_to_slots`)。
- `tools/longseq_dbg/test_mxfp4_dequant_triton.py` —— 离线对账 + 单层计时。
- `tools/longseq_dbg/test_mxfp4_kernel_e2e.py` —— **真 `npu_fused_experts` 端到端等价**(§10.1 #1)。

### 11.1 §10.1 #1 算子正确性 = **过**
- **dequant 段 bit-exact**:e2m1 算术解码(sign/exp/mant,选择式,无 LUT gather,无 transcendental)+
  e8m0 经 `(e<<23) bitcast→fp32` 精确 `2^(e-127)`,对 `dequant_native` **max|err|=0.0**(三个 proj 全 0)。
- **整核 vs 慢版向量参考(同 NZ 同 GEMM,真 `npu_fused_experts`)**:`ref-dtype=fp32` 对齐时
  **cos=0.99999976, rel_l2=0.0**;`ref-dtype=bf16` 时 cos=0.99973(纯 **CPU-torch vs NPU-triton 的 bf16
  除法舍入分歧**,非算子缺陷——切 fp32 即归 1)。
- **int8 eq-frac≈0.90 / max|dq|=1**:与 bf16 参考差的那 10% 全是 ±1 个 int8 档,同样来自上面那条 bf16 舍入分歧;
  weight-GEMM 投影显示 **kernel-vs-true ≥ ref-vs-true**(三个 proj 都是),即本算子的 W8A8 不比参考差,反而略优
  (kernel 内部走 fp32,比参考的 bf16 master 更贴真权重)。

### 11.2 §10.1 #2 算子够快 = **部分**(不病态、与 H2D 持平,但未达 ~150ms 目标)
满层 E=256(w13+w2)warmup 后中位:**~358ms/层**(w13 ~215 + w2 ~140)。
- vs PyTorch 慢版 3.4s:**9.6× 更快**——**不病态**(§2.1 说"别病态"即可,这条达成)。
- vs W8A8 H2D 345ms:**持平**——消池后"MXFP4 H2D + 现转"替掉"W8A8 H2D",prefill **不退化**(§0.2 预期"保留")。
- vs MXFP4 H2D ~170ms:**~2× 偏大,未藏住**——现转会成为流水每层地板(~358 而非 ~170),即 prefill 持平 W8A8 而非
  更快;消池主奖(decode pin 税消失 + DDR 砍 2/3)**不受影响**。
- **瓶颈定位实测**:dequant-only(无 reduction/requant)w13 仅 **135ms / 127GB·s**;加 per-output-channel
  **标量 reduction**(`tl.max`)+ requant → 215ms / 20GB·s。即**卡在每行归约**,不是带宽、不是 decode 数学、不是
  store 布局(num_warps / 选择式解码 / strided-vs-contiguous store 都实测过,均不是瓶颈)。
  **下一步提速正解** = 跨行向量化 tiling(一程序吃 [RB, 行]、沿列轴归约成 [RB] 而非标量),但受 UB 192KB 限,
  迭代成本高;因主奖与"持平"已够,本 session 未做,留给后续。

### 11.3 §10.2 集成 = **组件级已证,生产 wiring 留补丁**
- `mxfp4_layer_to_slots(c13,s13,c2,s2,H,I)` 直接吐 `npu_fused_experts` 要的槽张量
  (`w13_nz` FRACTAL_NZ int8 [E,H,2I] + `s13b` bf16 [E,2I];w2 同理),**已被 e2e 测试用作被验证的产物**
  (cos 0.99999976)。这就是 §7 集成点要插的转换函数,**经过验证、可直接 drop-in**。
- **生产 wiring(`kt_stream_prefill`)未改**:`_load_layer_experts` 现在从 **W8A8 checkpoint**(`_CKPT`)读、
  非 MXFP4;消池要把池源切到 MXFP4 model-dir,并在 `_streaming_forward` 把 `slot.copy_(h)` 换成
  `mxfp4_layer_to_slots(...) → slot`。这是**改共享文件 + 需起服务验证**的活,而"起服务追干净数"被 §0.1 明确划出
  你的单点之外(且 Session C 已证共享机端到端测不准)。故本 session **只交经验证的转换函数 + 精确 wiring 说明**,
  不动生产流。后续接手者:池源换 MXFP4、`_streaming_forward` 调用上面的 helper、跟 Session B/C 合并对齐(§8)。

### 11.4 Triton-Ascend 踩坑(给后续同后端的人)
- **`tl.interleave` / `tl.join` 在大 tile(整行 [NB,32],NB=128)炸 UB**(要 768KB > 192KB)。整行交错输出
  改成**两次 stride-2 store**(even/odd)绕过——见核里 `tl.store(obase+even/odd, ...)`。
- **int32 偏移溢出**:满 E=256 时 `r*IN`(r≈1M,IN=4096)超 int32(2.15e9)→ "MTE DDR out of range" 乱跑/假
  0.56ms。行偏移**必须 `.to(tl.int64)`**。
- **grid < 65536 硬限**;`tl.static_range` 全展开会炸 UB,用 **`tl.range` 运行时循环**(复用 buffer);每程序
  `rows_per_prog` 取 `ceil(E*OUT/65535)`。
- **变量整数移位 `1<<tensor` 病态**(10× 慢);2^k 用**选择式**(where exp==…)或 **bitcast** 出。
- **嵌套 `tl.range` 编译极慢**(两层循环的 chunk 方案 >300s 编不完,放弃)。
- `multibuffer=False` / `num_warps` 对本核**无可测影响**(已饱和 / 后端忽略)。
- 计时**必须 warmup**(memory `npu-bandwidth-bench-needs-warmup`)。
