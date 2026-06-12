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
- 向量化 PyTorch 转换 = **3.4s/层**,比 H2D 地板 345ms 慢 **10×**;
- 瓶颈在 **dequant 段**:nibble unpack + FP4 e2m1 + e8m0 per-block-32 scale,PyTorch 物化了 **int32
  flat-index** + **13GB bf16 中间量**,是 **launch / materialization-bound,不是带宽**;
- 一个融合算子把这些留在片上、单趟过,就能压回 H2D 地板量级。
- **requant + nz 不用动**:它们是 NPU-native 算子,已经快(C handoff 记的 requant ~196ms / nz ~599ms 量级)。

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

1. **算子**:单层离线,MXFP4→(算子)→int8+scale→native nz→`npu_fused_experts`,对参考路径 cosine ≈ 1.0,
   且 dequant ≲ 345ms/层量级。
2. **消池整网**:`kt_stream_prefill` 走 MXFP4 流式 + 现转,**不建 W8A8 池**;prefill 流式收益保留
   (4096 prefill ~14s 量级)。
3. **两个收益同框**:decode 开热专家常驻,`cpu_moe_wall` 不再被 pin 税抬(回到 ~20ms 量级而非 39-55),
   decode tok/s 拿到热专家提速。**注意**:这台共享机 NUMA 噪声大,小样本测不准——要稳定 median 需
   ≥500 token + 尽量独占窗口(C handoff §D-端到端测不准的教训)。
4. **DDR**:`free`/进程 RSS 证 W8A8 池 277GB 已消,DDR 落到 ~137GB 量级。
