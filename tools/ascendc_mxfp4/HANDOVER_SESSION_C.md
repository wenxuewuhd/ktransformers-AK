# HANDOVER → Session C：用 fused MXFP4→W8A8 算子先验证端到端收益

> 目标读者：Session C。本文给你**一条已跑通的路**：用 Session G 的 fused 算子（已集成、已服务验证）
> 先把端到端收益拿到手——① DDR 省 ~137GB（已实测，复跑确认）；② decode 热专家收益（这部分需要你接
> dynamic-resident，是你的主活）。算子怎么**从 0 编译 + 使用**在 §A/§B，服务怎么起+测在 §C/§D。
>
> 仓库根：`/workspace/code/kt-G-mxfp4kernel`（下称 `$REPO`）。算子目录：`$REPO/tools/ascendc_mxfp4`（下称 `$OP`）。

---

## 0. 一句话 + 现状

把一层 MoE 的 GGUF-native MXFP4 专家权重（fp4 e2m1 codes + e8m0 scale）在 NPU 上**现转**成 W8A8
（int8 + 每输出通道 bf16 scale → FRACTAL_NZ → `npu_fused_experts`），**替掉常驻的 277GB W8A8 流式池**。

- **已做（Session G）**：fused 单核算子（`mxfp4_fused_kernel.cpp`，核子 E=256 ~82ms/层，e2e cos 0.99999976）；
  接进 `kt_stream_prefill` 的 depool 路（gated，默认 off 时 W8A8 路逐字节不变）；起服务跑通、**DDR 省 ~137GB 实测**。
- **★ convert 已优化（2026-06-13）**：整层 `mxfp4_layer_to_nz_slots` E=256 从 **3077ms → 230ms（13.4×）**，
  cos 不退、零契约变更。原瓶颈是 ND→NZ 后处理（de-interleave 79% + int8 转置），现用连续 stack +
  **fp16-transpose**（`q.to(fp16).transpose.contiguous().to(int8)`，因 |q|≤127 字节完全一致）解掉。
  **含义**：① prefill 流式现转收益**找回了**（之前被 NZ-cast 的 137s 吃没，现在每层 ~230ms）；② decode 热专家
  现转切换成本大降（按 230ms/层量级，不是 3s/层）。你拿收益的前提已经铺好。
- **要你做（Session C）**：① 复跑确认端到端收益（DDR + prefill 流式 + decode）；② **decode 热专家收益**——当前
  depool 关掉了 dynamic-resident，decode 走静态 prefix-32 → off_cpu 高。把 dynamic-resident 接到 MXFP4 池
  （热专家现转成常驻 W8A8，用 `convert_proj`）是你的主活。详见 §D。

---

## A. 从 0 编译算子（小白照抄）

### A.0 一次性环境（每个新 shell）
```bash
cd /workspace/code/kt-G-mxfp4kernel
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0      # bisheng + tikcpp 头文件来源
# 选一张空卡（HBM 占用 <10%），不要用 card 2（别的容器）：
npu-smi info | grep -A1 910B3 | grep -E 'HBM|[0-9]+ +/ +65536'   # 找 ~3200/65536 的卡
export ASCEND_RT_VISIBLE_DEVICES=<空卡 id>
```

### A.1 编译 .so（两种方式，任选）
算子是单个 AscendC device kernel + host launcher，编出一个 `.so`，用 ctypes 调（device→device，无 host 往返）。

**方式 1（推荐，自动）**：首次 import 时按需 bisheng 编译并缓存，源码比 .so 新时自动重编。你什么都不用做——
直接进 §A.2 跑测试，它会触发编译。

**方式 2（手动，想自己确认编译链）**：
```bash
cd /workspace/code/kt-G-mxfp4kernel/tools/ascendc_mxfp4
CANN=$ASCEND_TOOLKIT_HOME; TK=$CANN/aarch64-linux/tikcpp
bisheng -x asc --cce-aicore-arch=dav-c220 -O2 -std=c++17 -fPIC -shared \
  -I$TK/tikcfw -I$TK/tikcfw/impl -I$TK/tikcfw/interface -I$TK/tikcfw/lib \
  -I$CANN/aarch64-linux/include \
  mxfp4_fused_kernel.cpp -o libmxfp4fused.so \
  -L$CANN/aarch64-linux/lib64 -lruntime -lascendcl
# => 生成 libmxfp4fused.so（无输出即成功）
```

### A.2 离线验收：单层 bit-exact（先确认算子对，秒级~分钟级）
```bash
cd /workspace/code/kt-G-mxfp4kernel/tools/ascendc_mxfp4
# kernel 层面正确性（int8 eq-frac、oscale 误差、单/多核一致）：
ASCEND_RT_VISIBLE_DEVICES=<空卡> python3 test_fused.py
# 端到端：过真实 npu_fused_experts，对 fp32 golden：
ASCEND_RT_VISIBLE_DEVICES=<空卡> python3 test_fused_e2e.py
# => 期望 cos(fused, fp32-golden) = 0.99999976  PASS
```
跑出 cos≈0.9999998 即算子正确，可进集成。

---

## B. 怎么用算子（API + 已接好的集成点）

### B.1 直接调用（你自己的代码里）
```python
import sys; sys.path.insert(0, "/workspace/code/kt-G-mxfp4kernel/tools/ascendc_mxfp4")
from mxfp4_fused_op import mxfp4_layer_to_nz_slots, convert_proj

# 整层：输入是该层合并后的 MXFP4（device uint8），输出正是 streaming slot + npu_fused_experts 直接消费的张量
w13_nz, s13b, w2_nz, s2b = mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I)
#   c13/s13: w13=cat(w1,w3) codes [E,2I,H/2] + e8m0 scale [E,2I,H/32]
#   c2 /s2 : w2  codes [E,H,I/2]  + e8m0 scale [E,H,I/32]
#   w*_nz: FRACTAL_NZ int8 [E,IN,OUT];  s*b: bf16 [E,OUT]

# 单投影（热专家现转会用到这个，见 §D）：
q_nz, oscale = convert_proj(codes_dev, scale_dev, IN)   # IN=H(w13) 或 I(w2)
```
`convert_proj` 按专家分块（`KT_MXFP4_NZ_CHUNK`，默认 32）以约束 HBM 瞬时占用；只有最终 `[E,IN,OUT]` NZ 是满尺寸。

### B.2 已接好的集成点（depool 路）
改在 sglang 子模块分支 `mxfp4-dequant-kernel-sglang`：
`$REPO/third_party/sglang/python/sglang/srt/layers/moe/kt_stream_prefill.py`
- `KT_MXFP4_DEPOOL=1` → 存 MXFP4（~140GB）替掉 277GB W8A8 池，每层现转（`_load_layer_mxfp4` / `_build_mxfp4_pool` /
  `_streaming_forward` 里的 depool 分支调 `mxfp4_layer_to_nz_slots`）。
- 默认 off 时走原 W8A8 路，逐字节不变（安全回退）。
- **注意**：当前 depool 下 dynamic-resident 被 gate 掉了（`_KT_DYN_RESIDENT and not _KT_MXFP4_DEPOOL`，约 line 562）——
  这正是 §D 你要改的地方。

---

## C. 起服务 + 验证 DDR 收益（已跑通的配置）

### C.0 容器重启后的前置（坑过，必查）
容器重启会丢三样，不补则起不来：
1. **libhwloc15**：丢了会报 "kt_kernel is not installed"。`apt-get install -y libhwloc15`（或确认 .so 在）。
2. **kt-kernel ext**：需重新 build（`CPUINFER_USE_ASCEND_NPU=1`），且 import 需包名注册——
   跑 `tools/p27_ensure_kt_kernel.sh`（已内置软链 kt_kernel→python）。
3. **llama.cpp MXFP4 patch**：重打 `tools/kt_dsv4_npu_patches/llama_cpp/0002-add-ggml-type-mxfp4.patch`
   （GGML_TYPE_MXFP4=39，丢了 GGUF 读 MXFP4 会失败）。

### C.1 启动（单卡，depool 开）
```bash
cd /workspace/code/kt-G-mxfp4kernel
# MXFP4 现转的源权重（safetensors，NPU 只用 safetensors，绝不喂 GGUF 给 NPU）：
export KT_MXFP4_CKPT=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash
export KT_MXFP4_OP_DIR=/workspace/code/kt-G-mxfp4kernel/tools/ascendc_mxfp4   # 可选，默认即此
# depool 开关 + 现转分块 + 给现转留 HBM headroom：
KT_PREFILL_STREAM=1 KT_MXFP4_DEPOOL=1 KT_MXFP4_NZ_CHUNK=32 \
MEM_FRACTION=0.72 NPU_DEVICE_ID=<空卡> \
  setsid bash tools/p27_launch_ds4flash_npu.sh > /tmp/sessionc_depool.log 2>&1 &
# 用 setsid 脱离终端，否则后台长跑服务会被回收（main process 消失 ≠ 崩）。
# 长跑想稳，建议你自己终端前台拉。
```
> `--mem-fraction-static 0.72`（而非默认 0.85）：depool 跳过了 W8A8 槽预留，KV 池会吃光 HBM 导致现转 OOM；
> 降 mem-fraction + `KT_MXFP4_NZ_CHUNK=32` 分块现转给转换留 headroom。

### C.2 量 DDR 收益
起服务前后各看一次 DDR 占用（`free -g` 看 used），对照：
- W8A8 池（depool off）：常驻 **277GB**。
- MXFP4 池（depool on）：约 **140GB**（实测 DDR 326→~475GB）。
- **省 ~137GB**。640-token prefill 应 0 streaming 失败、输出连贯。

---

## D. 验证 decode 热专家收益（**你的主活，未完成**）

现状：depool v1 把 dynamic-resident 关了（它读的是 W8A8 `_POOL`），decode 退回**静态 prefix-32**，
大量专家路由到 CPU → `cpu_moe_wall` 的 `off_cpu` 高（140–330ms，且共享机有噪声）。两件要做：

1. **把 dynamic-resident 接到 MXFP4 池**：用 `convert_proj`（§B.1）把**热专家的 MXFP4 现转成常驻 W8A8**，
   填进 resident 槽，让 real-topK 命中常驻而不是落 CPU。这是拿回 decode 收益的关键。
   - 判据**按 component 看，不看 decode tok/s**（共享机噪声大）：看 `cpu_moe_wall` / `off_cpu` 是否回到 ~20ms 量级。
   - 参考动态常驻已验证的机制（设备切片，别切 host NZ 池——host 切片 format-unaware 会字节错乱）。
2. **pin 税**：v1 另 pin 了 ~140GB MXFP4 池，pin 税只减半没消。要么让 NPU 复用 CPU 的 MXFP4（不另 pin），
   要么流式 unpinned。DDR 收益已拿到；这条是 pin 税的完整收益。

---

## E. 已做 vs 你的活（一页清单）
| 项 | 状态 | 谁 |
|---|---|---|
| fused 算子（82ms/层，cos 0.99999976） | ✅ 完成 | G |
| 接进 kt_stream_prefill depool 路（gated） | ✅ 完成 | G |
| 起服务跑通 + DDR 省 ~137GB | ✅ 实测 | G |
| 复跑确认端到端收益 | ⬜ 先做这个 | C |
| dynamic-resident 接 MXFP4 池（decode 热专家） | ⬜ 主活 | C |
| 消 pin 税（NPU 复用 CPU MXFP4 / unpinned） | ⬜ | C |

---

## F. 纪律（务必遵守）
- **选空卡**：`npu-smi info` 挑 HBM <10%（~3200/65536）的卡；**不要碰 card 2**（别的容器）。
- **别广播 kill**：绝不 `pkill -f sglang.launch_server`（会打到别的容器/session）；只杀自己的 PID/端口。
- 停服务用 **SIGTERM 不要 SIGKILL**（避免 HBM 泄漏）；重启前等 HBM 释放（看 npu-smi 降回基线再拉，否则 avail mem 不足）。
- **NPU 只用 safetensors**（W8A8 / MXFP4），绝不把 GGUF/MXFP4 直接喂给 NPU 路径。
- 重编 C++ 只在本 worktree 的 .so，别动别处。

---

## G. 关键文件
- 算子：`$OP/mxfp4_fused_kernel.cpp`（kernel+launcher）、`$OP/mxfp4_fused_op.py`（ctypes 包装 + API）。
- 验收：`$OP/test_fused.py`、`$OP/test_fused_e2e.py`；规格/golden：`$REPO/tools/mxfp4_w8a8_op/SPEC.md` + `golden.py`。
- 集成：`$REPO/third_party/sglang/.../kt_stream_prefill.py`（分支 `mxfp4-dequant-kernel-sglang`）。
- 现状全记录：`$OP/STATUS.md`。
- 服务启动：`$REPO/tools/p27_launch_ds4flash_npu.sh`（MODEL_PATH 默认 W8A8 模型；depool 用上面 §C.1 的 env）。
