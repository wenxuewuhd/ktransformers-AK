# DSv4-Flash + kt-kernel MoE offload 迁移 Ascend 950 的工作量评估

> 本文提到的补丁、脚本、日志等产物在复现工作区 `/mnt/workspace/dsv4-repro-2159/`，未随本文档入库。

调研日期 2026-08-26 ｜ 基线：本工作区在 Atlas A3 单 die 上已跑通的方案

## 结论先行

1. **`sgl-project/sglang` PR #33030 是对口的 950 支持 PR，但尚未合入，CI 红。**
   标题 `[NPU] add Ascend 950 (Atlas A5) backend paths for DeepSeek-V4`，作者 `AndyLi429`，
   state=**open**，2026-07-31 开，最后更新 2026-08-25，21 commits / 14 文件 / +1755−73，
   base=`sgl-project:main@0c7ff19e3`。有一条 `ping1jing2` 的 CHANGES_REQUESTED（2026-08-06）。
2. **950 上 MXFP4 routed 专家原生可算，不再需要 MXFP4→int8 那一层。** 已核实到具体接口。
3. **净效果是工作量减少** —— 前提是 A5 版 CANN 自定义算子包已就绪，而这是唯一的硬阻塞。
4. **粗略量级**（假定 A5 算子包可得）：

   | 路线 | 人天 | 置信度 |
   |---|---|---|
   | 最小可跑（关掉流式 prefill + 动态热专家） | **15–25** | 中 |
   | 完整对齐现状（保留流式 prefill，迁到 MXFP4） | **35–55** | 中低 |
   | A5 算子需自己造 | 不可估，+数月 | — |

---

## 1. PR #33030 给了什么

### 1.1 器件判别只有一行

`hardware_backend/npu/utils.py` 新增：

```python
@functools.lru_cache(maxsize=1)
def has_npu_a5_support() -> bool:
    if not is_npu(): return False
    import torch_npu
    return torch_npu.npu.get_device_name(0).startswith("Ascend950")
```

所有 A5 分支挂在这一个 gate 上；PR 声称 910B/910C 行为逐字节不变。

### 1.2 原生 MXFP4 routed 专家（对我们影响最大）

新文件 `hardware_backend/npu/quantization/fp4_moe_methods.py`（+547 行），类 `NPUW4A4Fp4MoEMethod`。
`create_weights` 直接按 **block-32 MXFP4 原生布局**开权重：

```python
MXFP4_BLOCK_SIZE = 32   # fixed by the msmodelslim export format
w13_weight       = Parameter(empty((E, 2*I, H // 2),  dtype=torch.uint8))   # 每字节两个 FP4
w13_weight_scale = Parameter(zeros((E, 2*I, H // 32), dtype=torch.uint8))   # E8M0 指数字节
```

计算核心是 **W4(MXFP4 权重) × A8(FP8-e4m3 激活)** 的 grouped matmul：

```python
x, x_scale = torch.ops.npu.npu_dynamic_mx_quant(input, block_size=32,
                                                dst_type=torch.float8_e4m3fn)
torch.ops.npu.npu_grouped_matmul(
    [x], [weight], antiquant_scale=[weight_scale],
    per_token_scale=[_pair_pack_mxfp_act_scale(x_scale)],
    x_dtype=torch.float8_e4m3fn,
    weight_dtype=_get_float4_e2m1fn_x2_dtype(),
    per_token_scale_dtype=_get_float8_e8m0fnu_dtype(), ...)
```

挂载点 `layers/quantization/fp8.py:get_quant_method`：
`if self.is_fp4_experts and is_npu() and has_npu_a5_support(): return NPUW4A4Fp4MoEMethod(...)`

顺带：该文件的 `_apply_swiglu_limit_npu()` **明确实现了非对称 clamp** 并在三条 dispatch 路径上调用
——即 A5 的 NPU routed 专家**会** clamp。（与我们 A3 上的 clamp 结论有关联，见 §5。）

### 1.3 A5 相对 A3/910B 的差异

| 维度 | A3/910B（现状） | A5/950（#33030） | 出处 |
|---|---|---|---|
| routed 专家算力 | 无原生 MXFP4 | **原生 MXFP4 GMM** | `fp4_moe_methods.py` |
| KV cache dtype | bf16（nope+rope 打包） | **`float8_e4m3fn`**，per-64 scale，pad 到 128B | `dsv4_memory_pool.py` `a5_packed_kv_dim` |
| KV 写入 | `buf_flat[loc] = cache` | `torch.ops.npu.kv_compress_epilog(...)` | `_write_a5_packed_kv()` |
| indexer KV | `int8` + `float16` scale | **`float8_e4m3fn` + `float32`** | `dsv4_memory_pool.py` |
| indexer 写入 | `set_compress_buffer` | `torch.ops.custom.indexer_compress_epilog` | `ascend_dsv4_backend.py` |
| sparse attn 算子 | `custom.npu_sparse_attn_sharedkv{,_metadata}` | **`custom.npu_kv_quant_sparse_attn_sharedkv{,_metadata}`** | `_sparse_attn_ops()` |
| compressor cache_mode | 只能 `cache_mode=1` | **`cache_mode=2` CYCLE** | `_build_cycle_state_block_table()` |
| indexer Q 量化 | `npu_dynamic_quant`（int8） | `npu_dynamic_block_quant(dst_type=float8_e4m3fn)` | `_forward_npu_fused` |
| dense linear | block-FP8 / deep_gemm | **MXFP8 GEMM** | `linear_method_npu.py` |
| DSv4 `wo_a` | deep_gemm FP8 | `npu_transpose_quant_batchmatmul` | `deepseek_v4.py`、`fp8.py` |
| `npu_hc_post` | 非 batched | **A5 build 是 batched** | `deepseek_v4.py:hc_post` |

---

## 2. 「950 原生 MXFP4」省掉了什么

| 环节 | A3 现状 | 950 | 判定 |
|---|---|---|---|
| NPU 常驻专家格式 | W8A8 int8 | **MXFP4 + E8M0** | **消失** |
| MXFP4→int8 AscendC 算子 | `tools/ascendc_mxfp4/`（598+375 行 + bisheng 编译链） | 不需要 | **删掉** |
| `KT_MXFP4_DEPOOL` + `_apply_resident_layer_depool` + `_mxfp4_convert_fn()` | `kt_stream_prefill.py` 里最复杂的一段 | 权衡不存在了 | **删掉** |
| `KT_MXFP4_GGUF_DEDUP` | 为在两份格式间去重 | 只有一份格式 | **删掉** |
| NPU 侧 checkpoint | 单独的 `DeepSeek-V4-Flash-W8A8/` | 原始 FP4 checkpoint 直读 | **删掉一份权重** |

**能否共用同一份 MXFP4 权重？逻辑上可以，物理上还需一层轻量重排。**
`NPUW4A4Fp4MoEMethod` 的块几何（32 元素/块、E8M0 scale）与 GGUF 的 `block_mxfp4`
（`loader.py:85` = 32 元素 / 17 字节）**完全一致**；差别只在排布：GGUF 把 scale 与 codes
交织在同一个 17 字节块里，NPU 要求分离成两个张量、scale 再做 `[E,N,K/32] → [E,K/64,N,2]`
配对重排。需要写的是**纯 layout 的去交织 + 重排 shim**（无反量化、无重量化），
估 **2–4 人天**，比现在那个 AscendC 转换算子便宜一个数量级。
⚠️ 这是推断 —— 官方 FP4 checkpoint 的确切张量命名/布局未直接核对。

**显存**：常驻专家从 1 B/权重降到约 0.53 B/权重，**约减半**。加上 950 HBM 更大，
可常驻专家数应显著超过现在的 32 个 —— 这会反过来削弱流式 prefill / 动态热专家的必要性。

**动态热专家换入不会更简单，但也不会更难。** FRACTAL_NZ 仍在（`process_weights_after_loading`
里还是 `npu_format_cast(..., FRACTAL_NZ, customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2)`），
但多了两个 kwargs，且 A5 上 NZ cast 产出的是 **`FRACTAL_NZ_C0_16`**（与 910 的 C0 不同）。
直接后果：`kt_stream_prefill.py:533 _inplace_nz()` 那个「同一块 pinned 内存上原地 ND→NZ」
的技巧**大概率会碎**（它假设 NZ 字节数 == ND 字节数）。
我们踩过的 `index_select(out=)` 静默丢写坑仍然适用，修法 `.copy_()` 是格式无关的，继续有效。

---

## 3. 逐块工作量

### 几乎不用改（0–1 人天/项）
- `kt-kernel/cpu_backend/vendors/ascend_npu.h`（64 行 CUDA shim）、`ascend_callback_worker.{h,cpp}`、
  `cpuinfer.h`、`ext_bindings.cpp` —— **零 SoC 耦合**，全部 ACL 符号只有 10 个通用运行时 API
  （`aclrtSubscribeReport` / `aclrtProcessReport` / `aclrtLaunchCallback` 等），
  无硬编码 SoC、无 core 数、无 format 常量、无 ACLNN
- kt-kernel 构建系统 —— `--soc-version` / `Ascend910` 在 CMakeLists / setup.py / install.sh 里
  **一次都没出现**，检测只做 `find_path(acl/acl_rt.h)` + `find_library(ascendcl)`
- MXFP4=39 枚举、GGUF loader、`ggml_vec_dot_mxfp4_q8_0` NEON 内核 —— 纯 CPU
- `moe_runner/ascend.py`、`kt_expert_masks.py` —— 纯编排
- DSv4 attention backend / memory pool —— #33030 已覆盖，rebase 后白拿

⚠️ 一个非 SoC 的前提：`moe.hpp:92` 的守卫是 `#if defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)`
—— **SVE 开了就走不到 MXFP4 快路径**。若 950 机器换了主机 CPU 或默认开 SVE，这条要重新拉平。

### 小改（0.5–2 人天/项）
- `hardware_backend/npu/utils.py: set_default_server_args` 加 >64GB 分支（现在只分 ≤32G / ≤64G 两档，950 都不匹配）
- `kt_ep_wrapper.py` 的私有 torch_npu API（`npu._subscribe_report` / `_launch_host_func`）+ ACL 107011 容忍 —— 重验，不重写
- 启动脚本：删掉 `KT_MXFP4_DEPOOL` / `KT_MXFP4_GGUF_DEDUP`

### 大改（3–8 人天/项）
- **sglang rebase**：Boyi 分支相对其上游基点 `eea2e5d6e` 的自有改动是 **16 commits / 15 文件 / +2857−58**，
  且**一行都没碰 `hardware_backend/npu/`**。#33030 改的 14 个文件里与之重叠的只有 `models/deepseek_v4.py`。
  确定的冲突点：`MQALayer.__init__` 的 `wo_a` 量化决策段（Boyi `@@ -695,14`，#33030 `@@ -689,14`），
  同一段 14 行，机械但真实。**机械 rebase 3–5 + A3 回归 2–3 = 5–8 人天**（迁不迁 950 都得付）
- `KTEPWrapperMethod` 换绑 `NPUW4A4Fp4MoEMethod`：dispatcher 输出 dtype 从 `"int8"` 变 `"bf16"`，
  `_ascend_pre_dispatch` / `_ascend_post_combine` 的假设要跟着改
- 权重路径重构（见 §2）
- torch_npu 2.9.1 → ≥2.10 环境迁移 + A3 回归

### 需要新写、或直接砍掉
- **`kt_stream_prefill.py`（1461 行）** —— 彻头彻尾 int8/W8A8/FRACTAL_NZ 写死：
  `torch.empty(..., dtype=torch.int8, pin_memory=True)`、O_DIRECT reader 的 1 字节/权重偏移算术、
  `_ACL_FORMAT_FRACTAL_NZ = 29` 本地字面量（绕过 `npu_format_cast` 的对齐守卫）、
  `_inplace_nz()` 的字节数守恒假设。两条路：
  - **(i) 950 上直接关掉 `KT_PREFILL_STREAM`**（默认本来就关）。代价：丢掉 A3 上实测 +9~15% 的
    动态热专家 decode 收益。**0 人天。**
  - **(ii) 重写成 MXFP4-native 流式路径**：不再需要 depool，但 `_inplace_nz` 要换成
    `npu_format_cast(..., customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2).transpose(1,2)`
    + scale 重排。**10–15 人天**，而且是我们撞过丢写坑的同一块代码。

  建议先走 (i) 打通 —— 950 HBM 更大、常驻专家数可能大幅提高，流式换入的边际收益需要重测。

### 可以删掉（负工作量）
- `tools/ascendc_mxfp4/`（本来要移植 `--cce-aicore-arch=dav-c220`，约 5–8 人天）
- `_apply_resident_layer_depool` / `_mxfp4_convert_fn` / `KT_MXFP4_DEPOOL` / `KT_MXFP4_GGUF_DEDUP`
- `DeepSeek-V4-Flash-W8A8/` checkpoint 及其 L1/L2/L3 校验流程

---

## 4. 唯一的硬阻塞：A5 版 `torch.ops.custom.*` 算子包

DSv4-Flash 在 NPU 上依赖 10 个 `torch.ops.custom.*` 算子：

```
custom.compressor                            custom.npu_mla_prolog_v3
custom.npu_sparse_attn_sharedkv              custom.inplace_partial_rotary_mul
custom.npu_sparse_attn_sharedkv_metadata     custom.npu_moe_gating_top_k
custom.npu_quant_lightning_indexer           custom.npu_hc_pre
custom.npu_quant_lightning_indexer_metadata  custom.npu_hc_post
```

**它们不在 sglang 里，也不在 torch_npu 里** —— 来自装进 `${ASCEND_TOOLKIT_HOME}/opp/vendors/`
的第三方包（cann-recipes-infer / ops-transformer / sgl-kernel-npu）。我们的 bootstrap 脚本按 910 编：

```
tools/setup_dsv4_env_from_clean_cann.sh:58   : "${SOC:=ascend910_93}"
tools/setup_dsv4_env_from_clean_cann.sh:183  build.sh -c "$SOC"
```

而 #33030 表明 **A5 需要的是不同的算子**（不是同一算子换 SoC 重编）：
`custom.npu_kv_quant_sparse_attn_sharedkv{,_metadata}`、`torch.ops.npu.kv_compress_epilog`、
`custom.indexer_compress_epilog` 都是**新算子名**，`custom.npu_hc_post` 的 A5 签名也不同。

**#33030 只调用它们、不提供它们。** 而 `hardware_backend/npu/utils.py:98` 的 `import custom_ops`
包在 try/except 里只打 warning —— **算子缺失不会在 import 时报错，而是在模型加载时才炸。**

→ **这一项不确认，上面所有估算都是空的。**

---

## 5. 环境约束

| | 现状 | 950 需要 |
|---|---|---|
| CANN | 9.0.0 | **未知**。仓库/文档里**没有任何 950/A5 的镜像 tag**（只有 `cann9.0.0-910b`、`cann9.0.0-a3-*`）。推断需 ≥9.1 或专门的 A5 版本 |
| torch | 2.9.1 | ≥2.10 |
| torch_npu | 2.9.1 | **≥2.10** —— Boyi 分支 `linear_method_npu.py:40` 注明 A5 FP4 链路 "Verified on A5 / torch_npu 2.10.0.post2.dev20260704" |
| 必需 dtype | — | `torch_npu.float8_e8m0fnu`（枚举 293）、`torch_npu.float4_e2m1fn_x2`（296，**必须**用 torch_npu 的枚举） |
| 必需 torch_npu 算子 | — | `npu_dynamic_mx_quant`、`npu_dynamic_block_quant`、`npu_transpose_quant_batchmatmul`、`npu_quant_matmul(scale_dtype=, group_sizes=)`、`npu_grouped_matmul(x_dtype=, weight_dtype=, per_token_scale_dtype=)` |

`pyproject_npu.toml` 里**没有**任何版本 pin，版本纪律全靠容器镜像。

---

## 6. 精度影响

- **消失的不一致**：现在 NPU 常驻是 int8、CPU 卸载是 MXFP4；950 上两边统一成 MXFP4，
  **权重格式的不一致消失**（块几何相同，均为 block-32 + E8M0）。
- **新引入的不一致**：**激活侧口径不同**。950 NPU 走 W4**A8**，激活被
  `npu_dynamic_mx_quant(block_size=32, dst_type=float8_e4m3fn)` 量到 MX-FP8；
  kt-kernel CPU 侧走 `ggml_vec_dot_mxfp4_q8_0`，激活是 **Q8_0（int8，block-32 对称）**。
  FP8-e4m3 的动态范围远好于 int8，**推断**这个新不一致的量级小于现在的权重不一致，
  但必须在 950 上实测 GPQA 才能确认。
- **clamp**：A5 的 `_apply_swiglu_limit_npu` **会** clamp。而我们在 A3 上实测
  （四臂十二轮，见 `CLAMP-SITES-MATRIX.md`）：NPU routed 专家 clamp 会掉 **3.65pp**（p<0.05）。
  迁到 950 后 NPU 侧默认就 clamp —— **这条要在 950 上重新评估**，A3 的结论不一定平移
  （A3 上 NPU 常驻的是 int8 热专家，950 上是 MXFP4，量化误差分布不同）。

---

## 7. 明确的未知项

1. **A5 版 `torch.ops.custom.*` 算子包是否存在、从哪里拿、要什么 CANN 版本。** 最高优先级，唯一硬阻塞。
2. 950 要求的 CANN 版本。
3. FRACTAL_NZ 在 950 上的 tiling（`FRACTAL_NZ_C0_16`），决定 `_inplace_nz` 的字节数守恒假设是否成立。
4. **950 单 die 的 HBM 容量。** 决定能常驻多少专家，进而决定「最小路径 vs 完整路径」。
5. 950 主机 CPU 是什么，决定 `moe.hpp:92` 的 NEON/SVE 守卫是否还成立。
6. W4A8（NPU）vs W4×Q8_0（CPU）激活口径差异的实际精度影响。
7. `npu_dynamic_mx_quant` / `npu_grouped_matmul(weight_dtype=fp4)` 在 950 上的实际吞吐 ——
   决定「MoE 是否还值得 offload 到 CPU」这个前提本身。
8. **#33030 会不会合入、以什么形态合入。** 现在 open + CHANGES_REQUESTED + CI 红。

---

## 8. 建议的下一步（都很便宜，能大幅收窄不确定性）

1. **问上游/华为**：A5 的 `custom.npu_kv_quant_sparse_attn_sharedkv` / `kv_compress_epilog` /
   `indexer_compress_epilog` / batched `npu_hc_post` 从哪个包出、什么 CANN 版本带。
   **这一条的答案决定要不要继续。**
2. 在 PR #33030 下面问 950 单 die HBM 容量和实测 MXFP4 GMM 吞吐。
3. **不依赖 950**：把 `deepseek_v2.py MoEGate` 的 fp32 router GEMM cherry-pick 到 A3 上跑一轮 GPQA。
   该改动**没有 A5 gate**（条件只是 `_is_npu and self.is_deepseek_v4`），我们现在**缺**它：

   ```python
   elif _is_npu and self.is_deepseek_v4:
       # DSV4's non-hash layers route on near-degenerate logits, so the
       # router GEMM must run in fp32 -- the CUDA branch below uses
       # linear_bf16_fp32 for the same reason. In bf16 the top-k
       # boundary flips on layers whose selected experts are within ~1%.
       if self.weight_fp32 is None:
           self.weight_fp32 = self.weight.data.float()
       logits = F.linear(hidden_states.float(), self.weight_fp32, None)
   ```

   已核实原文（`api.github.com/repos/sgl-project/sglang/pulls/33030/files`，
   `models/deepseek_v2.py` +12/−0）。可能是与 clamp 正交的另一个 pp 级变量。

---

## 附：置信度与最大不确定性来源

**置信度：中低。** 按影响排序：
1. A5 custom 算子包的可得性 —— 不可得则全部作废
2. **我们没有 950 硬件** —— 所有关于 NZ tiling、UB 预算、算子签名的判断都是从 diff 读出来的
3. #33030 尚未合入，可能被要求大改
4. 950 HBM 容量未知，直接决定走哪条路线
5. 上游主干从 Boyi 基点（`eea2e5d6e`，08-13）到 #33030 base（`0c7ff19e3`）之间的漂移未量化，
   rebase 成本可能被低估
