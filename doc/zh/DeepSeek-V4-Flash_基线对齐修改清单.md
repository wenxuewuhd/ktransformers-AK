# DeepSeek-V4-Flash 单卡 NPU + KT 基线对齐修改清单

> 状态：草案 v1（计划用，未实施）
> 日期：2026-05-13
> 范围：让本仓库 `third_party/sglang` 单卡 NPU + KT(LLAMAFILE) 路径稳定起服务，并为 MoE CPU offload 后续工作打好底盘。本文档不直接动代码，仅作改动落地的指南。

---

## 0. 背景与基线选择

本仓库内有两份 sglang，路径与来源完全不同：

| | 基线 `/sgl-workspace/sglang` | 当前 fork `third_party/sglang` |
|---|---|---|
| 远端 | `https://github.com/iforgetmyname/sglang`（sgl-project 的 NPU dsv4_release 分支） | `https://github.com/kvcache-ai/sglang`（ktransformers 团队 fork） |
| 当前分支 / HEAD | `dsv4_release` / `298193eb3` (`fix load weight; add vars`) | 工作目录在 `6db949a56` (`feat(kt): per-layer kt_weight_path template ...`) 之上有未提交改动 |
| 8 卡脚本 | `tools/test_code/sglang_script/launch_ds4flash_sglang.sh`（已验证跑通） | `tools/p27_launch_ds4flash_npu.sh`（单卡，KT，目前卡在 load_weights） |

> 决定：**以基线为参考，做定向同步**；不要整体替换，避免丢失 P2.x 已沉淀的 KT/NPU 改动。

---

## 1. 当前能直接定位到的「启动卡点」与基线对照

| 报错 | fork 现状 | 基线做法 | 结论 |
|---|---|---|---|
| `DeepSeekV4Config.__init__() got an unexpected keyword argument 'n_activated_experts'` 等 | `configs/deepseek_v4.py` 自定义 dataclass，字段集合不齐 | **不存在 `configs/deepseek_v4.py`**，统一通过 `_load_deepseek_v4_model` 把 `model_type` 改写成 `deepseek_v3`，让 transformers 自带 `DeepseekV3Config` 解析 | fork 思路偏离主线，应同步基线 |
| `KeyError: 'model.layers.32.mlp.shared_experts.gate_up_proj.weight_scale'` | `remap_weight_name_to_dpsk_hf_format` 把 mlp 路径强行 `.scale → .weight_scale_inv` | 内联式 `name = name.replace(".scale", ".weight_scale")`，**保持 `weight_scale`，不强转 `_inv`** | 同步基线 |
| `tilelang` import 失败 → `DeepseekV4ForCausalLM` 未注册 | 顶层引用 `layers/deepseek_v4_rope.py` 里 tilelang | **没有 `deepseek_v4_rope.py`**，rope 走 `layers/rotary_embedding/factory.py` + `IS_DEEPSEEK_V4=1` | 同步基线 |
| `--quantization fp8 / w8a8_int8` 与磁盘 `compressed-tensors / int-quantized` 矛盾 | 默认 `QUANTIZATION=fp8`，外加 `SGLANG_APPLY_CONFIG_BACKUP=auto` 注入 backup | `--quantization compressed-tensors --disable-shared-experts-fusion`，**不用 backup** | 同步基线 |

---

## 2. 两份 sglang 关键差异（结构性）

| 维度 | 基线 | fork |
|---|---|---|
| `configs/deepseek_v4.py` | **不存在** | 自定义 dataclass `DeepSeekV4Config` |
| `models/deepseek_v4.py` | 3803 行，`load_weights` 内联约 200 行 | 2143 行，抽出 `remap_weight_name_to_dpsk_hf_format`，多 `SGLANG_DSV4_MODE / 2604 / FP4_EXPERTS / _FP8_WO_A_GEMM` 分支 |
| `layers/deepseek_v4_rope.py` | **不存在** | 存在，含 `tilelang` 引用问题 |
| `layers/moe/kt_ep_wrapper.py` | **393 行薄壳**，封装现有 GPU MoE 方法 + `kt_kernel.KTMoEWrapper` | **2940 行**，自带 cpu_buffers / shm / 多 dtype / NPU stream 适配 |
| Config backup 机制 | **无**（无 `SGLANG_APPLY_CONFIG_BACKUP`） | 有 `_load_deepseek_temp_model` + `config_backup_{small,large}.json` |
| `IS_DEEPSEEK_V4=1` 钩子 | `mem_cache/common.py` / `mem_cache/memory_pool.py` / `disaggregation/decode.py` / `layers/rotary_embedding/factory.py` 多处 dsv4 专用分支 | **完全没有引用** |
| `SGLANG_DSV4_MODE / 2604 / FP4_EXPERTS` | 无 | 多处分支，默认值 `SGLANG_DSV4_MODE="2604"` / `SGLANG_DSV4_2604_SUBMODE="2604B"` / `SGLANG_DSV4_FP4_EXPERTS=True` |
| `--kt-*` 参数 | 已存在于 `server_args.py` | 已存在 |
| 启动量化 | `--quantization compressed-tensors` | 之前默认 `fp8`，调整中 |
| 融合 kernel env | `USE_FUSED_HC_PRE/POST_ASCENDC=1`、`USE_PA_DECODE=1`、`USE_PA_PREFILL=1`、`USE_FUSED_COMPRESSOR=1`、`LI_KV_DTYPE_INT8=1`、`USE_NPU_MOE_GATING_TOP_K=1`、`USE_FUSED_TRANSPOSE_BATCHMATMUL=1`、`USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=1` 等一大批 | 未在 launch 脚本中 export |

---

## 3. 优先级定义

- **A 必改**：当前 fork 不改起不来，或改了能让单卡 W8A8 直接跑通。
- **B 建议同步基线**：基线写法更干净、与官方 W8A8 checkpoint 对齐，做 KT MoE CPU offload 也更顺。
- **C 保留 fork**：P2.x 已沉淀且与"启 W8A8 + KT"直接相关的改动，不要回退。
- **D 暂不动**：fork 内实验路径，触动成本高，先做隔离 / 默认禁用。

---

## 4. 修改清单详项

### 4.1 Config 解析

#### 4.1.1 `third_party/sglang/python/sglang/srt/configs/deepseek_v4.py`（A）
- **现状**：自定义 dataclass `DeepSeekV4Config`，屡踩 `n_activated_experts` / `head_dim` / `sliding_window_size` / `model_type` 等字段。
- **基线做法**：根本没有这个文件。
- **建议改法**：
  - 方案 A1（推荐）：**整文件删除或缩为空壳**，把 `_CONFIG_REGISTRY` 中 `DeepSeekV4Config` 摘掉（见 4.1.2）。
  - 方案 A2（保留兼容）：保留为普通 `PretrainedConfig` 子类 + `__init__(self, **kwargs)`，只做别名规范化（`n_activated_experts→num_experts_per_tok`、`head_dim→v_head_dim`、`sliding_window_size→window_size`），其余靠 `super().__init__(**kwargs)` 兜底。
- **推荐**：A1。

#### 4.1.2 `third_party/sglang/python/sglang/srt/utils/hf_transformers_utils.py`（A）
- **现状**：
  - 顶部 `from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config` 与 `_CONFIG_REGISTRY` 注册
  - `_load_deepseek_temp_model(...)` 走 packaged backup config
  - `get_config()` 内 `elif envs.SGLANG_APPLY_CONFIG_BACKUP.get() != "none":` 强行走 backup
- **基线做法**：`get_config()` 主路径直接 `AutoConfig.from_pretrained(...)`，捕获 `ValueError("deepseek_v4")` 后跑 `_load_deepseek_v4_model`，里面把 `config.json` 拷贝一份并 `model_type="deepseek_v3"` 再交给 `AutoConfig`。
- **建议改法**：
  1. 删除 fork 中 `DeepSeekV4Config` import 与 `_CONFIG_REGISTRY` 注册。
  2. 引入 `_load_deepseek_v4_model(model_path, ...)`：拷贝 `config.json` 到 tmp，置 `architectures=["DeepseekV4ForCausalLM"]` + `model_type="deepseek_v3"`，再 `AutoConfig.from_pretrained`。
  3. `get_config()` 调整：先 `AutoConfig.from_pretrained(...)`；`except ValueError as e:` 若 `"deepseek_v4" in str(e)` 则回退 `_load_deepseek_v4_model`；删除/不再使用 backup 分支。
  4. 保留 `_load_deepseek_v32_model`、mistral_large_3 等不变。
- **依赖**：`models/deepseek_v4.py` 必须把 `config.xxx` 读取改成 `getattr(config, "xxx", default)`，参 4.2。

#### 4.1.3 `SGLANG_APPLY_CONFIG_BACKUP` 整套（A→B）
- **涉及**：`environ.py:168`、`utils/hf_transformers_utils.py` backup 分支、`configs/config_backup_{small,large}.json`、`configs/model_config.py` 中相关读取。
- **建议改法**：
  - `environ.py` 中默认值由 `"auto"` 改为 `"none"`。
  - `hf_transformers_utils.get_config` 中 backup 分支注释/删除。
  - `configs/model_config.py::_maybe_auto_set_dsv4_fp4_experts` 中"按磁盘 `quantization_config` 自动关 `SGLANG_DSV4_FP4_EXPERTS`"逻辑保留（对 W8A8 仍有用），仅清理对 backup 的依赖。
  - `config_backup_*.json` 不立即删除，先归档到 `archive/`，便于回滚。

---

### 4.2 模型文件 `models/deepseek_v4.py`

#### 4.2.1 入口与配置类型（A）
- 与 4.1 同步：把类型注解 `config: DeepSeekV4Config` 改为 `config: PretrainedConfig`（或 transformers 内置 DSv3Config）。
- 全文件扫一遍 `config.xxx`：对 `index_head_dim / index_n_heads / index_topk / o_lora_rank / o_groups / compress_ratios / hc_mult / hc_sinkhorn_iters / hc_eps / n_hash_layers / num_nextn_predict_layers / swiglu_limit / window_size` 等字段，改成 `getattr(config, "...", default)`，避免在 DSv3Config 上断言存在。

#### 4.2.2 `load_weights` & `remap_weight_name_to_dpsk_hf_format`（A）
- **现状**：抽出函数对 mlp 路径 `.scale → .weight_scale_inv`，多 `SGLANG_DSV4_MODE` / `MOE_BIT_WISE_EQUAL_MODE` / `ATTN_BIT_WISE_EQUAL_MODE` / `COMPRESSOR_PART` 分支。
- **基线做法**（内联简化版，要点）：
  - `name = "model." + name`
  - `name.replace("ffn", "mlp")`、`"gate.bias" → "gate.e_score_correction_bias"`
  - `if ".scale" in name: name = name.replace(".scale", ".weight_scale")`（**保留 `weight_scale`**）
  - experts: `w1/w2/w3 → gate_proj/down_proj/up_proj`
  - `.attn. → .self_attn.`、`.attn_norm. → .input_layernorm.`、`.mlp_norm. → .post_attention_layernorm.`
  - `embed.weight → embed_tokens.weight`、`model.head. → lm_head.`
  - 之后用 `stacked_params_mapping` + `expert_params_mapping` + `cached_a_proj/cached_eh_proj` 等分类处理
- **建议改法**：用基线 `load_weights` 整段替换 fork 现有实现；删除 `remap_weight_name_to_dpsk_hf_format`、`MOE_BIT_WISE_EQUAL_MODE / ATTN_BIT_WISE_EQUAL_MODE / COMPRESSOR_PART` 等私有分支。保留 KT MoE 接入位点（即 `self.mlp = deepseek_v2.DeepseekV2MoE(...)` 那一段）。
- **依赖**：`params_dict` 中实际注册的是 `weight_scale`（compressed-tensors int8 W8A8）；KT 接入位点不依赖 `weight_scale_inv` 命名。

#### 4.2.3 `SGLANG_DSV4_MODE` / `_FP8_WO_A_GEMM` / `_dequant_fp8_wo_a`（D）
- 默认环境下走 2601 / 简化路径；**不删代码**，但分支用环境量短路（默认 False/disabled）。
- 后续若不再做 2604 / FP4 实验，再整体清理。

#### 4.2.4 attn_sink / hc_attn / hc_ffn / 融合 kernel hooks（A）
- 不替换实现，但在 launch 脚本中 export 基线那批 `USE_FUSED_*_ASCENDC=1` / `USE_PA_*=1` 等 env（见 4.6）。

#### 4.2.5 RoPE 子模块（A）
- 现状：`from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton`、`precompute_freqs_cis`。
- 改法：
  - 删除 `third_party/sglang/python/sglang/srt/layers/deepseek_v4_rope.py`。
  - `models/deepseek_v4.py` 改用 `layers/rotary_embedding/factory.py` 的 `get_rope(...)`；具体 API 与调用方式以基线 `models/deepseek_v4.py` 内的写法为准（基线该文件 import 行与 MQA 构造均使用 factory）。
  - launch 脚本 export `IS_DEEPSEEK_V4=1`，配合 factory 内 `is_dsv4` 分支。
- 依赖：4.5 节 `IS_DEEPSEEK_V4` 钩子 patch。

---

### 4.3 模型文件 `models/deepseek_v2.py`（B，短期不动）
- fork 3376 行 / 基线 2397 行；差异较多。
- 短期建议：**保留 fork 现版本**；4.2 改完后跑通单卡，再回头逐段比对，挑出"KT/NPU 必要" diff 单独合并。

---

### 4.4 `layers/deepseek_v4_rope.py`（A，删除）
- 仅被 `models/deepseek_v4.py` 两处 import。删除后必须同步改 4.2.5。

---

### 4.5 `IS_DEEPSEEK_V4=1` 钩子代码（B）
基线在以下 4 个文件里有 dsv4 专用分支，fork 同名文件目前**全部没有**这些分支：

| 文件 | 基线增量内容 | 优先级 |
|---|---|---|
| `mem_cache/common.py` | dsv4 sliding-window + c4/c128 sparse kv 的 `alloc_paged_token_slots_extend` 分支（`LastLoc`） | A |
| `mem_cache/memory_pool.py` | `req_to_token_swa / req_to_token_c4 / req_to_token_c128 / req_to_token_c4_state / req_to_token_c128_state` 池及 `get_all_locs_by_req / get_all_locs_by_kv_lens` 的 dsv4 分支 | A |
| `disaggregation/decode.py` | dsv4 in disaggregation | C（单卡用不到） |
| `layers/rotary_embedding/factory.py` | `is_dsv4 = get_bool_env_var("IS_DEEPSEEK_V4")` 分支 | A（配合 4.2.5） |

- 改法：把基线相应分支 patch 到 fork 同名文件；非 dsv4 路径行为不变（基线写法都包在 `if get_bool_env_var("IS_DEEPSEEK_V4"):` 之内）。
- `environ.py` 可选追加 `IS_DEEPSEEK_V4 = EnvBool(False)`，或继续用 `get_bool_env_var` 风格。

---

### 4.6 启动脚本 `tools/p27_launch_ds4flash_npu.sh`（A）

参照基线 `launch_ds4flash_sglang.sh` 做以下对齐：

| 调整 | 基线 / 推荐值（单卡同义） | fork 现状 | 优先级 |
|---|---|---|---|
| `--quantization` | `compressed-tensors` | 在 `fp8 / w8a8_int8` 之间切 | A |
| `--disable-shared-experts-fusion` | 带 | 未带 | A |
| `SGLANG_APPLY_CONFIG_BACKUP` | 不存在 | 默认 `auto`，脚本里 `export none` | A |
| `IS_DEEPSEEK_V4=1` | 带 | 未带（但 fork 当前未接入相应代码，参 4.5） | A |
| ASCEND 融合 kernel env | `USE_FUSED_COMPRESSOR=1` / `LI_KV_DTYPE_INT8=1` / `USE_PA_DECODE=1` / `USE_PA_PREFILL=1` / `USE_FUSED_HC_POST_ASCENDC=1` / `USE_FUSED_HC_PRE_ASCENDC=1` / `USE_NPU_MOE_GATING_TOP_K=1` / `USE_FUSED_TRANSPOSE_BATCHMATMUL=1` / `USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=1` / `ASCEND_USE_FIA=1` 等 | 未 export | A（影响精度/性能，但不影响启动） |
| `STREAMS_PER_DEVICE=32` / `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` / `TASK_QUEUE_ENABLE=1` / `SGLANG_SET_CPU_AFFINITY=1` | 带 | 未带 | A |
| `--page-size 128` | 显式 | 隐式（NPU 默认即 128） | C |
| `--chunked-prefill-size` | `-1`（关闭 chunked prefill） | `4096` | B |
| `--mem-fraction-static 0.75` | 带 | 未带 | A |
| `--watchdog-timeout 18000` | 带 | 未带 | C |
| `--dtype bfloat16` | 带 | 未带 | A |
| `--trust-remote-code` | 带 | 未带 | A |
| MTP / DeepEP / DP-attention / DP-LM head | 8 卡带；单卡不需要 | 单卡禁用 | C |
| KT 参数：`--kt-method LLAMAFILE --kt-num-gpu-experts 16 --kt-weight-path ... --kt-threadpool-count 8 --kt-cpuinfer 24` | 基线 `server_args.py` 已支持 | 脚本已带 | C |

---

### 4.7 KT MoE 入口 `layers/moe/kt_ep_wrapper.py`（B，最关键）

- **现状**：fork 2940 行；基线 393 行。两边已是不同实现。
- **基线极简版要点**：
  - `KTConfig` dataclass：`layer_idx / num_gpu_experts / cpuinfer_threads / threadpool_count / weight_path / chunked_prefill_size / method / num_layers / max_deferred_experts_per_token`。
  - `KTEPWrapperMethod(FusedMoEMethodBase)`：包一层 GPU 量化方法，CPU 侧由 `kt_kernel.KTMoEWrapper` 黑盒处理。
  - `create_weights` 中只为 GPU 端创建 `num_gpu_experts` 份权重；CPU 端 `KTMoEWrapper.load_weights(physical_to_logical_map_cpu)`。
  - `submit/sync` 用 `torch.cuda.current_stream(x.device).cuda_stream`（torch_npu 兼容）；
  - 没有 cpu_buffers / shm / 多 dtype / 多 stream 队列。
- **建议改法（两步）**：
  1. 用基线 393 行整段替换 fork 当前 `kt_ep_wrapper.py`，作为"干净基底"。
  2. 把 fork 中真正解决 NPU 适配的 patch 抽出，重新打到该基底上：
     - 设备无关 Stream/Event 抽象（NPU 上 `cuda_stream` → `kt_current_stream_handle`）。
     - alt_streams / 多 stream 入口（与 `models/deepseek_v4.py` 内 alt_streams 一致）。
     - `resolve_kt_weight_path_for_layer`：`{}` / `{layer_idx}` per-layer 模板解析。
     - `--kt-max-deferred-experts-per-token` 最后一层置 0 的逻辑（基线已带）。
- **CPU offload 设计提示**：MoE CPU offload 的核心数据流在 `kt_kernel`（外部 wheel）C++ 侧；`kt_ep_wrapper.py` 仅负责"传地址/topk + 同步"。fork 现版 cpu_buffers/shm 那一套大概率是早期 Python 侧自管 CPU mirror 的实验产物，对 NPU 单卡不必要。
- **回滚保险**：合并前在 `third_party/sglang` 创建分支 `dsv4_release-sync` 与备份分支 `kvcache-ai-archive`。

---

### 4.8 `configs/model_config.py`（C/D）
- `_quantization_config_as_dict / _hf_layout_looks_like_int_compressed_tensors / _maybe_auto_set_dsv4_fp4_experts` 等 fork 增量功能保留。
- 清理函数体内对 backup config 的依赖（与 4.1.3 同步）。

---

### 4.9 P2.x 已沉淀且保留的项（C）

| 文件 | 内容 | 处理 |
|---|---|---|
| `tools/p27_launch_ds4flash_npu.sh` | `KT_GGUF_TEMPLATE` 单引号、`NPU_DEVICE_ID` / `ASCEND_RT_VISIBLE_DEVICES` / PYTHONPATH 等 | 保留，按 4.6 调整 quant / env |
| `tools/p27_curl_generate.sh` | curl 冒烟 | 保留 |
| `tools/phase12_llamafile_moe_smoke.py` | 多 GGUF / per-layer 模板 | 保留 |
| `third_party/sglang/.../layers/moe/kt_ep_wrapper.py` 内 NPU stream/event 抽象、`stream_handle` 字段 | 现 2940 行中的部分 | 提炼后 patch 到基线 393 行版（见 4.7） |
| `third_party/sglang/.../layers/moe/kt_ep_wrapper.py` 中 `resolve_kt_weight_path_for_layer` 模板解析 | per-layer 模板 | 保留并 patch 进基线版 |
| `doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md` / `doc/zh/Phase0_Phase1_变更记录与复现手册.md` | handoff / Phase 文档 | 保留，建议在二者中追加"对基线的取舍"小节 |

---

## 5. 风险与开关

- **DSv3 vs DSv4 字段兼容**：基线 `_load_deepseek_v4_model` 把 `model_type` 改成 `deepseek_v3`，`DeepseekV3Config` 不带 `index_head_dim / index_n_heads / index_topk / o_lora_rank / o_groups / compress_ratios / hc_mult / hc_sinkhorn_iters / hc_eps / n_hash_layers / num_nextn_predict_layers / swiglu_limit` 等，但 transformers `PretrainedConfig` 会把任意 kwargs `setattr` 进去——前提是模型代码以 `getattr(config, "xxx", default)` 风格读取。**需要扫一遍 `models/deepseek_v4.py` 所有 `config.xxx` 引用**。
- **`transformers==5.3.0`**：基线脚本未 pin 版本即可跑通；fork 之前的报错（`model_type / n_activated_experts`）本质是 dataclass 与 HF 父类构造方式不兼容，对齐基线"伪装 v3"思路后，5.x 也能直接走。
- **`kt_kernel` 版本**：基线 `kt_ep_wrapper.py` 直接 `from kt_kernel import KTMoEWrapper, generate_gpu_experts_masks`；fork 当前也是。需要确认 wheel 一致。
- **`SGLANG_DSV4_MODE / FP4_EXPERTS / _FP8_WO_A_GEMM` 实验路径**：先视为黑盒，在默认环境下短路；不立即删除代码。
- **分支保险**：建议合入前在 `third_party/sglang` 拉 `dsv4_release-sync` 分支，原 fork 分支留作回退。

---

## 6. 落地节奏

### 第 1 批：启动卡死根因（按本清单的 A）
1. 删 `configs/deepseek_v4.py`（或缩成空壳），同步 `_CONFIG_REGISTRY`。
2. `utils/hf_transformers_utils.py`：加 `_load_deepseek_v4_model`，捕获 `ValueError("deepseek_v4")` 回退；摘掉 `DeepSeekV4Config` 注册；砍/降级 `SGLANG_APPLY_CONFIG_BACKUP` 主分支。
3. `models/deepseek_v4.py`：把 `load_weights` & `remap_*` 替成基线的"内联"版；保留 attn_sink / hc_attn / KT MoE 接入点；把 `config.xxx` 改 `getattr`。
4. 删 `layers/deepseek_v4_rope.py`，模型改走 `rotary_embedding/factory.py`。
5. `tools/p27_launch_ds4flash_npu.sh`：quant 改 `compressed-tensors`；export `IS_DEEPSEEK_V4=1` + 基线那批 ASCEND 融合 kernel env；加 `--disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code --mem-fraction-static 0.75`。

### 第 2 批：KV / RoPE 行为对齐（B+A）
1. 把基线 `mem_cache/common.py`、`mem_cache/memory_pool.py`、`layers/rotary_embedding/factory.py` 的 `IS_DEEPSEEK_V4` 分支 patch 到 fork 同名文件。
2. 视情况加 `IS_DEEPSEEK_V4 = EnvBool(False)`（可选）。

### 第 3 批：KT MoE CPU offload 重构（B 核心）
1. 用基线 `kt_ep_wrapper.py` 393 行替换 fork 版本。
2. 把 fork 里 NPU stream/event 抽象、per-layer 模板解析等 patch 上去。
3. 评估是否仍需要 fork 的 `cpu_buffers/shm`——大概率不需要，能省 1000+ 行。

### 第 4 批：实验路径下线（D）
1. `SGLANG_DSV4_MODE / FP4_EXPERTS / _FP8_WO_A_GEMM` 默认值置否；模型代码中相关分支用 `if False:` 整体短路。
2. `config_backup_*.json` 归档至 `archive/`。

### 第 5 批：文档与 smoke
1. 更新 `Phase0_Phase1_变更记录与复现手册.md` / `DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md`。
2. 跑 `p27_launch_ds4flash_npu.sh` + `p27_curl_generate.sh` 端到端冒烟。

---

## 7. 备查：基线 launch 脚本关键点

```bash
# /workspace/code/test_code/sglang_script/launch_ds4flash_sglang.sh 节选
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# Fused Kernels
export USE_FUSED_COMPRESSOR=1
export LI_KV_DTYPE_INT8=1
export USE_PA_DECODE=1
export USE_PA_PREFILL=1
export USE_FUSED_HC_POST_ASCENDC=1
export USE_FUSED_HC_PRE_ASCENDC=1
export USE_NPU_MOE_GATING_TOP_K=1
export USE_FUSED_TRANSPOSE_BATCHMATMUL=1
export USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=1
export ASCEND_USE_FIA=1
export IS_DEEPSEEK_V4=1

QUANT_MODE=compressed-tensors
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 --tp-size 8 --trust-remote-code \
    --attention-backend ascend --device npu \
    --watchdog-timeout 18000 --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.75 --cuda-graph-bs 1 \
    --disable-radix-cache --chunked-prefill-size -1 \
    --max-prefill-tokens 65535 --context-length 65536 \
    --max-running-requests 8 --dtype bfloat16 \
    --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
    --quantization ${QUANT_MODE} --disable-shared-experts-fusion \
    --skip-server-warmup \
    --moe-a2a-backend deepep --deepep-mode auto \
    --speculative-algorithm NEXTN --speculative-num-steps 2 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
```

> 单卡 P2.7 等价的最小子集（仅作示意，不在本文档执行）：
> - 删 `--tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head`
> - 删 `--moe-a2a-backend deepep --deepep-mode auto`
> - 删 `--speculative-algorithm NEXTN ...`
> - `--max-running-requests` 改为 1
> - 加 `--kt-method LLAMAFILE --kt-num-gpu-experts 16 --kt-weight-path ${KT_GGUF_TEMPLATE} --kt-threadpool-count 8 --kt-cpuinfer 24`

---

## 8. 相关文档

- `doc/zh/DeepSeek-V4-Flash_Ascend_NPU_Single_Card_Handoff.md`
- `doc/zh/Phase0_Phase1_变更记录与复现手册.md`
- `doc/zh/DeepSeek-V4-Flash-K920-Single-NPU-Spec.md`
- 基线脚本：`/workspace/code/test_code/sglang_script/launch_ds4flash_sglang.sh`
- 基线 sglang：`/sgl-workspace/sglang`（`dsv4_release` @ `298193eb3`）
- 当前 fork sglang：`third_party/sglang`（`kvcache-ai/sglang` @ `6db949a56` + 未提交改动）
