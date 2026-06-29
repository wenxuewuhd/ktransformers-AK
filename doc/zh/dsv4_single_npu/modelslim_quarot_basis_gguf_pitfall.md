# 坑⑰：modelslim 权重单卡乱码 —— CPU GGUF 与 NPU 侧"量化基底"必须一致（quarot 旋转）

> 注：本文"加 `KT_FORCE_SYNC_SUBMIT=1` 稳 greedy"（§下方第2点）已过时——异步竞态已根治（commit `e5f53ad`），`force-sync=0` 即对且确定，无需再加。**基底不一致这个主结论仍有效**（换 NPU checkpoint，CPU GGUF 必须同基底）。

**日期**：2026-06-22　**现象**：用 `/workspace/models/DeepSeek-V4-Flash-w8a8-mtp`（msModelSlim **W8A8_DYNAMIC**，记 **B**）单卡起服务，`15×17` 输出多语言 token 乱炖（前向数值坏，非崩溃）。

## 一句话

**不是模型不同、不是 int8/mxfp4 精度问题，是"坐标系（量化基底）"不一致。** B 的 modelslim recipe 在量化前做了 **quarot 全局正交旋转**，把整条 residual stream 换了基底；而 CPU MoE 用的老 GGUF（`/workspace/models/cache/dsv4_layer*_mxfp4.gguf`）是从**没旋转**的 checkpoint 转的。NPU 侧吐出"旋转基底"的激活，喂给"非旋转基底"的 CPU 专家 → 224/256 专家算错 → 乱码。

## 为什么（原理）

- 机器上有三个 checkpoint，**B 和另两个是同一个底模**（都从 native-MXFP4 量化）：
  - `DeepSeekV4/DeepSeek-V4-Flash`（native MXFP4）→ 老 mxfp4 GGUF 的来源
  - `DeepSeekV4/DeepSeek-V4-Flash-W8A8`（compressed-tensors，**A**，朴素 int8）
  - `DeepSeek-V4-Flash-w8a8-mtp`（modelslim，**B**）
- 逐张量比 A vs B 的 expert/attn 权重 raw-int8 **cosine ≈ 0.01（正交）**——一度误判"不同模型"，**错**。真因：B 的 `best_practice.yaml` 在 `linear_quant` 之前先 `quarot`（`optional/quarot.safetensors` 里的 `global_rotation[4096,4096]`，已**离线 fuse 进权重**）+ `flex_smooth_quant`。A 没旋转。**同模型、不同基底 → cosine≈0。**
- quarot 是全局正交旋转、已 fuse、对 KT **透明**：KT 只做矩阵乘，只要**专家权重和喂进来的激活在同一基底**，旋转在数学上自动抵消。fork 全 NPU（B 自己的专家全在 NPU）能跑通正是此理（runtime 从不加载 `global_rotation`）。
- 单卡坏在：NPU 侧 = B（旋转基底）→ 激活是旋转的；CPU 侧 = 老 GGUF（非旋转基底）→ 读错。merge 在相加前就已经错了，救不回来。

## 排除链（都已证伪为非因）

- config 5 处（compress_ratios 首尾=1 等，和 fork `--json-model-override-args` 一致）✅
- dsv4 chat 编码自动生效（arch=DeepseekV4 触发 `encoding_dsv4`，prompt_tokens 9→13）✅ —— raw `/generate` 不组 prompt，必须 `/v1/chat`
- quant/attention/indexer 代码：和 fork `/workspace/code/dsv4-acc-compare/sglang-fork @298193eb3` **逐文件 diff 完全一致**（只差 2 行 KT 补丁）
- modelslim 注意力线性**离线数值对账 cos 0.99997**（wq_a/wkv/wq_b 全对）
- **Ctrl 对照**：checkpoint A（compressed-tensors）走**完全相同单卡 launcher** → `15×17=255` 完全对 → 单卡管线/KT/tp1/编码全没问题

## 修法（已验证跑通 + 稳定 255）

1. 给 **B 自己**的 experts 转 GGUF：`tools/convert_w8a8_to_gguf_q8_0.py` 能读 B 的 modelslim 格式（`.weight`int8 + `.weight_scale`，offset=0 对称→忽略正确，dequant=int8×scale）。
   - ⚠️ B 的 index 名是 `quant_model_weights.safetensors.index.json`，需软链成 `model.safetensors.index.json`。
   - ⚠️ mxfp4 converter（`convert_mxfp4_layer_to_gguf.py`）**读不了** B —— 它要 native-MXFP4 的 `.scale`+nibble。所以走 **Q8_0**。
   - 输出到**新目录** `/workspace/models/cache_b/`（勿覆盖 A 的 cache，别的 session 在用）。
   ```bash
   ln -s quant_model_weights.safetensors.index.json \
     /workspace/models/DeepSeek-V4-Flash-w8a8-mtp/model.safetensors.index.json
   python3 tools/batch_convert_w8a8_layers_mp.py \
     --input /workspace/models/DeepSeek-V4-Flash-w8a8-mtp \
     --output-dir /workspace/models/cache_b \
     --layer-start 0 --layer-end 42 --quant q8_0 --jobs 4 --verify-sample 2
   ```
2. 启动（NPU modelslim + CPU 走 B 的 Q8_0）：
   ```bash
   export MODEL_PATH=/workspace/models/DeepSeek-V4-Flash-w8a8-mtp QUANTIZATION=modelslim
   export KT_NUM_GPU_EXPERTS=32
   export KT_GGUF_TEMPLATE='/workspace/models/cache_b/dsv4_layer{layer_idx}.gguf'
   export KT_FORCE_SYNC_SUBMIT=1   # 见下；不要开 KT_MXFP4_DEPOOL（会选回错基底的老 mxfp4）
   ```
3. **还有第二个坑**：换上 B 自己的 GGUF 后乱码消失、变连贯数学，但 greedy 在**请求间不稳定**（对/跑题/除法大幅跳变）。根因 = KT MoE stream-callback 非确定（见 `offline-moe-check`）。加 **`KT_FORCE_SYNC_SUBMIT=1`** 后 **3/3 字节一致 + 稳定 =255**。

## 教训

- 单卡 KT 是 **NPU 权重 + CPU GGUF 混算**；二者**必须同一次量化（同一基底）**出来的。换 NPU 侧 checkpoint，CPU GGUF 必须同步换。
- 判"是不是同一个模型"别只看 raw 权重 cosine —— 带 quarot/smooth 的量化会把权重旋到正交基底，cosine≈0 但其实同模型。
- mxfp4 与 modelslim 不能直接换：格式不同（mxfp4 nibble+`.scale` vs int8+`.weight_scale`+`.weight_offset`），且基底不同。
