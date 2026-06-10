# DeepSeek-V4-Flash 原生 MXFP4 → GGUF 转换与校验指南

> 面向开源用户/新机器部署。从官方 checkpoint 出发，完成 **转换 → 三级校验 → （可选）kernel 数值对账**。
> 全程**无损 bit-repack**（不是再量化）：GGUF 反量化与官方 checkpoint 反量化**逐元素 bit-exact**。
> 内部研发记录见同目录 `mxfp4_cpu_moe_handoff.md`。

## 0. 这是什么

DeepSeek-V4-Flash 官方发布了原生 MXFP4 专家权重（E2M1 nibble + ue8m0 per-32 scale，
`expert_dtype: "fp4"`）。本仓库让 kt-kernel 的 CPU offload MoE 直接消费它（0.53125 B/元素，
对比 Q8_0 的 1.0625 搬运字节减半），需要把 safetensors 按层 repack 成 GGUF。

关键格式事实（实现已对齐，列出供审计）：

| 项 | 官方 checkpoint | GGUF (`GGML_TYPE_MXFP4=39`，与上游 llama.cpp 对齐) |
|---|---|---|
| 块结构 | weight `I8 [N, K/2]` + scale `F8_E8M0 [N, K/32]` 两张量 | `block_mxfp4{uint8 e; uint8 qs[16]}`，17B/32 元素 |
| nibble 序 | **consecutive**：byte i = K 位置 2i(lo), 2i+1(hi)（`inference/convert.py` 实锤） | **half-block**：qs[j] = K 位置 j(lo), j+16(hi) |
| 数值 LUT | `FP4_TABLE × 2^(e-127)` | `kvalues_mxfp4 × 2^(e-128)`（×2 折进 scale，**bit 级等价**） |

转换器在每个 32 元素组内做 nibble 重排（不是 byte copy！），e8m0 scale 字节原样直存。

## 1. 前置

- **代码**：本仓库 `mxfp4-cpu-moe` 分支。vendored `third_party/llama.cpp`（b3173，commit `a94e6ff`）打 patch：

  ```bash
  cd third_party/llama.cpp   # 干净的 b3173
  git apply -p1 ../../tools/kt_dsv4_npu_patches/llama_cpp/0001-fix-gguf-NumPy-2-GGUFReader.patch
  git apply -p1 ../../tools/kt_dsv4_npu_patches/llama_cpp/0002-add-ggml-type-mxfp4.patch
  ```

  两个 patch 改动文件**不相交**（0001 仅 gguf_reader.py；0002 是 ggml C 侧 + constants.py），顺序无关、均对
  pristine b3173 clean apply。**0002 永远必须**（MXFP4 类型+NEON kernel 硬依赖）；**0001 在 NumPy≥2 环境必须**
  （本指南的 L3 校验/转换工具用 GGUFReader 读 GGUF 会触发；serving 本身不依赖它——kt-kernel 自带 GGUF 解析）。
- **权重**：`huggingface.co/deepseek-ai/DeepSeek-V4-Flash`（46 shard，**逐 shard 验完整**：
  文件 >135B 且 `8 + header_len + max(data_offsets) == 文件大小`；git-lfs 渐进下载常见半截文件）。
  注意 shard 可能**整个缺失**（不是指针文件），用 `model.safetensors.index.json` 对名单。
- **Python**：3.10+，`numpy`、`torch`（CPU 即可）、`safetensors`。gguf-py 用 vendored 的
  （转换脚本自动 `sys.path` 注入，**不要**用 pip 的 gguf——枚举 39 是本仓库扩展）。
- **磁盘**：产物 43 层 × 3.42GB ≈ **138GiB**；峰值内存 ~6GB/进程。

## 2. 转换

```bash
# 全量 43 层，多进程（每层独占一个 shard，层间并行安全）
python3 tools/batch_convert_mxfp4_layers_mp.py \
    --input  /path/to/DeepSeek-V4-Flash \
    --output-dir /path/to/cache \
    --layer-start 0 --layer-end 42 --jobs 4 --skip-existing

# 单层（调试/补漏）
python3 tools/convert_mxfp4_layer_to_gguf.py \
    --input /path/to/DeepSeek-V4-Flash --layer-idx 16 \
    --output /path/to/cache/dsv4_layer16_mxfp4.gguf
```

⚠️ **并发安全**：不要让两个进程写同一层输出文件（实测会留下截断文件且不报错）；
`--skip-existing` 只跳过 >1GiB 的文件，半截文件会被重转。转完**必须**跑第 3 节校验。

**可复现性承诺**：转换是字节级确定性的——同一 checkpoint、本仓库代码，重转结果与发布产物
**byte-identical**（`cmp` 验证过）。因此 sha256 指纹可作为跨机器的强校验。

## 3. 校验（三级，逐级加强）

```bash
# L1（秒级）：43 层齐全 + 每层精确 3,422,552,640 字节
# L2（分钟级）：sha256 对照发布清单 tools/mxfp4_gguf_sha256.txt
python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache \
    --sha256-manifest tools/mxfp4_gguf_sha256.txt

# L3（最强，需原生 checkpoint 在场）：抽 3 层做 GGUF 反量化 vs 官方反量化逐元素 bit-exact
python3 tools/verify_mxfp4_gguf_set.py --dir /path/to/cache \
    --sha256-manifest tools/mxfp4_gguf_sha256.txt \
    --deep 3 --model-dir /path/to/DeepSeek-V4-Flash
```

判定：三级全 PASS → 权重集可部署。任何 FAIL 按输出提示重转对应层。
（如果你的 checkpoint 版本与我们发布清单时的不同，L2 失配但 L3 通过 ⇒ 以 L3 为准，
并请重新生成你自己的清单：`cd cache && sha256sum dsv4_layer*_mxfp4.gguf | sort -V -k2 > manifest.txt`。）

## 4.（可选）kernel 数值对账 — 新机器/改 kernel 后必跑

重编 `.so` 后（`apt-get install -y libhwloc-dev`；
`cd kt-kernel && CPUINFER_USE_ASCEND_NPU=1 python3 setup.py build_ext --inplace`）：

```bash
# KTMoEWrapper(MXFP4 GGUF) vs torch 参考（同一母权重），唯一损失=激活 Q8 量化
# 期望 cosine >= 0.999（实测 0.999939）；脚本已内置 KT_FORCE_SYNC_SUBMIT=1
python3 tools/p27_cpu_moe_reference_check_mxfp4.py \
    --model-dir /path/to/DeepSeek-V4-Flash \
    --gguf /path/to/cache/dsv4_layer16_mxfp4.gguf \
    --layer-idx 16 --batch 4 --device npu
```

注意：kernel 的 FMA 累加序意味着与其他实现比对要用 **cosine/连贯性**判定，
不要预期 token 级逐字相同（贪心解码对 1e-7 数值差敏感，属预期）。

## 5. 使用

```bash
NPU_DEVICE_ID=<空卡> PORT=<端口> \
KT_GGUF_TEMPLATE='/path/to/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
KT_CPUINFER=128 \
MODEL_PATH=/path/to/DeepSeek-V4-Flash-W8A8 \
bash tools/p27_launch_ds4flash_npu.sh
```

实测收益（K920 192 核 / 8 NUMA / DDR4-3200 3-of-4 通道，单卡 910B3）：
cpu_moe_wall 55→~27ms，decode 8.5→~13 tok/s，CPU 权重常驻 275→137GB。
