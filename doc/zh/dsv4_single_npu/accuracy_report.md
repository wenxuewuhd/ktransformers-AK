# DeepSeek-V4-Flash 单卡精度报告(GPQA-Diamond / AIME）

> 范围:单卡 Ascend 910B3 + Kunpeng-920 CPU-MoE offload。对齐目标 = fork/PR 基线。
> 一句话:**静态 prefix-32 与动态 depool 两套配置的 GPQA off 精度均已对齐 PR 基线;depool 在零精度代价下 decode +18%、省一份 DDR。**

## 1. 评测方法(三套共用)

EvalScope · OpenAI 兼容 `/v1/chat/completions`(DSv4 架构自动走内置 `encoding_dsv4`)· **temperature=1, top_p=1** · max_tokens=32768 · 全量(GPQA-Diamond 198 / AIME 30）· 单次 `repeats=1`。

> ⚠️ **temp=1 是 reasoning 模型的标准采样设置**(greedy/temp=0 在长思维链上反而退化,且本栈即使 greedy 也跨 boot 不可复现)。代价是单次有抽样方差:GPQA 198 题标准误 ≈ **±3.2pp**,AIME 30 题 **≈ ±3.3pp/题**。本报告"对齐"= 落在该噪声带内,**非数值相等**;要消除方差需 `--repeats 3~5` 报 mean±std。

## 2. 精度对比

| 配置 | GPQA off | GPQA on | AIME on | 来源 |
|---|---|---|---|---|
| **① PR 基线**¹ | **73.23%** (145/198) | **86.36%** (171/198) | **96.67%** (29/30) | 开发者自报,见 PR 描述 |
| **② prefix-32 baseline** | **75.25%** (149/198) | — | — | 本环境实测 |
| **③ dynamic-hot + side(depool)** | **72.22%** (143/198) | 未测² | 未测² | 本环境实测 |

- **② vs ① off:75.25% vs 73.23% = +2.02pp** —— 噪声带内,**对齐**。
- **③ vs ① off:72.22% vs 73.23% = −1.01pp** —— 噪声带内,**对齐**。
- ② 与 ③ 相差 3.03pp,**在 ±3.2pp 单次标准误内,属同一精度水平**,不能判定孰优(需 `--repeats` 才能分高下)。

**结论:单卡两套配置的 GPQA off 精度都已对齐 PR 基线,depool 未引入精度回退。**

## 3. 性能收益(decode,空载实测)

| 配置 | decode tok/s | ms/tok | 相对 |
|---|---|---|---|
| ② prefix-32 baseline | ~16 | ~62 | 基准 |
| **③ depool(dynamic-hot + side)** | **18.90**(median 18.81) | 52.9(6 reps 抖 <2.3%) | **+18%** |

> 探针 `tools/p27_sidestream_perf.sh`(TAG=depool PORT=8500,空载):每轮 prefill 探针(max_new=1）+ 全量 256 tok,decode ms/tok =(e2e_full − e2e_prefill)/(completion−1）。**必须空载测**——评测占用时 host-DDR 被抢,decode 会假性掉到 ~10 tok/s。

depool 的额外收益:
- **decode +18%**(side-stream 调度重叠 + 同源 mxfp4 现转);
- **省一份 DDR**:NPU 常驻从同源 mxfp4 **现转**,不再额外读独立 W8A8 ckpt(baseline 是 NPU W8A8 + CPU mxfp4 两套量化各占一份);
- **short-prompt TTFT ~436ms**(现转已折进 prefill,无每请求切换停顿);
- 精度零代价(③ 与 ① off 对齐)。

## 4. 配置差异(报告里必须标,否则误读为同栈对比)

| | PR 基线 ① | 单卡 ② prefix | 单卡 ③ depool |
|---|---|---|---|
| NPU 权重 | modelslim W8A8 | compressed-tensors W8A8 | compressed-tensors W8A8(常驻从同源 mxfp4 现转) |
| CPU MoE | — | mxfp4 GGUF | mxfp4 GGUF |
| 专家放置 | (DP) | 静态 prefix-32(0–31 常驻) | 动态 hot-K 常驻 |
| attention backend | `dsv4` + full DP-attention | `ascend`,TP=1 | `ascend`,TP=1 |
| 关键 env | — | `FORCE_SYNC_SUBMIT=0`,depool/dynamic/side 全关 | `MXFP4_DEPOOL=1 + DYNAMIC_RESIDENT=1 + SIDE_STREAM=1 + PREFILL_STREAM=1`,`FORCE_SYNC_SUBMIT=0` |
| 硬件 | (多卡 DP) | 单卡 910B3 + Kunpeng CPU-MoE | 同 ② |

> 共同点:`KT_NUM_GPU_EXPERTS=32`、`KT_CPUINFER=128`、`CHUNKED_PREFILL_SIZE=8192`、`--max-running-requests 1`、`--context-length 65536`、`--mem-fraction-static 0.85`、`--disable-shared-experts-fusion`、card5/port 8500。
> ⚠️ **所以这是功能对齐(同一基准 / 同一 eval harness / 同一采样),不是 bit-level 同栈复现。**

## 5. 复现指引

- 起服务 + 自检 + 跑 off:见 [`gpqa_accuracy_align_REPRODUCE.md`](./gpqa_accuracy_align_REPRODUCE.md)(depool 需显式 `export KT_MXFP4_DEPOOL=1 KT_DYNAMIC_RESIDENT=1 KT_SIDE_STREAM=1 KT_PREFILL_STREAM=1`)。
- 结果存档:
  - ② prefix baseline:`dsv4-acc-compare/eval_results/fix_off_8500_1642/20260623_164226/`(75.25%)。
  - ③ depool:`dsv4-acc-compare/eval_results/depool_dyn_off_8500/20260624_020832/`(72.22%)。
- **`KT_FORCE_SYNC_SUBMIT=0` 现在又对又快**(prefill 异步竞态已根治,commit `e5f53ad`);旧文档里写 `=1` 的地方已过期。

## 6. 待办

- **③ 的 GPQA on / AIME on 未测**²:thinking 输出超长(3k–11k tok/题)+ `max_running_requests=1` 串行 → 客户端超时;且 `KT_PREFILL_STREAM` 长 decode 时 NPU OOM。修法:**调大客户端超时 + `--max-running-requests 8` + 关 `KT_PREFILL_STREAM`**。对标 PR on 86.36% / AIME 96.67%。
- 若要消除单次采样方差:补 GPQA off `--repeats 3` 报 mean±std。
- ① 的 PR 编号/URL 待补(下方脚注 ¹）。

---

¹ PR 基线数字取自所提供的 PR 描述(modelslim,`--attention-backend dsv4`,full DP-attention,temp=1 / top_p=1)。正式引用请补 PR 链接 / 编号:**TODO**。
² 本会话 thinking-on 评测此前崩在工程问题(超时 / OOM),已 deferred,见 §6。
