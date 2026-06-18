# 专家激活分布分析(prefill 直方图 → decode 常驻池)

> 子目标 2 的数据基础。配套 handoff `longseq_prefill_handoff.md` §3.5.C / C-bis。
> 日期 2026-06-10｜模型 DeepSeek-V4-Flash(43 层全 MoE,256 routed experts,top-6)。

本文把**数据 schema、生成方式、解析/分析代码、结论**固化,并给出**后续 decode 专家相关性分析**
所需的数据接口与采集方案。

---

## 1. 代码地图(生成 + 解析 + 分析,均已落盘)

| 角色 | 文件 | 说明 |
|---|---|---|
| **采集**(直方图)| `third_party/sglang/.../layers/moe/kt_ep_wrapper.py`(`KT_PREFILL_EXPERT_HIST=1`,sglang commit `d8c460d6b`)| `apply()` 里累加 `topk_output.topk_ids` → 每遍 forward dump `[L×E]` 表;guard 住 graph capture / 单 token decode(避免 D2H memcpy 107030)|
| **复现** | `tools/longseq_dbg/gen_expert_hist.sh` | 拉 32-expert 生产配置 + 发真实多样文本,产出 `expert_hist.pt` |
| **解析/分析** | `tools/longseq_dbg/analyze_expert_hist.py` | 读 `.pt` → 偏度/冷专家/动态 vs 静态对比 + 导出 `_top{K}.pt`(常驻表)与 `_summary_top{K}.json` |
| **算子/带宽 bench** | `tools/longseq_dbg/npu_grouped_matmul_bench.py`、`hbm_probe.py` | §3.3/§3.4 三带宽反推 |

## 2. 数据 schema(`expert_hist.pt`,`torch.save` dict)

| key | 类型 | 含义 |
|---|---|---|
| `layers` | int64[L] | 实际出现的 `layer_idx`(升序);DSv4-Flash L=43 |
| `counts` | int64[L, E] | `counts[i,e]` = 第 `layers[i]` 层 expert `e` 的累计命中(token-assignment)数 |
| `tokens` | int64[L] | 每层累计 token-row 数 |
| `num_experts` | int | =256 |

**关键约定**:`counts` 是 **logical expert id**(router 原始 `topk_ids`,未经 `gpu_experts_mask` 重映射)。
每个 `.pt` 旁有 `*.meta.json`(provenance:prompt、config、caveats、实测值)。

⚠️ **直方图当前跨请求全局累加**(非按请求复位)。测代表性分布前别混退化数据(全同 filler token →
路由退化,只激活 ~44 专家)。真实现要**按请求清零 + 结束 dump 该请求池**(handoff D 步骤待补)。

## 3. 解析方法

```bash
python3 tools/longseq_dbg/analyze_expert_hist.py <hist.pt> <K>
# 输出:每层 topK 占激活比(mean/min/max)、冷专家/层、静态 prefix[0:K] vs 动态 topK、
#       skew ratio;落盘 <hist>_top{K}.pt(decode 常驻 [L×K] id 表)+ <hist>_summary_top{K}.json
```

最小 Python 解析:
```python
import torch
d = torch.load("expert_hist.pt")
counts = d["counts"].float()              # [L, E]
share = counts.sort(1, descending=True).values[:, :32].sum(1) / counts.sum(1)  # 每层 top32 占比
resident = counts.sort(1, descending=True).indices[:, :32]  # [L,32] 每层热专家 id
```

## 4. 结论(真实文本 ~89k token/层,32-expert 生产配置)

数据:`expert_hist_realtext.pt`(摘要 `expert_hist_realtext_summary_top32.json`)。

| K(HBM 常驻槽)| 动态 top-K 占激活 | 静态 prefix[0:K](现生产)| 动态增益 |
|---|---|---|---|
| 16 | 25.7% | 6.5% | 3.9× |
| **32** | **39.5%** | **12.8%** | **3.1×** |
| 64 | 58.4% | 25.7% | 2.3× |
| 96 | 71.7% | 38.0% | 1.9× |
| 128 | 81.5% | 50.0% | 1.6× |

1. **冷专家/层 = 0 → 长 prefill 256 专家全激活**。流式必须每层搬全 256;一个不能省。
2. **现生产静态 prefix-K ≈ K/256(随机水平)**:专家 0..K-1 不是热的,现 NPU 32 常驻只接 ~13% 激活,
   87% 砸 CPU。**当前放置策略实际未起作用。**
3. **prefill 直方图定的动态 top-K 多接 2–4×**:同样 32 HBM 槽,动态放置让 NPU 命中率 ~3×
   → decode 更多激活走快 NPU。偏度温和(~3×,负载均衡训练),80% 覆盖需 128/256 常驻
   → HBM 预算 vs 命中率可调。

**对子目标 2 的含义**:prefill 末取每层动态 top-K 写 `gpu_experts_mask`(替代静态 prefix),
热专家进常驻 HBM slot → decode 命中率 ~3×,**零额外 HBM**。

---

## 5. 后续:decode 专家相关性分析(规划,数据接口预留)

目标:量 **decode 阶段专家选择的相关性**,回答两问 →
(a) **prefill→decode 局部性**:prefill 定的热专家集,decode 命中率多高?(决定子目标 2 是否成立)
(b) **decode 步间相关性**:相邻 decode token 是否激活相似专家?(决定缓存刷新节奏 / §8 实时 evict 策略)

### 需要的数据(现直方图不够)

现 `counts` 是**聚合计数**,丢了时序与 token 粒度。decode 相关性需 **per-step、per-layer 的 topk id 序列**:
- `decode_topk[step][layer]` = 该 decode token 在该层的 top-6 logical expert id(+可选 weight)。

### 采集方案(待实现,注意 decode 走 NPU graph)

decode 默认在 NPU graph 内,`apply()` 不进 python → 直方图 guard 已跳过。采集 decode 数据需其一:
1. **分析专用 run 关图**:`EXTRA_FLAGS="--disable-cuda-graph"` 跑 decode,在 `apply()` 里(M==1 分支)
   记录每 step 每层 topk → `decode_topk.pt`。最简单,但非生产路径(够做相关性分析)。
2. graph host-callback 内采集(贴近生产,复杂)——留后。

### 分析产物(规划)

- **prefill→decode 命中率**:`mean over steps,layers [ decode top6 ∈ prefill_topK 的比例 ]`,扫 K。
  高 → 局部性成立、residency 划算(子目标 2 前提)。
- **步间 Jaccard**:相邻 decode step 同层 top6 的 Jaccard 相似度均值 → 高=时序局部性强=可懒刷新。
- **专家共激活矩阵**:`co[e1,e2]` = e1,e2 同 token 同层共现频次 → 看专家是否成簇(影响成组调入)。
- 复用 `analyze_expert_hist.py` 框架,新增 `analyze_decode_corr.py`(待建)。

> 注:本文 §1–§4 已落盘可复现;§5 是为下一步分析预留的接口与方案,数据采集代码待 decode 验证阶段实现。
