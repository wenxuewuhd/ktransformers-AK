# 2c-ii 集成设计:串行流式 prefill 接入 sglang(待 review)

> 状态:**设计待 review,未动代码**｜日期 2026-06-10｜配套 `longseq_prefill_handoff.md` §D。
> 前置:2a/2b/2c-i 已实测验证(流式机制 bitwise 正确、serial 单 slot 最优 ~13s、checkpoint
> 加载器 cosine 0.99964)。本设计把验证好的串行流式 loop 接进生产 prefill forward。

---

## 0. 目标 & 范围

- **目标**:`KT_PREFILL_STREAM=1` 时,长 prefill(`S≥T`)走纯 NPU 流式(逐层 H2D 256 专家 → 跑生产
  算子),把长 prefill MoE 从 hybrid 的 ~930s(32k)压到 ~13s(~80×);`S<T` 或开关关 → **现状不变**。
- **范围**:仅 2c-ii(forward 集成 + 启动建池 + 流式 slot)。decode 热专家池(子任务 4)单列,本设计预留钩子。
- **不做**:双缓冲(2b 证无收益);attention/MoE 重排(下方 §5 说明为何不需要)。

## 1. 核心设计思想:流式是**独立旁路分支**,不碰 hybrid 的 submit/sync

`KTEPWrapperMethod.apply` 现状 = hybrid(submit CPU → gpu_method.apply 32 专家 → sync → 合并)。
**流式分支在 apply 顶部分流、early-return,完全绕开 submit/sync/gpu_method 那套**:

```
apply(layer, dispatch_output):
    x, topk = ...
    histogram_record(...)                          # 现状(我加的)
    if _stream_enabled and _is_prefill(x) and x.shape[0] >= T and tp_rank==0:
        return _streaming_forward(layer_idx, x, topk)   # ← 新增旁路,early return
    ... 现状 hybrid 路径(645-707,一行不动)...     # ← B 在这里改
```

**意义(对 B 友好)**:流式路径**不调用** `submit`/`sync`/`_submit_cpu_npu_graph`/`gpu_method.apply`,
与 B 的"NPU↔CPU 并行 + submit/sync/overlap"改动**正交**。唯一共享编辑 = apply 顶部那个 if 分流
(~line 612 之后)。合并冲突面 = 1 个 if 块,可控(§6)。

## 2. 组件

### 2.1 启动期 pinned NZ 权重池(模块级,全 43 层)
- 复用 `stream_2c_ckpt_loader.py` 的 `load_layer_experts` + `process_after_loading`,**对 43 层全建**:
  每层 → NZ → D2H 到 pinned host。总 277GB pinned(2a 已验可行,DDR 1.5TB)。
- 存为**模块级**结构 `_STREAM_POOL[layer_idx] = (w13_host_nz, w2_host_nz, s13_bf16, s2_bf16)`
  (scale 小,可常驻 NPU 或随层)。
- 触发:`KT_PREFILL_STREAM=1` 时,在 server warmup 后 / 首个 streaming forward 前**惰性建一次**
  (或显式建池入口)。耗时 ~43×5.9s ≈ **4min**(优化项:预处理一次落盘 NZ 字节,启动 mmap+pin,~100s)。
- **关键**:建池要用 NPU(`npu_format_cast` NZ on-device)→ 与模型自身加载错峰,避免 HBM 抢占。

### 2.2 流式 weight slot(模块级,1 个,跨层复用)
- 1 个 `[256,H,2I]` + `[256,I,H]` int8 NZ slot(6.4GB HBM),**所有层共用、每层覆盖**(serial 单 slot,2b 最优)。
- 模块级单例(非 per-layer instance),首次用时分配。

### 2.3 流式 forward(`_streaming_forward`)
```
_streaming_forward(layer_idx, x, topk):
    slot13, slot2 = _ensure_stream_slot()
    h13, h2, s13b, s2b = _STREAM_POOL[layer_idx]
    slot13.copy_(h13); slot2.copy_(h2)              # H2D 本层 256 专家(default stream)
    out = npu_fused_experts(x, slot13, s13b, slot2, s2b, topk.topk_weights,
                            topk.topk_ids.int(), top_k)   # 生产算子,全 256,无 mask
    return StandardCombineInput(hidden_states=out)
```
- 直接调生产 `npu_fused_experts`(2c-i 已验 cosine 0.99964);**无 CPU、无 submit/sync**。
- topk_ids 用 router 原始 logical id(0..255),全在 NPU,不 mask。
- serial 安全性:forward 顺序执行,层 L compute 完才到层 L+1 apply,单 slot 不冲突(2b 已验)。

### 2.4 prefill 判定 & 模式选择
- `_is_prefill(x)` = `not torch.npu.is_current_stream_capturing()`(decode 走 graph replay,不进 eager apply;
  capture 期也跳过,避免 graph 里塞 H2D)。
- `T` = 服务器参数 `--kt-prefill-stream-threshold`(默认 512,§D-阈值)。`x.shape[0]` = 本 chunk token 数 M。
- 仅 `tp_rank==0`(单卡恒真)。

### 2.5 直方图按请求复位 + post-prefill 钩子(接子任务 4)
- 直方图(已实现)改**按请求复位**:prefill 首层清零、末层 dump 该请求池(需 request 边界信号;
  简单版:streaming forward 里按 layer_idx 回绕检测)。
- prefill 末:取每层 top-K → 写 `gpu_experts_mask` + 把 K 个热专家从 pool H2D 进**常驻 slot**,
  切 decode 走 hybrid(子任务 4,本设计只留 `_residency_hook(layer_idx, topk_ids)` 接口,不实现策略)。

## 3. 精确触点(文件/函数/行,基于当前 `kt_ep_wrapper.py`)

| 改动 | 位置 | 内容 |
|---|---|---|
| 模块级状态 | 顶部(~line 70,直方图旁)| `_STREAM_POOL`、`_stream_slot`、`_KT_PREFILL_STREAM` env、`_T` |
| 池/ slot 构建 | 新函数 | `_build_stream_pool()`、`_ensure_stream_slot()`、`_streaming_forward()` |
| 生产算子 | import | `npu_fused_experts`(进程内从 npu kernel 模块导,非 standalone,无循环 import 问题)|
| **分流点** | `apply` ~line 612 后(histogram 之后、should_log 之前)| `if 流式条件: return _streaming_forward(...)` |
| 服务器参数 | `server_args.py` | `--kt-prefill-stream-threshold`(默认 512)、`--kt-prefill-stream`(或纯 env)|
| hybrid 路径 645-707 | **不动** | B 的区域 |

## 4. 回退 / 安全

- **总开关** `KT_PREFILL_STREAM=1`(env)或 `--kt-prefill-stream`:关 → apply 完全走现状,**零风险零改变**。
- 池建失败 / OOM / pool 缺某层 → `try/except` 回退该层走 hybrid(打 warn,不崩)。
- `T` 默认 512 → 短 prompt 自动走 hybrid(现状)。
- 不改 create_weights / process_weights_after_loading(32-resident hybrid 权重照常加载,decode 用)。

## 5. 为什么**不需要**重排 attention/MoE(原本担心的难点)

sglang forward 本就**逐层**处理整个 chunk 的全部 token(layer0 attn+MoE → layer1 → ...)。所以
**每层 MoE 在一个 chunk 内只被 apply 一次**,M=chunk token 数 → 天然就是"serial 单 slot、每层流一次"。
只要 `chunked_prefill_size ≥ S`(单 chunk,长 context launcher 已设 32768),全程 = 43 次 H2D = 一遍扫层。
`S > chunk` 时多 chunk = 多遍扫层(§3.5.B 的 re-stream 惩罚,每 chunk +~13s),但仍远胜 CPU。
**∴ 不用解耦 attn/MoE,自然满足 layer-at-a-time。**

## 6. 与 B 的合并策略(§2)

- 共享文件:`kt_ep_wrapper.py`(分流 if + 新函数)、`experts_base.py`(**本设计不改**,流式不走 CPU 路径)。
- B 改 hybrid 的 submit/sync/overlap;C 加 streaming 旁路。**两者代码不重叠**,除 apply 顶部分流 if。
- 合并:apply 顶部按"先 histogram → 再 streaming 分流 → 否则 hybrid(B 的)"线性叠加,易 rebase。
- 建议:streaming 新增逻辑尽量收进**独立新函数/新模块文件**(如 `kt_stream_prefill.py`),apply 里只留一行分流调用 → 进一步减小共享编辑面。

## 7. 风险 / 待验(实现期)

1. **池建 4min 启动开销** → 先接受,后续落盘缓存优化。
2. **HBM 预算**:流式 slot 6.4GB + 32-resident hybrid 权重 0.8GB + KV + 激活。需确认 `mem_fraction_static`
   下放得开(M=32k 激活 ~GB 级)。streaming 时其实不需要 hybrid 的 32-resident 占 HBM,但它们 create_weights 时已驻;可接受(0.8GB)。
3. **H2D 与 NPU 算子在 default stream 串行**:已是 2b 最优,无并发互扰问题。
4. **request 边界**:直方图按请求复位需可靠的请求开始/结束信号;简单版用 layer_idx 回绕,稳健版需 scheduler 钩子。
5. **进程内 import** `npu_fused_experts`:确认 sglang 运行态导入无循环问题(standalone 才有)。

## 8. 测试计划

1. 开关 off → 回归:与现状 bit 对齐(decode tok/s、prefill 不变)。
2. 开关 on,短 prompt(<512)→ 走 hybrid,行为同现状。
3. 开关 on,长 prompt(8k/32k)→ 走流式:① 不崩、② 输出合理(logits sane / 与 hybrid 同 prompt 的输出 cosine 高)、
   ③ prefill 墙钟大幅下降(对比 hybrid 同长度)。
4. 直方图 dump 正确(按请求)。
5. 端到端长 prompt prefill 加速实测(目标 hybrid ~930s → 流式 ~13s @32k 单 chunk)。

## 9. 分步(每步可独立 review/验证)

- **2c-ii-a**:模块级池/slot + `_streaming_forward` + 分流 if(env 门控),先用 `KT_PREFILL_STREAM` 跑通
  单请求长 prompt 不崩 + 输出 cosine vs hybrid。
- **2c-ii-b**:服务器参数 `T` + 短/长自动路由 + 回退兜底。
- **2c-ii-c**:端到端 prefill 加速实测 + 池建落盘缓存优化(可选)。
- **2c-ii-d**:直方图按请求复位 + post-prefill `_residency_hook`(交子任务 4)。

---

**待你 review 的关键决策点**:
1. 流式 slot 直接调 `npu_fused_experts`(我倾向)vs 临时换 `layer.w13_weight` 走 `gpu_method.apply`?
2. 池建:启动惰性建(4min)先用 vs 一开始就做落盘缓存?
3. streaming 逻辑收进独立 `kt_stream_prefill.py`(减小与 B 共享编辑面,我倾向)vs 直接写在 kt_ep_wrapper?
4. `T` 走 server arg vs 纯 env(先 env 快)?
