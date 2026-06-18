# 后续工作项(动态热专家 resident 落地之后)

> 来源:Session C，2026-06-18。动态热专家 resident 已折叠进流式 prefill、零切换停顿
> (sglang `c905e5fa3` / parent `6cd0cea`)。本文是剩余的优化空间,按优先级。

## 当前状态(基线)

- 长 prompt warm:**prefill ~20s(流式 depool + 每请求动态热专家 share~0.6)→ 直接 decode**,无切换。
- 一次性:**拉起服务 ~180-310s** + **MXFP4 池懒构建 ~347s**(在第一个长请求里)。
- DDR 省 137GB(depool)。

---

## Item 1(主):MXFP4 depool 池构建 → 挪到 startup + 并行读 【建议新 session 做】

**现状(慢点)**:`_build_mxfp4_pool`(`kt_stream_prefill.py:359`)在**首个流式 forward**(`maybe_streaming_forward:874`,`if not _MXFP4_POOL`)才**懒构建**,且 `_load_layer_mxfp4`(:325)是**串行 safetensors 读**,~8s/层 × 43 = **347s**,全算在第一个长请求头上。

**现成模板**:W8A8 路**已经**做了"load 时后台并行读 + load 末 NZ-cast":
- `_start_bg_reads`(:231)在 `maybe_reserve_slot`(process_weights_after_loading,model-load 时)启动 8-worker **O_DIRECT 并行读**,与模型加载重叠;
- `_finish_bg_build`(:255)/ `_build_pool_parread`(:270)在 load 末收尾;注释(:138)说 O_DIRECT parread 比 safe_open **~5× 快**。

**要做**:
1. 把 depool 池构建**触发点从懒(:874)挪到 model-load**(在 `maybe_reserve_slot` 的 `_KT_MXFP4_DEPOOL` 分支里启动,现在那里是 `return`,:540 附近);
2. 把 depool 的读**并行化/O_DIRECT 化**(把 W8A8 的 `_start_bg_reads`/`_build_pool_parread` 模式套到 MXFP4 codes 上 —— MXFP4 是 4bit、量更小,读更快)。

**预期**:第一个长请求不再付 347s;startup 吸收掉(并行后可能 347s→~70-100s);对所有请求都是干净 ~20s。
**坑**:① 140GB pinned host 池在 load 时分配,确认 host RAM 够 + 分配时机(W8A8 路注释提过 load-peak 内存紧);② depool 跳过 W8A8 槽,HBM 不是约束(池在 host)。

---

## Item 2:流式 convert 换 G 的 advance kernel(115ms vs 230ms)【低效、边际】

STATUS `QUEUED UPGRADE`:G 的 advance kernel 整层 convert ~115ms(当前 230ms),同输出契约 drop-in。
流式 prefill 每层转 256 → prefill 内部省 ~5s(230→115 × 43)。**边际、不阻塞**,等 G 那边稳了顺手换。

---

## Item 3:decode 热专家**净收益**复测 → 独占机 + B 的 overlap 【验证/B 域,非本仓 code】

热专家把 off_cpu 地板砍 -45%,但**共享机 NUMA 噪声盖住地板**(median 是地板 3-5×),net decode tok/s ≈ 0。
要兑现:① **独占/空载机**复测(去噪);② **Session B 的 CPU↔NPU overlap**(把 off_cpu 藏到 NPU 背后,floor 收益才变 tok/s)。
这不是本仓能改的 code,是测量条件 + B 域。

---

## Item 4:长 prompt(> chunk)的 per-chunk 重复流式 【低优先】

流式 convert 是**按 chunk 重跑**的(每 chunk 转 256×43)。prompt > `CHUNKED_PREFILL_SIZE` 时翻倍。
缓解:把 chunk 开大到装下整 prompt(HBM 受限,本卡 ~16-32k)。根治(跨 chunk 缓存已转 NZ)= HBM 重,低优先。
另外:DSv4 **NSA chunked-prefill 有个 bug**(`set_compress_buffer` assert,prompt 跨 chunk 边界崩),`CHUNKED_PREFILL_SIZE ≥ prompt` 单 chunk 可绕开 —— 这是注意力路的 bug,不是本仓。

---

## Item 5:pin 税完全消除 【深改,handoff 遗留】

depool 仍另 pin ~140GB MXFP4 池(比 W8A8 277GB 省一半)。彻底消 pin 税需 NPU **复用 CPU 的 MXFP4**(不另 pin)或流式 unpinned。深改,优先级低。

---

## 机制备查:那 100s 的真因(已解决,别再踩)

运行时写"模型已加载的注册权重 param"(`layer.w13_weight`)触发 **Ascend 设备级 weight-region coherence flush**,把 NSA 每层 `.item()` 同步拖慢 ~100s。
修法已落地:**init 时(图捕获前)remap resident param 到 caching-allocator**(`maybe_reserve_slot` 里 `_p.data = _p.data.clone()`)。
详见 memory `npu-weight-region-write-stalls-nsa`。**别再用 alloc/HBM/graph/side-stream 方向去解 —— 全证伪过。**
