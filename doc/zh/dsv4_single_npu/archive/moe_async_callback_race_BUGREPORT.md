> ⚠️ **已归档 — 此竞态已被根治(commit `e5f53ad`)**。修法不是"force-sync=1 掩盖",而是 prefill 改同步 `cpu_infer.submit` + 无条件 `_wait_device` + 保留 `subscribe_ascend_stream`。**`KT_FORCE_SYNC_SUBMIT=0` 现在又对又快**(GPQA 75.25%、decode 稳态 16tok/s、与 force-sync 路径逐字对齐)。下方"force-sync=1 绕过"是旧的掩盖式结论,仅留作竞态定位过程的历史记录。最新见 memory `prefill-async-race-fixed-forcesync-off-fast`。

# BUG REPORT: 单卡 KT MoE 异步 stream-callback 路径疑似竞态(导致 serving 静默算错)

> ## ✅ 已结案(2026-06-23,独占机 py-spy + 受控 A/B)——下方"并列嫌疑"已裁决
> - **根因 = prefill 的 eager async CPU-MoE 路径**:`kt-kernel/cpu_backend/vendors/ascend_npu.h` 的 `cudaLaunchHostFunc` 用 `ACL_CALLBACK_NO_BLOCK` → stream 不等 host callback 完成 → H2D 读 `output_cpu` 时 CPU 还没写完 → 静默污染。`KT_FORCE_SYNC_SUBMIT=1` 走同步 submit/sync 绕过即修复。
> - **线索 E(side/shared stream)裁决 = 清白**:`[开 fs, 开 side-stream]`(实验 #4)实测 **正确 + bit 确定 + 17.2 tok/s**。force-sync 不改 decode 的 side-stream 路径;side-stream 纯调度加速、token byte-identical。坏轮污染来自 force-sync 缺失,不是 side-stream。
> - **`KT_SHARED_EXPERTS_STREAM` = no-op(DSv4)**:接线在 `deepseek_v2.py`,DSv4 跑 `deepseek_v4.py`(py-spy 实证)。从没真正运行过;移植到 v4 会崩 507011。
> - **`NO_BLOCK→BLOCK` 试修**:消了粗污染(乱码→答案对),但 128 线程残留 benign FP 抖动(非数据损坏);已 revert,生产用 force-sync。
> - 最优配置 = `KT_FORCE_SYNC_SUBMIT=1` + `KT_SIDE_STREAM=1`(launcher 默认)。详见 memory `offline-moe-check-needs-force-sync`。
> **下方表格 #4「未测」/「并列线索 E」已过期,仅留作调查记录。**

> 给 review 代码的 agent。请**只读+分析**,先别改;目标是确认/否定下面的竞态怀疑,定位根因。
> 现象已用数据集复现并定位到开关级别;底层根因待你确认。

## 1. 现象(已复现)

单卡(Ascend 910B + Kunpeng CPU MoE offload)serving。**不开 `KT_FORCE_SYNC_SUBMIT=1`** 时:
- MoE 输出**错误但非零**:服务正常起、能出连贯英文句子,但答案错。
- GPQA-Diamond off 全量:**关 = ~15%(退化乱写,低于随机 25%)/ 开 = ~72%(对标 PR 73.23%)**。
- temp=0 单条:问 "What is 15 multiplied by 17?(>250?)" → 关时答 **"245"**(算错),开时答 255(对)。
- 早期 temp=0 自检:关时同一 prompt 在请求间结果乱跳(对 / 德语乱写 / 除法),开时 3/3 字节一致正确。

**关键**:输出是"半连贯但错",不是"全 0/全乱码"。说明 CPU MoE **大部分时候算对了,偶发污染**——典型同步/竞态特征,不是 kernel 算错或权重错。`force-sync` 串行化提交可绕过,但那是**掩盖**,不是根治。

## 2. 开关切换了什么(已定位)

`kt-kernel/python/experts_base.py`:
- `_should_bypass_stream_callback()` (L139-143):`KT_FORCE_SYNC_SUBMIT=1` → `bypass=True`。
- MoE forward (L644-683):
  - `bypass=True`(force-sync):`self.cpu_infer.submit(immediate_task)` —— 同步/legacy 路径。
  - `bypass=False`(默认):`kt_kernel_ext.subscribe_ascend_stream(cuda_stream)` 然后 `self.cpu_infer.submit_with_cuda_stream(cuda_stream, immediate_task)` —— **异步**,经 ACL callback queue。
- buffer 槽位轮转:`current_slot = layer_idx % buffer_depth`,`next_slot=(current_slot+1)%depth`(L640-641);deferred 专家写 `output_cpu[next_slot]`,并用 `_layer_has_pending_deferred` 跨层传状态(L645,664,683)。

机制注释在 `experts_base.py` L117-135 + `kt-kernel/cpu_backend/vendors/ascend_npu.h` L17-19:`submit_with_cuda_stream` 用 `aclrtLaunchCallback`,需要 `aclrtSubscribeReport`+`aclrtProcessReport` 的订阅线程(`ascend_callback_worker.cpp`,`init_ascend_callback_worker`)来派发回调。
- ⚠️ 注释 L125-128 还写着 "kt-kernel does NOT currently start such a worker ... callbacks silently never fire → output_cpu all-zero (TODO Phase 3)",但 L132-135 又说 worker 已存在能 overlap。**注释自相矛盾/可能过时**,请核对 worker 到底有没有在 serving 路径被启动、以及现在的真实行为。

## 3. 我的怀疑(待确认)

> ★★ **先读:confound 未隔离(2026-06-22 更正)。** "好/坏"两轮**同时改了两个变量**,不是只有 force-sync:
> | 配置 | force-sync | side/shared stream | 结果 |
> |---|---|---|---|
> | 坏(off 15%)| 关 | **开** | 错(245/乱写)|
> | 好(off 69%)| **开** | 关 | 对 |
> | Ctrl A(早期 3/3 对)| 关 | 关 | 对(小样本)|
>
> 所以污染源可能是 **(下面 A-D 的异步 submit 竞态)** 或 **side/shared stream 实现(下面 E)** 或两者交互,**未定论**。证据其实**略偏向 E**:`[关fs,关stream]`→对、`[关fs,开stream]`→错,唯一差是 stream;但样本少不能定论。**请先做隔离实验**(见 §5 末)再决定查 A-D 还是 E。

**主线索 A-D:异步 submit 缺 barrier** —— 保证 `output_cpu[slot]` 被 CPU MoE 写完之后,NPU 才去消费它 / 槽位才被复用:

- **怀疑 A(主):** CPU MoE 异步写 `output_cpu[current_slot]` 与 NPU 侧读取/拷回(output_cpu→output_gpu)之间没有正确的依赖序。NPU 偶尔在 CPU 写完前就读 → 读到**上一次/半完成**的数据 → 偶发错 token → 长生成被带偏 → 答案错。(全 0 会更惨;半连贯=偶发,符合竞态。)
- **怀疑 B:** 槽位轮转 `buffer_depth` 太浅 / `sync_with_cuda_stream` 没在复用前 drain。layer L 的异步任务还没完,layer L+depth 就复用了同一 `output_cpu` 槽 → 覆盖/读脏。
- **怀疑 C:** ACL callback worker 的派发顺序/完成信号与 NPU graph stream 的依赖没建立(`subscribe_ascend_stream` / `aclrtProcessReport` 的 ordering)。callback 触发了(所以非零),但"完成"早于实际写完,或与拷回乱序。
- **怀疑 D:** deferred 专家路径(`max_deferred_experts_per_token>0` + 跨层 `_layer_has_pending_deferred`)在异步下把 `output_cpu[next_slot]` 写串。本次 serving 默认 `max_deferred=0`,但请确认默认下 deferred 分支确实不进。

**并列线索 E:side/shared stream 实现(同等嫌疑,甚至证据更偏向它)** —— 坏那轮**额外开了** `KT_SIDE_STREAM=1 KT_SHARED_EXPERTS_STREAM=1`(多 NPU stream 重叠 side/shared expert 计算)。grep 这两个 env 找代码:它新增的 stream 与 MoE/attention 主 stream 之间,**event/依赖是否建全**?是否与 CPU MoE 的 output_cpu 拷回、或与 force-sync 的同步点冲突?`[关fs,开stream]` 错而 `[关fs,关stream]` 对 → 这条很可疑。请重点对比"开/关 side/shared stream"在 `[关 force-sync]` 下的数值差异。

## 4. 请重点看的文件

1. `kt-kernel/python/experts_base.py` —— forward (L600-710),尤其异步 submit 后**哪里 sync**、output_cpu→output_gpu 拷回在哪、是否 ordered after CPU 完成;`run_pinned_forward_sync` (L685+) 对比同步路径怎么保证序。
2. `kt-kernel/cpu_backend/ascend_callback_worker.cpp` / `.h` —— 订阅线程派发/完成语义,是否保证回调完成 → 下游 NPU op 的 happens-before。
3. `kt-kernel/cpu_backend/cpuinfer.h` —— `submit_with_cuda_stream` (L87)、`sync_with_cuda_stream` (L112、`allow_n_pending` 语义)。
4. `kt-kernel/cpu_backend/vendors/ascend_npu.h` —— L17-19 的 callback 路径注释 + TODO Phase 3 是否已落实。
5. sglang 侧消费 MoE 输出处:`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`(GPU/CPU 专家 merge、output_gpu 何时被读)。

## 5. 给 reviewer 的具体问题

1. 默认(async)路径,从 `submit_with_cuda_stream` 到 NPU 读 `output_cpu`/`output_gpu` 之间,**有没有显式 `sync_with_cuda_stream` 或 stream 依赖**保证 CPU 写完?在哪一行?
2. `buffer_depth` 是多少?够不够覆盖在飞的异步任务数(每层一个,流水多深)?槽复用前是否 drain?
3. `ascend_callback_worker` 在 serving 默认是否启动?若没启动,async 路径理应全 0(与现象"半连贯非零"矛盾)——所以它**应该**启动了,那竞态就在"启动了但同步不足"。请确认。
4. 同步路径(`bypass`/force-sync)为什么就对?对比它多做了哪一步 sync —— 那一步就是异步路径缺的。

## 6. 复现

公共底:`cd /workspace/code/ktransformers-AK`;每条都 `NPU_DEVICE_ID=<空闲卡> PORT=8200 SKIP_WARMUP=1 KT_MXFP4_DEPOOL=1 KT_NUM_GPU_EXPERTS=32 CHUNKED_PREFILL_SIZE=4096 bash tools/p27_launch_ds4flash_npu.sh`,再叠加下表的开关。**前台 `setsid ... & disown` 起,防回收。**

| # | 额外 env | 已测? | 结果 |
|---|---|---|---|
| 1 | `KT_FORCE_SYNC_SUBMIT=1`(无 stream)| ✅ | **对**(off 69%,~10 tok/s)—— 已知正确基线 |
| 2 | `KT_SIDE_STREAM=1 KT_SHARED_EXPERTS_STREAM=1`(无 fs)| ✅ | **错**(off 15%,15×17→245,~18 tok/s)|
| 3 | (都不设)| ⚠️ 仅 3 样本 | 对(Ctrl A);早期 modelslim 同配置出过乱写 → **需全量复测** |
| **4** | **`KT_FORCE_SYNC_SUBMIT=1` + `KT_SIDE_STREAM=1 KT_SHARED_EXPERTS_STREAM=1`** | ❌ **未测** | **关键实验**:对→force-sync 修复且能拿回 ~18tok/s;错→side-stream 与 fs 冲突 |

**隔离逻辑**:#3 全量复测 + #4,就能把"异步 submit(A-D)" vs "side/shared stream(E)"分开。
- #3 对 & #2 错 → 锅在 **side/shared stream(E)**,force-sync 非必需。
- #3 错 → 锅在 **异步 submit(A-D)**,force-sync 必需。
- #4 对 → 直接拿到"快且对"(force-sync + stream),最优。

自检:`/v1/chat/completions` 发 "What is 15 multiplied by 17?" temp=0 thinking:false(暖机 ~10 条后);稳定 255=对,245/乱写/请求间乱跳=错。数据集:EvalScope gpqa off(见 `known_good_forcesync_config.md` §4),≥几十题别拿 3 条下结论。

## 7. 备注

- 根治目标:让 async(overlap)路径也正确,拿回吞吐(force-sync 关掉 NPU/CPU overlap,decode ~8-13 vs async ~17-18 tok/s)。
- 未验证:`force-sync + side/shared stream` 是否既快又对(本次是连带改的,没单独隔离)。
- 相关 memory:`offline-moe-check-needs-force-sync`(离线对账同源,output_cpu 全 0)。
