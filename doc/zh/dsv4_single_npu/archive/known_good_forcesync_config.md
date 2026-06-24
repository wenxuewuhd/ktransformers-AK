> ⚠️ **已归档 — 本文的"force-sync=1 是已知正确配置"已过时**。竞态已被 commit `e5f53ad` 根治:**`KT_FORCE_SYNC_SUBMIT=0` 现在又对又快**(GPQA 75.25% > PR 73.23%、decode 稳态 16tok/s)。当前最优配置见 memory `depool-dynamic-correct-convert-folded`(depool+dynamic decode 18.9)和 `prefill-async-race-fixed-forcesync-off-fast`。下方内容仅留作历史。

# 单卡 DSV4 已知正确配置(force-sync)—— 给 debug 异步竞态的 agent 参考

> ## ✅ 已结案(2026-06-23,独占机受控实测)——下方 confound 已隔离,隔离实验不必再做
> - **污染源 = prefill 的 eager async CPU-MoE 路径**(`kt-kernel` cpuinfer `submit_with_cuda_stream` 的 `ACL_CALLBACK_NO_BLOCK`)。`KT_FORCE_SYNC_SUBMIT=1` 走同步 submit 即修复。
> - **side-stream 本身清白**:实验 #4 `[开 fs, 开 side-stream]` 实测 **正确 + bit 确定(跨 boot 4/4 prompt 逐字一致)**,decode **17.2 tok/s**(vs 只开 fs 的 ~12.6,side-stream +37%)。坏轮污染是 force-sync 缺失,不是 side-stream。
> - **`KT_SHARED_EXPERTS_STREAM` 对 DSv4 是 no-op**:接线在 `deepseek_v2.py`,DSv4 跑 `deepseek_v4.py`(py-spy 实证 forward_normal@v4:1117)。早先把它当嫌疑/当功臣都错了——那 +37% 全是 side-stream;强行移植到 v4 会崩 507011。已从 launcher 默认移除。
> - **最优配置 = `KT_FORCE_SYNC_SUBMIT=1` + `KT_SIDE_STREAM=1`**(均已 launcher 默认):正确 + bit 确定 + 17.2 tok/s。下方"正确但慢 ~10 tok/s"是只开 force-sync、没开 side-stream 的旧基线。
> - 详见 memory `offline-moe-check-needs-force-sync` + `shared-experts-stream-event-leak-eager`。

> 配套 bug report:[`moe_async_callback_race_BUGREPORT.md`](./moe_async_callback_race_BUGREPORT.md)
> 目标:你要把**异步 overlap 路径**修到和这里**同样正确**(再拿回吞吐)。本文件 = 那个"正确但慢"的基线全配置。

## 0. 一句话

- **正确路径**(本配置):`KT_FORCE_SYNC_SUBMIT=1` **且 side/shared stream 关**,decode ~10 tok/s,GPQA off **69.29%**(对齐 PR 73.23%,-3.94pp 在带内)。
- **坏路径**(eval 15%):`KT_SIDE_STREAM=1 KT_SHARED_EXPERTS_STREAM=1` **且 force-sync 关**,decode ~17-18 tok/s,但 **GPQA off ~15%(乱写)**、temp=0 问 15×17 答 "245"。

> ★★ **重要:confound 未隔离。** 坏→好之间**同时改了两样**:(1) force-sync 关→开,(2) side/shared stream 开→关。所以污染源是 **force-sync 缺失(异步 MoE 竞态)/ side+shared stream 实现 / 两者交互** 中的哪个**没定论**。证据其实**略偏向 side/shared stream**:`[关fs,关stream]`(Ctrl A)→对,`[关fs,开stream]`→错,这俩唯一差是 stream;但 Ctrl A 仅 3 样本、早期 `[关fs,关stream]` 也出过乱写,故不能定论。
>
> **需要的隔离实验(请先做,再决定改哪条代码):**
> - `[开 force-sync, 开 side/shared stream]` → 若**对**:force-sync 是修复、side-stream 本身没问题(且能拿回 ~17 tok/s,最优解);若**错**:side-stream 与 force-sync 冲突。
> - `[关 force-sync, 关 side/shared stream]` 跑**全量/几十题**(非 3 条)→ 若**对**:side/shared stream 才是污染源、force-sync 非必需;若**错**:异步 submit 是污染源、force-sync 必需。
>
> 两个嫌疑代码点:① 异步 submit(`kt-kernel/python/experts_base.py` L644 `bypass` 分支,`submit` vs `submit_with_cuda_stream`);② **side/shared stream 实现**(sglang KT 侧 `KT_SIDE_STREAM`/`KT_SHARED_EXPERTS_STREAM` 走的多 stream 路径,grep 这两个 env 找代码)。

## 1. 版本 / 路径

| 项 | 值 |
|---|---|
| 主仓 | `/workspace/code/ktransformers-AK` @ `a64e45e` (branch `dsv4_one_card_dev`) |
| sglang | `third_party/sglang` @ `e7b3f8b25` (branch `kt-sidestream-sharedstream`) |
| 模型(NPU 侧) | `/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8`(**compressed-tensors** W8A8,config.json 字段齐全,无需 override) |
| CPU MoE 权重 | `/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf`(MXFP4,43 层,3.4GB/层) |
| 启动脚本 | `tools/p27_launch_ds4flash_npu.sh` |
| 硬件 | Ascend 910B3 单卡(逻辑 device 0)+ Kunpeng-920 CPU MoE offload |

## 2. 完整启动(正确配置)

```bash
cd /workspace/code/ktransformers-AK
export NPU_DEVICE_ID=6           # 任一空闲卡(HBM~3.4GB);进程内为逻辑 npu:0
export PORT=8200
export SKIP_WARMUP=1             # 评测不需 warmup;但冷启前几条慢(mxfp4 mmap lazy page-fault),要先打 ~10 条暖机再测
export KT_MXFP4_DEPOOL=1         # CPU 专家走 mxfp4 gguf 模板(否则 Q8_0,decode ~2× 慢)
export KT_NUM_GPU_EXPERTS=32     # 32 expert 常驻 NPU,其余 224 走 CPU
export KT_FORCE_SYNC_SUBMIT=1    # ★正确性开关(不是确定性!)不设=异步竞态=静默算错
export CHUNKED_PREFILL_SIZE=4096 # ★必须 ≥ 最长 prompt(GPQA max=2577);默认 2048 会让长 prompt 切 chunk → NSA compressor 跨 chunk 崩(坑⑯)
# 不设 MODEL_PATH/QUANTIZATION → 默认就是上面那份 compressed-tensors(A)
# 不开 KT_SIDE_STREAM/KT_SHARED_EXPERTS_STREAM(本基线没开;与 force-sync 一起是否既快又对未验证)
# 不开 KT_DYNAMIC_RESIDENT(每请求切换 ~108s,拖死)/ KT_PREFILL_STREAM(短 prompt 用不上)
nohup setsid bash tools/p27_launch_ds4flash_npu.sh > /tmp/server.log 2>&1 < /dev/null & disown
# ↑ 前台 setsid 起、PPID=1 脱离任务生命周期,否则后台任务结束时服务被回收
#   (日志会出现 "TBE Subprocess ... main process disappeared" = 被回收,不是崩)
```

启动脚本最终 exec 的 sglang(关键 flag):
```
python -m sglang.launch_server --model-path <A> --device npu --tensor-parallel-size 1 \
  --page-size 128 --attention-backend ascend --quantization compressed-tensors \
  --disable-shared-experts-fusion --dtype bfloat16 --trust-remote-code \
  --mem-fraction-static 0.85 --disable-radix-cache --max-prefill-tokens 65535 \
  --context-length 65536 --watchdog-timeout 18000 --skip-server-warmup \
  --kt-method LLAMAFILE --kt-num-gpu-experts 32 \
  --kt-weight-path /workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf \
  --kt-threadpool-count 8 --kt-cpuinfer 128 --max-running-requests 1 \
  --chunked-prefill-size 4096 --host 0.0.0.0 --port 8200
```

启动脚本还 export 的关键 NPU/算子 env(都在 `p27_launch_ds4flash_npu.sh` 里,无需手动):
`ASCEND_USE_FIA=1 USE_FUSED_COMPRESSOR=1 LI_KV_DTYPE_INT8=1 USE_PA_DECODE=1 USE_PA_PREFILL=1 USE_FUSED_HC_POST_ASCENDC=1 USE_FUSED_HC_PRE_ASCENDC=1 USE_NPU_MOE_GATING_TOP_K=1 USE_FUSED_TRANSPOSE_BATCHMATMUL=1 USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=1 IS_DEEPSEEK_V4=1 TASK_QUEUE_ENABLE=1 STREAMS_PER_DEVICE=32 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，并 source CANN/ATB/vendor set_env。

冷启 + 建池 ~3-5min;`grep "fired up and ready" /tmp/server.log`。

## 3. 正确性自检(必过才算对)

prompt 走 OpenAI chat 接口(DeepseekV4 架构自动用内置 `encoding_dsv4`,non-thinking 时 prompt 结尾 `<｜Assistant｜></think>`):
```bash
# 1) 15×17(temp=0, non-thinking)→ 必含 255。坏路径会答 245/乱写。
curl -s http://127.0.0.1:8200/v1/chat/completions -H 'Content-Type: application/json' -d \
 '{"model":"x","messages":[{"role":"user","content":"What is 15 multiplied by 17?"}],"temperature":0,"max_tokens":80,"extra_body":{"chat_template_kwargs":{"thinking":false}}}'
# 2) GPQA temp=0 抽几题:输出应是连贯分步推理 + 结尾 "ANSWER: X"(不是 "universe pleiotropy" 那种退化乱写)
```
force-sync 下 temp=0 是确定的(3/3 字节一致);坏路径请求间乱跳。

## 4. 数据集评测(EvalScope,off / non-thinking)

```bash
cd /workspace/code/dsv4-acc-compare
# 暖机 ~10 条后:
GEN='{"temperature":1,"top_p":1,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}'
evalscope eval --model /workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  --api-url http://127.0.0.1:8200/v1/chat/completions --api-key EMPTY --eval-type openai_api \
  --datasets gpqa_diamond --generation-config "$GEN" --eval-batch-size 8 --repeats 1 \
  --work-dir eval_results/<TAG>
# 判分:acc>=0.5 占比;reviews/.../gpqa_diamond_default.jsonl 里 sample_score.score.value.acc
```
注意坑:① **必须 chat 接口**(不能纯文本);② thinking 默认关,测 off 别传 true;③ **别用 `--use-cache`**(续跑 hang);④ `max_tokens` 给大(32768);⑤ temp=1 是采样,看全量 198 或 `--repeats 3`,别拿小样本下结论。

## 5. 实测结果(本基线)

| 配置 | decode | GPQA off | 正确性 |
|---|---|---|---|
| **force-sync(本基线)** | ~10 tok/s(p25-p75 8.7-12.3) | **88/127=69.29%**(-3.94pp vs PR 73.23%,带内)| ✅ |
| async(默认,你要修的)| ~17-18 tok/s | ~15%(127 题中途) | ❌ 退化乱写 |

- off 完整完成的是 127/198 那轮(force-sync,旧 `chunk=2048`,在第 128 题那道 2577-token 上撞坑⑯崩);带 `chunk=4096` 的重跑健康但只跑了 17 题(人为叫停)。**off 已判定对齐**;on(thinking)未测。
- prompt encoding 与 fork `@298193eb3` **逐文件 diff 完全一致**(`encoding_dsv4.py`/`serving_chat.py` 字节相同),且离线渲染字节一致、prompt_tokens=13。
- modelslim 注意力线性离线对账 cos 0.99997。所以"算法/反量化/config/prompt"都已排除,**剩下就是异步 MoE 竞态**。

## 6. 你的目标

让**异步 overlap 路径**(`bypass=False`,`submit_with_cuda_stream` + `ascend_callback_worker`)产出和 force-sync **相同正确**的 MoE 输出,把 decode 拿回 ~17-18 tok/s。修好后,用第 3 节自检 + 第 4 节全量 198 off(以及 on)在**异步路径**上复测,对齐 PR(off 73.23% / on 86.36%)。怀疑点见 bug report 第 3 节(主怀疑:异步写 `output_cpu[slot]` 与 NPU 消费之间缺 barrier / 槽位 `buffer_depth` 复用前没 drain)。
