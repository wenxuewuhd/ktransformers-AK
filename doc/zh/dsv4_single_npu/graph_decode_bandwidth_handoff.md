# Handoff — graph decode 子问题:CPU MoE 的 DDR 带宽利用率太低(~13%)

> **状态**:开放(新 session 起点)｜**日期**:2026-06-09｜**分支**:`dsv4_one_card_dev` @ `68f8556`
> **前置已完成**:graph decode 已用 `--kt-cpuinfer 24→96` 提速 ~1.7×(3.6→6.12 tok/s),见
> [Plan-and-Progress §6.6](DeepSeek-V4-Flash_Single-NPU_Plan-and-Progress.md) +
> [graph_decode_profiling_report.md](graph_decode_profiling_report.md)。
> **本文目标**:把 CPU MoE 的 DDR 带宽利用率从 ~13% 拉高,进一步压低 decode。

---

## 1. 问题陈述(一句话)

decode ~70% 时间是 CPU MoE,而 CPU MoE 是**内存带宽瓶颈**:每层把 ~140MB(最恶劣 160MB)int8 专家
权重从 DDR 搬一遍。但即便用 96 核,有效带宽也只 **~95 GB/s**,而 Kunpeng-920 理论 DDR4 峰值
**~751 GB/s** → **只用了 ~13%**。**把这 13% 拉高 = 直接的 decode 提速。**

实测有效带宽(真实权重,输出校验过,~140MB/层反推):

| cpuinfer | 每 NUMA 线程 | 单层 ms | 有效带宽 | 占峰值 |
|---|---|---|---|---|
| 24 | 3 | 4.5 | 31 GB/s | 4% |
| 48 | 6 | 2.67 | 53 GB/s | 7% |
| **96** | **12** | **1.48** | **95 GB/s** | **13%** |
| 128 | 16 | — | — | **真实权重 thrash 崩(1149ms/token)** |
| 192 | 24(满核) | 32 | — | 崩 |

**关键悖论**:13% ≪ 100%,说明**不是 DRAM 真饱和**,而是卡在**内存级并行度 / 访存模式 / 不能用满核**
(192 崩)。带宽没榨干 = 有空间。

---

## 2. Roofline 锚点(数字,新 session 直接用)

- 单专家(Q8_0)= 25.2M 元素 × 1.0625 B = **26.7 MB**;gate[2048,4096]+up[2048,4096]+down[4096,2048]。
- 最恶劣(top-6 全落 CPU)= **160 MB/层**,×43 = **6.9 GB/token**;平均(32 GPU 专家,~5.25 CPU)~140MB。
- 算力:NEON SDOT 16 int8-MAC/指令,~16 TMAC/s(192 核)→ 最恶劣单层算 ~10µs(×43=0.4ms/token)**可忽略**。
- 算术强度 **AI = 0.94 MAC/byte**,机器平衡点 **21 MAC/byte** → **深度 memory-bound**(差 ~22×)。
- 若有效带宽能到 50% 峰值(~375 GB/s):最恶劣 MoE → ~18ms/token(理论 ~54 tok/s 上界,仅 MoE)。

---

## 3. 候选方向(按性价比/风险排序,均未验证)

1. **根治多线程崩溃(优先,可能最划算)**:96→128→192 真实权重 thrash 崩。若能让更多核(更多内存级
   并行)不崩,带宽↑。先定位**为什么崩**(`top -H` 看是哪些线程争抢 / NPU host 线程 vs CPU worker;
   是否 oversubscription;`do_numa_job`/`InNumaPool` 自旋等待 + NPU host 轮询线程抢核)。kt-kernel
   线程池:`kt-kernel/cpu_backend/worker_pool.cpp`(`NumaJobDistributor::do_numa_job` :369 /
   `InNumaPool::do_work_stealing_job_async` :162 / `worker_thread` 混合自旋-park :385)。
2. **kernel 访存优化**:M=1 GEMV 的 Q8_0 反量化 + 点积访存模式。查每线程 outstanding loads / prefetch /
   cache-line 利用。kernel 在 `kt-kernel/operators/llamafile/moe.hpp`(`forward_one` :343,decode 走它)
   + vendored `third_party/llamafile/iqk_mul_mat_arm.inc`(arm82 SDOT 路径)。
3. **降精度 Q4**(访存量直接减半 → ~2× 上限,但要工程量):权重转 Q4_0/Q4_K,过
   `tools/p27_cpu_moe_reference_check.py` 对账 + F2 验收(精度会降,需评估)。`tools/batch_convert_w8a8_layers_mp.py`
   的 `--quant`。注意 K920 无 i8mm,Q4 kernel 在 dotprod-only 上要能跑且不 NaN(历史坑⑧)。
4. **减少 CPU 专家数**:`--kt-num-gpu-experts` 调大(更多专家上 NPU)/ 热专家放置(`--kt-expert-placement-strategy
   frequency`),CPU 搬的字节直接变少。但 NPU HBM/算力预算要重算(NPU 也有 attention)。

---

## 4. 方法论纪律(本 session 血泪,务必遵守)

- **任何"快参考"先验证输出非零/对账**。本 session 一个大坑:拿 `wrapper.forward()`(PATH_A)当快参考,
  但隔离环境无 ACL callback 订阅者 → host 回调永不触发 → **forward 从没执行、输出全 0**,"0.6ms" 是幻象,
  害我误追 GIL/NUMA 数小时。**真正会算的是 `run_pinned_forward_sync`(PATH_B)**:先
  `copy_inputs_to_cpu_buffers(h,ids,wt)` 再 `run_pinned_forward_sync(h,0)`,然后
  `copy_forward_output_to_device(h)` 取输出,**校验 `out.float().norm()>0`**。
- **扫线程数/找崩溃点必须用真实权重**。dummy(`KT_DUMMY_CPU_WEIGHTS=1`)路由退化、访存少 → 崩溃点失真
  (dummy cpuinfer=128 是峰、真实 128 崩)。dummy 只可用于"能不能跑通图"这类结构调试。
- **`pkill -f` 自匹配陷阱**:命令行里含 `sglang.launch_server.*--port 8001` 这种 pattern 时,内联
  `pkill -f` 会把执行命令的 shell 自己杀掉(exit 144)。**杀自己服务器用显式 PID**;脚本文件里的 pkill
  才安全(脚本 cmdline 不含 pattern)。
- **拉服务前**:`npu-smi info` 选空闲卡 + `ss -ltnp | grep :PORT` 看端口;只杀自己端口/PID,**绝不广播
  `pkill -f sglang.launch_server`**(会杀别的 session/容器的服务器)。共享机,卡 0/2 常被别的容器占。

---

## 5. 复现 / 测量工具(都现成)

- **隔离 CPU MoE 微基准**(秒级迭代,**记得加正确性校验 + 真实权重**):构造
  `KTMoEWrapper(layer_idx=3, weight_path=/workspace/models/cache/dsv4_layer3.gguf, cpuinfer_threads=N,
  threadpool_count=8, num_experts=256, num_experts_per_tok=6, hidden_size=4096, moe_intermediate_size=2048,
  gpu_experts_mask=...)`;`load_weights()`;循环 `copy_inputs_to_cpu_buffers + run_pinned_forward_sync`,
  `perf_counter` 计时,**断言输出 norm>0**。PYTHONPATH=`third_party/sglang/python:kt-kernel`,
  `ASCEND_RT_VISIBLE_DEVICES=<空卡>`。(本 session 用过的临时脚本已删,逻辑见报告 §2。)
- **服务器端到端**:`KT_CPUINFER=<N> KT_DECODE_TIMING=1 NPU_DEVICE_ID=<空卡> PORT=8001
  bash tools/p27_launch_ds4flash_npu.sh`。decode 日志会打 `[KT_DECODE_TIMING] tok#N cpu_moe_wall=…ms
  (sync=… on_cpu=… off_cpu=…)`(`KT_DECODE_TIMING` 是已提交的 env 门控计时桩,在
  `kt-kernel/python/experts_base.py` `run_pinned_forward_sync`)。`gen throughput` 看吞吐。
- **精度对账**:`tools/p27_cpu_moe_reference_check.py`(KTMoEWrapper vs PyTorch dequant,cosine)。
- **F2 验收**:`PORT=8001 bash tools/p27_curl_f2_prompts.sh`(4 prompt 连贯)。
- 真实权重加载 ~480s;dbg 结构调试可 `KT_DUMMY_CPU_WEIGHTS=1`(但**不可用于带宽/线程结论**)。

---

## 6. 代码地图(decode CPU MoE 路径)

- 入口(graph 回调):`third_party/sglang/.../moe/kt_ep_wrapper.py` `apply` → `_submit_cpu_npu_graph`
  → `torch_npu.npu._launch_host_func(...)` → `kt-kernel/python/experts_base.py` `run_pinned_forward_sync`
  → `cpu_infer.submit(forward_task)` + `cpu_infer.sync()`(TaskQueue,`kt-kernel/cpu_backend/task_queue.cpp`)。
- 计算:`forward_task` enqueue 到 TaskQueue worker → `MoeClass::forward`(moe.hpp:819)→ decode(qlen=1)走
  `forward_one`(moe.hpp:343)→ `pool->do_work_stealing_job`(InNumaPool)/ 跨 8 NUMA 的
  `dispense_backend()->do_numa_job`(moe.hpp:854)。实际 int8 GEMV 在 vendored llamafile(iqk_mul_mat_arm)。
- 线程池:`kt-kernel/cpu_backend/worker_pool.cpp`。`--kt-cpuinfer` = 总线程,`--kt-threadpool-count`=8(每
  NUMA 一个 subpool,`cpuinfer/threadpool` 线程/NUMA)。
- 配置:`tools/p27_launch_ds4flash_npu.sh`(`KT_CPUINFER` 默认 96)。

---

## 7. 起步建议

1. 先复现 §5 隔离微基准的带宽曲线(真实权重 + norm 校验),确认 95 GB/s @96 的现状。
2. 定位 §3.1「为什么 ≥128 崩」——这是「能否用更多核拿更多带宽」的钥匙,可能是最快的下一步。
3. 若崩溃可解 → 直接拿更多核 → 更高带宽;若不可解 → 转 §3.2 kernel 访存 或 §3.3 Q4。
4. 任何结论都用真实权重 + 正确性校验收口,再 F2 验收精度。

> 记忆文件 `graph-decode-profiling-findings.md`(Claude memory)有完整的实测数字 + 已证伪方向
> (GIL / NUMA 绑定 —— **别再走**)。
