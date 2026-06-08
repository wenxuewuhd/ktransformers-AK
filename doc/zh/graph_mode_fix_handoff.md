# Handoff — 修复 NPU graph capture 崩溃(坑⑥ `aclrtMemcpy 107030`)

> 目标:让 DeepSeek-V4-Flash 单卡 910B + KT CPU-MoE 在 **cuda-graph(NPU graph)ON** 下能干净
> 完成 graph capture 并跑通,恢复生产性能路径(基线 ~3.6 tok/s),取代当前 eager 回退。
>
> 仓库:`/workspace/code/ktransformers-AK`,分支 `dsv4_one_card_dev`。
> 环境固化见 `doc/zh/DeepSeek-V4-Flash_单卡910B_从0拉起服务全记录.md`(坑①~⑦)。

---

## 1. 现象

graph ON(默认)时,模型完整加载成功,但在最后 graph capture 阶段崩:

```
init_device_graphs → npu_graph_runner → cuda_graph_runner.py 的 capture
Exception: Capture cuda graph failed: aclrtMemcpy, error code is 107030
EE9999: Not allow to synchronize captured-stream, stream_id=42.
  rtMemcpy ... the current capture mode does not support this operation
```

即 graph capture 期间发生了 host↔device **同步 memcpy / 同步等待**,被 ACL capture 模式禁止。

> ⚠️ 注意:上面的调用栈/结论来自坑⑥**当时**的记录。在那之后已合入
> `b31d349 "NPU graph capture with ACL callback worker and CPU MoE in graph"`,
> 加了 host-callback 异步路径(见 §3)。**第一步必须先复跑、确认当前是否还崩、崩在哪一行**
> —— 不要默认坑⑥的栈仍然成立。

## 2. 当前默认行为(很关键,别误解)

- launch 脚本 **默认 graph ON**(不传 `--disable-cuda-graph`,也不强制 `KT_FORCE_SYNC_SUBMIT`)。
- `KT_FORCE_SYNC_SUBMIT=1` 是 **eager 调试/回退** 用的同步路径,**不是** graph 路径。graph 下不要设它。
- 控制开关在 `tools/p27_launch_ds4flash_npu.sh`:
  - 默认 graph ON;`EXTRA_FLAGS="--disable-cuda-graph"` → eager。
  - `SGLANG_NPU_PROFILE_ENABLE=1`(默认 0)会自动追加 `--disable-cuda-graph`(`:113-116`)。

复跑(默认 graph):
```bash
cd /workspace/code/ktransformers-AK
MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
NPU_DEVICE_ID=0 \
bash tools/p27_launch_ds4flash_npu.sh
# 盯加载尾段 graph capture;崩则抓完整栈与具体 aclrt 调用
```
已知可用回退(功能正确,~1.6 tok/s):加 `KT_FORCE_SYNC_SUBMIT=1 EXTRA_FLAGS="--disable-cuda-graph"`。

## 3. 代码地图(带 path:line)

### 3.1 graph 下 MoE 的分流(sglang KT EP wrapper)
`third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py`
- `apply()` 里判定是否走 graph:`use_npu_graph = tp_rank==0 and wrapper and _npu_use_graph_host_callback(x.device)`(约 `:517-520`)。
- 走 graph 时调用 `self._submit_cpu_npu_graph(dispatch_output, x)`(`:524-525`),否则普通 `submit()`。
- Step4 sync:`self.sync(x, cpu_already_synced=use_npu_graph)`(`:567`)——graph 下声称已由 callback 同步,只取 output。
- `_npu_use_graph_host_callback(device)`(`:55-69`):靠 `torch.npu.is_current_stream_capturing()` 或全局 `get_is_capture_mode()` 判 capture。
- `_submit_cpu_npu_graph()`(`:439-457`)—— **graph 安全路径的核心**:
  - `_ensure_npu_subscribe_report(stream)`(`:450` / 定义 `:72-80`,`torch_npu.npu._subscribe_report`)。
  - `self.wrapper.copy_inputs_to_cpu_buffers(x, topk_ids, topk_weights)`(`:452`)—— **D2H 把输入灌进 pinned buffer;这里的拷贝是否 blocking 是头号嫌疑**。
  - `torch_npu.npu._launch_host_func(stream, _kt_npu_graph_host_forward, (...))`(`:453-456`)—— 把 host callback 塞进 graph。
- `_kt_npu_graph_host_forward(args)`(`:84-87`):callback 体,调 `wrapper.run_pinned_forward_sync(hidden_states, stream_handle)`。

### 3.2 kt-kernel submit/sync/拷回(experts_base.py)
`kt-kernel/python/experts_base.py`
- `_should_bypass_stream_callback(device)`(`:43-47`):仅 `KT_FORCE_SYNC_SUBMIT=1` 时 True。
- `_wait_device(device)`(`:64-80`):**已含 capture 保护** —— `torch.npu.is_current_stream_capturing()` 为真时直接 return,否则 `torch.npu.synchronize()`。
  → 含义:若 capture 期间 `is_current_stream_capturing()` 返回 False(torch_npu 的 graph capture 未必如 CUDA 那样上报),保护会失效、`synchronize()` 仍会执行 → 107027/107030。**需实测确认 capture 期该函数返回值**。
- `submit_forward()`(约 `:621-695`):async 路径 `submit_with_cuda_stream` + `subscribe_ascend_stream`;bypass 路径 `_wait_device()` + `submit()`。
- `sync_forward()`(约 `:713-767`):async `sync_with_cuda_stream`;bypass `sync()`。
- `copy_forward_output_to_device()`(约 `:697-711`):`output_gpu.copy_(output_cpu, non_blocking=True)`(`:710`)—— 输出 H2D 用了 non_blocking。
- `run_pinned_forward_sync(...)`:graph callback 实际跑 CPU MoE 的同步版(在 pinned buffer 上),确认其内部没有再触发 stream 同步/同步 memcpy。

### 3.3 ACL callback worker(C++)
`kt-kernel/cpu_backend/ascend_callback_worker.{h,cpp}`
- `worker_main()`(`.cpp:26-39`):循环 `aclrtProcessReport(timeout)`(`:37`)派发 callback。
- `ensure_callback_worker(ctx)`(`:81-96`)、`ensure_stream_subscribed(stream)`(`:98-106`,`aclrtSubscribeReport`)、`shutdown_callback_worker()`(`:108-123`,`aclrtUnSubscribeReport`)。
- Python 侧入口:`kt_kernel_ext.init_ascend_callback_worker` / `shutdown_ascend_callback_worker`(experts_base `_ensure_ascend_callback_worker` `:51-62`)。

### 3.4 graph capture 入口(sglang)
- `third_party/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py`:
  - 全局 `is_capture_mode` + `model_capture_mode()`(`:374-388`)、`get_is_capture_mode()`。
  - capture 入口 `with model_capture_mode(): self.capture()`(`:676-682`),失败抛 `Capture cuda graph failed`。
- `third_party/sglang/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`:
  - `_capture_graph()`(`:104-117`):`with torch.npu.graph(graph, pool, stream, auto_dispatch_capture=True): out = run_once_fn()`。

## 4. 主要嫌疑点(待逐一证伪)

1. **`copy_inputs_to_cpu_buffers`(kt_ep_wrapper.py:452)的 D2H 拷贝是否 blocking** —— 在 capture 期做同步 D2H 会直接撞 107030。需确认它用的是 `non_blocking=True` 的 pinned-buffer 拷贝,且不在 callback 外同步。
2. **`_wait_device` 的 capture 保护是否真的生效** —— 即 capture 期 `torch.npu.is_current_stream_capturing()` 是否返回 True。若返回 False,则即便没设 FORCE_SYNC,某条 sync 仍会跑。需打点实测。
3. **`run_pinned_forward_sync` 内部** 是否还有 stream 同步 / 同步 memcpy(它是 callback 体,理应只在 pinned buffer 上算,不碰 device stream)。
4. **`_launch_host_func` 的 capture 语义** —— host callback 是否被正确 capture 进 graph,replay 时由 worker 线程派发,而非在 capture 期同步执行。
5. **`auto_dispatch_capture=True`(npu_graph_runner.py:110)** 与 KT callback/subscribe 的交互。

## 5. 验证手段

- 复跑 §2 默认 graph 命令,抓 capture 阶段崩溃完整栈 + 具体 aclrt API。
- 在 §4 各嫌疑点打点(尤其 `is_current_stream_capturing()` 返回值、各 `.copy_` 的 non_blocking 实参)。
- 离线对账工具确认 MoE 数值仍对:`tools/p27_cpu_moe_reference_check.py`(坑⑦,cosine≈1.0)。
- e2e 通过判据:`curl http://127.0.0.1:8000/health`=200;`/generate "中国的首都是"` 连贯;tok/s 回到 ~3.x。

## 6. 约束 / 备注

- 平台:Kunpeng 920(aarch64,无 SVE/i8mm)+ Atlas 910B,CANN 8.5.0,Python 3.11.14。
- 新 container 每次需 `apt-get install -y libhwloc-dev`(系统包,非 /workspace,不持久化);其余(.so/权重/子模块/symlink)已持久化。
- 生产勿长期开 `KT_DEBUG_HYBRID_MOE`(kt_ep_wrapper.py:483)/`KT_DEBUG_MOE_OUT`(experts_base.py:747)/`SGLANG_NPU_PROFILE_ENABLE`。
- 改动落点预计:`kt-kernel/cpu_backend/ascend_callback_worker.*`、`kt-kernel/python/experts_base.py`、`third_party/sglang/.../kt_ep_wrapper.py`。
  注意 sglang 是子模块(yy_repo `dsv4_release@a347a9ad5`),改动要在子模块里 commit 并同步父仓指针(参考本分支 `[chore](sglang)` commit 的做法)。
