# 验证 prompt — workerpark 开机预热收益（贴给主干新 session）

> 用途：在主干 `dsv4_one_card_dev`（worktree `/workspace/code/ktransformers-AK`，@ c2ffc2c 起）独立复现并确认
> 去掉 `--skip-server-warmup` 的开机冷启收益。整段贴进新 session 即可。
> 详细数据/机制见同目录 `workerpark_tune_handoff.md` 的 Closeout + Follow-up。

---

```
你在主干 dsv4_one_card_dev（worktree /workspace/code/ktransformers-AK，@ c2ffc2c）独立验证
Session F workerpark 调查的收益，确认是否值得保留。背景：

park 阈值假说已被受控 A/B 证伪（worker_pool.cpp 未改），唯一采纳的改动是启动脚本
tools/p27_launch_ds4flash_npu.sh 把 --skip-server-warmup 改成 ${SKIP_WARMUP:-1} 门控：
SKIP_WARMUP=1（默认，传 --skip-server-warmup，旧行为/基线）；=0 则开启 sglang 开机预热。

【要验证的 claim（仅启动侧收益，稳态 ~16 tps 不变）】
开机后第一发 20-tok 请求：SKIP_WARMUP=1 约 3.37s → SKIP_WARMUP=0 约 2.13s（−37%，~1.2s）；
到稳态从第 4 发提前到第 2 发；60-tok 吞吐 11 → 13.9 tps。冷启慢在 NPU graph/prefill 一次性
建立，不在 cpu_moe_wall（skip 第一发的 cpu_moe_wall 仍只 25–65ms，可用 KT_DECODE_TIMING=1 核）。

【A/B 协议（单变量=SKIP_WARMUP，其余全同，每臂 boot 两次取 n=2）】
拉服务（在空卡上，端口自选如 8022）：
  KT_DECODE_TIMING=1 KT_CPUINFER=128 NPU_DEVICE_ID=<空卡> PORT=8022 SKIP_WARMUP=<0或1> \
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  MODEL_PATH=/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8 \
  bash tools/p27_launch_ds4flash_npu.sh
等日志出现 "fired up and ready" 后【立刻、在任何其他流量之前】连发 3 个 20-tok 请求计时：
  curl -s http://127.0.0.1:8022/generate -H 'Content-Type: application/json' \
    -d '{"text":"Explain what a transformer model is:","sampling_params":{"max_new_tokens":20,"temperature":0}}' \
    -o /dev/null -w "%{time_total}\n"
比较两臂 req1/req2/req3，看 SKIP_WARMUP=0 的 req1 是否稳定低于 =1（两次 boot 区间不重叠即坐实）。

【坑（都踩过，别重复）】
- 选卡：npu-smi info 看 HBM，挑 ~3.4GB 那种空卡（模型要 ~48GB）；拉前确认 AICore 0%。
- NPU 显存释放有延迟：杀掉服务后同一张卡的显存不会立刻回收，别在同卡马上重拉（会 OOM）——
  换一张空卡或等释放。
- 千万别用 pgrep/pkill -f "port 8022" 找进程：会自匹配到你自己的监控/shell 命令，曾把启动脚本
  半路打死。要匹配就用 "sglang.launch_server"。只杀自己的 PID，绝不广播 pkill。
- 长跑服务自己终端前台拉（后台会被回收）；只杀自己 PID。
- 共享机邻居争抢大（loadavg 常 ~40/192、邻居 NPU 卡可能 100%）→ run-to-run 方差大，所以要 n=2、
  挑清净窗口、同 prompt 同窗口配对；别被单次数值带偏。
- 纯调度/启动改动，数值必 bit 不变；本任务不需要重编 .so（C++ 未改）。

详细数据与机制见 doc/zh/dsv4_single_npu/workerpark_tune_handoff.md 的 Closeout + Follow-up 两节。
确认收益后回报：两臂 req1-3 实测、是否复现 −37%、以及你对"serving 默认开 SKIP_WARMUP=0"的建议。
```
