# R1 msprof Graph Baseline 采集手册

> **目标**：DeepSeek-V4-Flash **R1**（hybrid N=32 prefix、graph on、inline profiler off、单并发 F2）的**进程外 msprof** 基线。  
> **约束**：不改 `kt-kernel/`、`third_party/sglang/` 业务逻辑；不用 eager+inline `SGLANG_NPU_PROFILE_*` 当生产基线；必须 graph on。

关联文档：[DeepSeek-V4-Flash_NPU_KT_Profiling_优化路线图.md](./DeepSeek-V4-Flash_NPU_KT_Profiling_优化路线图.md) §8.6。

---

## 1. 推荐方案：exec + sidecar

`msprof --application` 在 **GGUF 加载 + TBE 编译** 阶段采数易 **SIGSEGV**；`msprof --dynamic --pid` attach 已运行服务会报 `Dynamic profiling client connect fail`（启动时未开 dynamic 环境）。

**可行做法**：

1. `msprof --application` 包住 launch（进程树在 msprof 下）
2. **`--delay`** 跳过加载/编译阶段，只在 decode 窗口采数
3. **sidecar** 等 `server.log` 出现 `fired up` 后跑 F2（与 launch 并行、两终端或 `p27_msprof_graph_baseline_exec.sh` 一键）

---

## 2. 一键采集（仓库根目录）

```bash
cd /workspace/code/ktransformer/ktransformers-AK

NPU_DEVICE_ID=2 PORT=8001 MSPROF_DELAY=900 \
  bash tools/p27_msprof_graph_baseline_exec.sh
```

产物目录：`tools/msprof_dbg/r1_<YYYYMMDD_HHMMSS>/`

---

## 3. 关键参数：`--delay=900`（推荐 900，勿用 4200）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `MSPROF_DELAY` / `--delay` | **900**（15 min） | 从 **msprof 进程启动**起算，前 N 秒**不采数** |
| 曾用错误值 | 4200（70 min） | 加载 ~9 min + graph ~36 s 后即 ready，F2 须傻等 70 min 才进采数窗 |
| 加载参考 | ~50 min GGUF + ~36 s graph capture | 仅作经验值；以 `server.log` 中 `fired up` 为准 |

**时间轴（`delay=900`，msprof 13:29 启动示例）**：

```text
13:29  msprof 启动，delay 计时开始
13:38  服务 fired up（可 curl / 跑 F2，但 msprof 仍不记）
13:44  delay 结束，开始采 NPU device 数据
13:44+ sidecar 跑 F2 → decode kernel 进入 profile
```

**原则**：`delay` 略大于「msprof 启动 → graph capture 完成」，不必覆盖整段 GGUF 加载后的额外等待。

**sidecar 行为（`p27_msprof_sidecar_f2.sh`）**：先等 `fired up`，再 **sleep 至 `MSPROF_START + delay`**，然后才跑 F2。仅 ready 不够——若 F2 早于 delay 结束，msprof **不会创建 `PROF_*` 目录**（2026-05-24 `r1_20260524_154827` 即此情况：ready ~8.5 min，F2 ~10 min，delay 900 采数从 15 min 才开始）。

计算采数开始时刻：

```bash
python3 - <<'PY'
from datetime import datetime, timedelta
start = datetime(2026, 5, 24, 13, 29, 0)  # 改成你的 msprof 启动时间
delay = 900
print("预计开始采数:", start + timedelta(seconds=delay))
PY
```

---

## 4. msprof 命令形态

```bash
MSPROF_BIN=/usr/local/Ascend/cann-8.5.0/bin/msprof

"$MSPROF_BIN" \
  --application="$REPO/tools/p27_msprof_launch_exec.sh" \
  --output="$REPO/tools/msprof_dbg/r1_<tag>" \
  --delay=900 \
  --aic-mode=task-based \
  --task-time=on \
  --runtime-api=on \
  --aicpu=on \
  --hccl=off
```

sidecar（另一终端，或 exec 脚本自动调用）：

```bash
bash tools/p27_msprof_sidecar_f2.sh \
  "$OUT/server.log" \
  "$OUT/f2_throughput.log"
```

F2  workload：`tools/p27_curl_f2_prompts.sh`（四 prompt，稳态看 prompt 2+4）。

---

## 5. 路径与工作目录（必读）

**必须从仓库根 `ktransformers-AK/` 启动**，或使用脚本内 `$REPO` 绝对路径。

| 错误做法 | 后果 |
|----------|------|
| 在 `tools/` 子目录手动跑，`OUT=tools/msprof_dbg/...` | `server.log` 写到 `tools/tools/msprof_dbg/...`，sidecar 监听 `$REPO/tools/msprof_dbg/...` → **永远等不到 fired up** |
| `msprof --output=tools/msprof_dbg/...` 相对路径 + cwd=`tools/` | PROF 产物路径混乱 |

脚本已修复：`p27_msprof_sidecar_f2.sh` 会解析 log 绝对路径；`p27_msprof_launch_exec.sh` 将 `MSPROF_WORKLOAD_LOG` 归一到 `$REPO/...`；`p27_msprof_graph_baseline_exec.sh` 使用 `$REPO/tools/msprof_dbg/...` 绝对 `--output`。

---

## 6. F2 完成后停服与 export

1. F2 完成后 **sleep 30**（给 device 缓冲刷盘）
2. **SIGTERM 优雅退出** launch_server，等待 60–90 s，**避免立刻 `pkill -9` / exit 137**
3. `msprof --application` 退出时会**自动 export**；**勿在 export 进行中再跑** `msprof --export=on`（冲突）
4. 终端回到提示符后验收：

```bash
find "$OUT" -name 'kernel_details.csv' -o -name 'trace_view.json'
python3 tools/p27_parse_msprof_baseline.py \
  --run-dir "$OUT" --label R1 --f2-log "$OUT/f2_throughput.log"
```

若自动 export 后仍无 `kernel_details.csv`，可对含 F2 的 `PROF_*` 目录补一次：

```bash
$MSPROF_BIN --export=on --output="$OUT/PROF_xxx" --type=text --summary-format=csv
```

---

## 7. 失败模式与规避

| 现象 | 原因 | 规避 |
|------|------|------|
| launch 阶段 SIGSEGV | msprof 无 delay 包住加载/编译 | `--delay=900` 或更大（不超过必要值） |
| sidecar 一直 waiting | log 路径错（见 §5） | 仓库根启动；或 grep 实际 `server.log` 路径 |
| F2 有吞吐、无 `PROF_*` | F2 早于 delay 结束，采数从未开始 | sidecar 已修复：ready 后等到 `MSPROF_START+delay` 再 F2 |
| `device_2/data` 空、`Collect data failed` | **exit 137** 强杀 / 双 PROF 会话 rollover | 优雅 SIGTERM + 等 90 s；F2 后 sleep 30 |
| `no timeline/summary data` | 采数窗内无 NPU task 或 device 未落盘 | 确认 decode 在 delay 后；查 `device_2/data` 非空 |
| dynamic attach 失败 | 服务启动未在 msprof 下 | 只用 `--application`，勿 attach 已起服务 |
| OOM exit 137（双实例） | 8001 上旧 launch 未清 | 采集前 `pkill -f 'sglang.launch_server.*--port 8001'` |

---

## 8. 验收标准

- [ ] `PROF_*/mindstudio_profiler_output/` 或等价目录含 **`kernel_details.csv` + `trace_view.json`**
- [ ] F2 稳态 **~3.5–3.8 tok/s**（prompt 2+4；server 日志 `gen throughput` ~3.6–4.0 亦可）
- [ ] decode 日志 **`npu graph: True`**
- [ ] 无 SIGBUS / ERR 107027
- [ ] 解析脚本输出 Wait Time 汇总（供路线图 §7 R1 行）

---

## 9. 实测记录

### 9.1 `r1_manual_delay_20260524_132948`（delay=4200，手动）

| 项 | 结果 |
|----|------|
| F2 | ✅ ~3.8–3.9 tok/s |
| kernel | ❌ device_2 空（exit 137） |

### 9.2 `r1_20260524_154827`（delay=900，exec 脚本）

| 项 | 结果 |
|----|------|
| msprof 启动 | 15:48:27 |
| fired up | 15:56:55（~8.5 min） |
| F2 | ✅ steady **3.857 tok/s** |
| PROF | ❌ **无 `PROF_*` 目录** — F2 ~15:59，delay 采数从 16:03:27 才开始 |
| 修复 | sidecar 在 ready 后增加「等到 `MSPROF_START+delay`」 |

### 9.3 `r1_20260524_165930`（delay=900 + sidecar 等 delay，exec 脚本）

| 项 | 结果 |
|----|------|
| sidecar | ✅ `wait 399s before F2`，F2 落在采数窗内（17:14:40+） |
| F2 | ✅ steady **3.869 tok/s**，`npu graph: True` |
| PROF | ✅ 两个 `PROF_*`；host `api_statistic.csv` 有 acl 数据 |
| device_2 | ❌ 仍空 / 仅 `lpmFreqConv`；**无 `kernel_details.csv`** |
| 停服 | ⚠️ 仍 **exit 137**（msprof 强杀子进程） |
| 参数 | ⚠️ 当次 `--runtime-api=off --aicpu=off`（已改为 on） |

### 9.4 待重采（第 4 轮）

脚本已更新：`--runtime-api=on --aicpu=on`、F2 后 sleep 60、去掉 `pkill -f`、SIGTERM 等 180s 再 SIGKILL。

---

## 10. 工具索引

| 文件 | 用途 |
|------|------|
| `tools/p27_msprof_graph_baseline_exec.sh` | 一键 exec + sidecar + export + 解析 |
| `tools/p27_msprof_launch_exec.sh` | msprof `--application` 入口 |
| `tools/p27_msprof_sidecar_f2.sh` | 等 ready 后跑 F2 |
| `tools/p27_curl_f2_prompts.sh` | F2 四 prompt |
| `tools/p27_parse_msprof_baseline.py` | 解析 kernel / F2 |
| `tools/p27_launch_ds4flash_npu.sh` | R1 launch（graph on，profiler off） |

环境变量：`MSPROF_DELAY`（默认 900）、`NPU_DEVICE_ID`（默认 2）、`PORT`（默认 8001）。
