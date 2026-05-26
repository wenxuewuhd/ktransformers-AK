# P1.3 阻塞记录

## 目标

`python tools/attn_microbench/scripts/dump_production_indexer_call.py results/production_indexer_dump.json`

## 尝试

| 次 | timeout | 结果 |
|----|---------|------|
| 1 | 900s | `/health` 未就绪；KT MoE 仍在加载 |
| 2 | 1800s | 加载至 MOE layer ~38 时被脚本 `terminate()`；日志见 `results/p1_3_server.log` 末尾 `main process disappeared` |

## 根因

DeepSeek-V4-Flash W8A8 + KT LLAMAFILE 单卡 **冷启动 >30 min**，超过脚本 health 等待上限；**非 monkey-patch 逻辑错误**（hook 未有机会触发）。

## 下一步（需人工）

1. 先用 `bash tools/p27_launch_ds4flash_npu_num_expert_0.sh` 预热至 `/health` 200。
2. **重启** server 并注入 hook：
   ```bash
   export ATTN_DUMP_INDEXER_PATH=$PWD/results/production_indexer_dump.json
   export PYTHONSTARTUP=$PWD/scripts/dump_hook_startup.py
   # 再 launch_server …
   ```
3. 或把 `dump_production_indexer_call.py --timeout` 提高到 **≥3600** 后重跑。

**未生成** `production_indexer_dump.json` — P1.4 正式 diff 未完成。
