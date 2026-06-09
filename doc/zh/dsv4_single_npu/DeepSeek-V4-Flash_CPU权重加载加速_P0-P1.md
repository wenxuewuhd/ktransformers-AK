# DeepSeek-V4-Flash 单卡 NPU：CPU MoE 权重加载加速（P0 zero-copy + P1 并行重排）

> 适用：`tools/p27_launch_ds4flash_npu.sh` 拉起 DeepSeek-V4-Flash-W8A8（KT/LLAMAFILE 后端）单卡服务时，
> 启动阶段 CPU 逐层加载 MoE 专家权重很慢（即刷屏的 `TP MOE layer N` / `Llamafile TP splitting` 阶段）。
>
> 相关提交：`bb0eafe`（P0 zero-copy）、`2ea90d7`（P1 并行重排）。本文档配套可运行复现脚本
> `kt_load_checksum_driver.py`（同目录）。

---

## 1. 背景与现象

- 模型 **DeepSeek-V4-Flash-W8A8**：`num_hidden_layers=43`，`hidden_size=4096`，`moe_intermediate_size=2048`，
  `n_routed_experts=256`，`num_experts_per_tok=6`。
- 权重以**每层一个 Q8_0 GGUF**提供：`/workspace/models/cache/dsv4_layer{layer_idx}.gguf`，**每个 6.85GB，共 ~275GB**。
- 机器：**1.5TB 内存**（模型可整体常驻 page cache）、**192 核 Kunpeng-920（ARM aarch64）**、**8 个 NUMA 节点**
  （对应 KT 的 `tp_count`/`threadpool_count=8`）。
- 现象：每次拉起服务，CPU 逐层加载权重耗时数分钟。因为内存够大、模型已在 page cache，所以**“每次都慢”不是冷盘 IO，而是 CPU 拷贝开销**。

### 加载链路（代码路径）

```
sglang KTEPWrapperMethod.process_weights_after_loading
  └─ LlamafileMoEWrapper.load_weights()            kt-kernel/python/utils/llamafile.py:165
       ├─ GGUFLoader.get_undequanted_tensor_and_ggml_type()   kt-kernel/python/utils/loader.py:1055
       │     （读 GGUF mmap 切片 → 张量；旧实现在此处 .copy()）        ← B1
       ├─ MOE(moe_config) 构造（即 TP-split 打印）        kt-kernel/operators/moe-tp.hpp:46/86
       └─ moe.load_weights_task() → C++ load_weights()   kt-kernel/operators/llamafile/moe.hpp
             （per-expert memcpy 把权重重排进 NUMA-local buffer）       ← B2
```

### 基线计时（`KT_TIME_LOAD=1`，采样，非常稳定）

| 阶段 | 每层 | 占比 | 瓶颈 |
|---|---|---|---|
| read（GGUF mmap 读 + `np.copy()`，单线程） | ~7.5s | ~70% | **B1** |
| construct（C++ `MOE()` 构造） | ~0.02s | ~0% | — |
| load（C++ 8 线程 NUMA 重排拷贝） | ~3.2s | ~30% | **B2** |
| **合计** | **~11s/层** | | |

外推 43 层 ≈ **~475s（~7.9 分钟）**。

---

## 2. 瓶颈定位

- **B1（`loader.py:1114`，旧）**：`torch.from_numpy(np.frombuffer(data_bytes, dtype=np.uint8).copy())`
  对每层 6.85GB 的 mmap 数据做一次**单线程**全量拷贝到匿名内存；而紧接着 C++ `load_weights` 还要再从这块
  buffer memcpy 一次进 NUMA-local buffer。这份 `.copy()` 是**纯冗余**——返回值只被 C++ 当**只读源**消费。
- **B2（`moe.hpp` `LLAMA_MOE_TP::load_weights`）**：per-expert reshuffle 是**单线程串行循环**；TP-level
  `load_weights()` 用 `do_numa_job`（每 NUMA 一个 job）整体只有 **8-wide**，在 192 核机器上严重欠并行。

---

## 3. 修改方案与详细改动点

### P0 — zero-copy（去掉冗余 `.copy()`）  提交 `bb0eafe`

**文件**：`kt-kernel/python/utils/loader.py`，`GGUFLoader.get_undequanted_tensor_and_ggml_type`。

- 把返回张量从“mmap → 单线程 `.copy()` 到匿名内存”改为**直接 view mmap**（`np.frombuffer` 只读视图 +
  `torch.from_numpy`，不 `.copy()`）。C++ 的重排成为**唯一一次必要拷贝**。
- 安全性：mmap 由 `GGUFLoader.file_data_map` 持有；张量由 `LlamafileMoEWrapper.weights_to_keep` 钉到
  `cpu_infer.sync()` 返回；C++ 只读该源。
- 开关：`KT_ZEROCOPY_LOAD`（默认 1；`=0` 回退旧 `.copy()` 路径做 A/B）。
- 故意用 `torch.from_numpy` 而非 `torch.frombuffer`（后者在 `torch_npu` 的 `transfer_to_npu` 里会被重定向到 NPU）。

**效果**：read `~7.5s → 0.000s`，且因页已热、C++ 8 线程并行 fault，load 也降到 `~1.6s`。

### P1 — 并行重排（reshuffle 铺到 NUMA subpool）  提交 `2ea90d7`

**文件**：`kt-kernel/operators/llamafile/moe.hpp`，`LLAMA_MOE_TP::load_weights(int complete_intermediate_size, int offset)`。

- 把原 per-expert 串行 `for` 循环改写为：先用**显式 stride**表达每个专家 i 的 gate/up/down 源→目标字节范围
  （源张量按完整 `complete_intermediate_size` 排布，本 TP 只拥有 `[offset, offset+intermediate_size)` 块）：
  - gate/up：`dst_stride = intermediate_size*hidden*ts/bs`，`src_stride = complete_intermediate_size*hidden*ts/bs`；
  - down：每行 `dst_row = intermediate_size*ts/bs`、`src_row = complete_intermediate_size*ts/bs`，每专家 `hidden_size` 行。
- 各专家**写入不相交的目标区、读不相交的源区** → 天然可并行。用
  `config_.pool->get_subpool(tp_part_idx)->do_work_stealing_job(config.expert_num, copy_expert)`
  在该 NUMA subpool 的工作线程上并行执行——**与 `forward()` 在 `do_numa_job` 内嵌套 `do_work_stealing_job`
  的既有模式完全一致**（安全、无新并发风险）。
- 开关：`KT_PARALLEL_LOAD`（默认并行；`=0` 回退串行循环，A/B + 安全兜底）。

**效果**：load `~1.6s → ~1.04s/层`（串行 `KT_PARALLEL_LOAD=0` 实测 43 层累计 89s ÷ 并行 47s ≈ **1.9×**）。
受单 NUMA 内存带宽/线程数限制（`kt-cpuinfer=24` 分到 8 个 subpool ≈ 每 NUMA 3 线程），故为 ~2× 量级而非线性。

### 总体效果

| 阶段 | 每层加载 | 43 层累计 |
|---|---|---|
| 基线 | ~11s | ~475s（~7.9min） |
| + P0 | ~1.6s | ~70s |
| + P1 | **~1.04s** | **~47s** |

**启动 MoE 权重加载 ~7.9 分钟 → ~47 秒，约 10×。**

放到整体拉起里看（实测，sglang stderr 时间戳，单卡 910B；注意 `TP MOE layer` 走 stdout、
sglang 日志走 stderr，合并日志里顺序会被缓冲打乱，MoE 段以 Python 端直接计时为准）：

| 阶段（Load weight begin → server ready） | 旧 | 现（P0+P1） |
|---|---|---|
| 加载段（46 safetensors shard + 建模 + 43 层 MoE GGUF） | ~9 min | **~100s** |
| └ 其中 43 层 MoE GGUF | ~7.9 min | **~47s** |
| └ 其中 46 shard + 建模（未优化） | ~54s | ~54s |
| NPU graph capture | ~5–11s | ~5s |

> 即：加速后“加载段”的大头从 MoE GGUF（~474s）变成了 46 个 safetensors shard + 建模（~54s）。
> 后者走 sglang 自己的 loader、不在本次改动范围内；若还要再压加载，下一个目标是它，而非 MoE。

---

## 4. 精度对齐：验证方案与结果

### 为什么不能用推理输出做精度判据

本 stack（单卡 KT 子集，WIP）在 **temp=0 时推理输出本身就不确定**：同一服务、同一份已加载权重、同一 prompt
重复请求，会给出不同回答甚至复读乱码（`-`--`-`...、“13 的平方”循环等）。且这一现象在
**串行加载（`KT_PARALLEL_LOAD=0`，即改之前的加载逻辑）与并行加载、graph 与 eager（`--disable-cuda-graph`）下
同样出现** → 不确定性出在 **NPU 前向 / serving 路径（Ascend 浮点 reduction、graph 重放等）**，与加载方式无关。
因此**推理输出无法用于验证加载改动的精度**。

### 决定性方案：权重字节校验和 A/B（对推理噪声免疫）

P1 只改写入顺序、纯 `memcpy`、无浮点 / 无共享状态 / 无重叠写，对串行 **bit-identical 是构造可证的**。
为给出经验铁证，在 C++ `load_weights` 末尾加一段 **FNV-1a 校验和**（仅验证用，未入库），对加载完成的
`m_local_{gate,up,down}` 三个本地 buffer 取哈希，分别在 `KT_PARALLEL_LOAD=1`（并行）与 `=0`（串行）下加载，
逐 `(layer, tp)` 比对。

### 结果

跨 **7 层（0,1,2,5,17,30,42）× 8 TP = 56 组**校验和（gate/up/down 各一），**并行 vs 串行逐字节完全一致**。
→ P1 的并行重排加载出的权重与串行（改之前）路径**字节完全相同**，不改变任何权重比特。

---

## 5. 复现精度对齐的具体流程

> 注意：收尾清理 commit 会移除 `KT_PARALLEL_LOAD` 开关；下面的 checksum 是**临时插桩**（验证完即删）。
> 要复现，需临时把 checksum 段加回 `load_weights`，并临时恢复 `KT_PARALLEL_LOAD` 串行分支（若已被清理）。

### 5.1 临时给 C++ `load_weights` 加 checksum（插在 per-expert 拷贝 dispatch 之后、函数右括号之前）

```cpp
// 临时验证插桩：KT_LOAD_CHECKSUM=1 时对加载完成的本地 buffer 取 FNV-1a。
if (std::getenv("KT_LOAD_CHECKSUM")) {
  auto fnv1a = [](const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t b = 0; b < n; ++b) { h ^= p[b]; h *= 1099511628211ULL; }
    return h;
  };
  fprintf(stderr, "[KT_LOAD_CHECKSUM] layer=%d tp=%d gate=%016llx up=%016llx down=%016llx\n",
          config.layer_idx, tp_part_idx,
          (unsigned long long)fnv1a(m_local_gate_proj_, (size_t)config.expert_num * gate_dst_stride),
          (unsigned long long)fnv1a(m_local_up_proj_,   (size_t)config.expert_num * up_dst_stride),
          (unsigned long long)fnv1a(m_local_down_proj_, (size_t)config.expert_num * down_dst_stride));
}
```

（`gate_dst_stride`/`up_dst_stride`/`down_dst_stride` 即 P1 改动中定义的每专家本地字节步长。
若 `KT_PARALLEL_LOAD` 开关已被清理，需临时把串行分支加回：
`if (getenv("KT_PARALLEL_LOAD")&&getenv("KT_PARALLEL_LOAD")[0]=='0') for(...) copy_expert(i); else <并行>;`。）

### 5.2 编译 `.so`（在本仓库内，**勿**用 sibling `ktransformers-AK` 工作区编译）

```bash
cd <repo>/kt-kernel
CMAKE_BUILD_PARALLEL_LEVEL=64 MAX_JOBS=64 \
  /usr/local/python3.11.14/bin/python3 setup.py build_ext --inplace
# 产物自动 copy 到 kt-kernel/python/kt_kernel_ext*.so
```
（若本仓库 `third_party/{pybind11,llama.cpp}` 子模块为空，从可用来源只读拷入即可；fmt 头来自 torch 的 include。）

### 5.3 跑 standalone driver（不需要 NPU、不需要 sglang）

driver 见同目录 `kt_load_checksum_driver.py`。它只构造若干层 `LlamafileMoEWrapper` 并调用 `load_weights()`，
让 C++ 打印 checksum。

```bash
cd <repo>
export PYTHONPATH=$PWD/kt-kernel:$PWD/kt-kernel/python
export PYTHONDONTWRITEBYTECODE=1
PY=/usr/local/python3.11.14/bin/python3
LAYERS=0,1,2,5,17,30,42

KT_LOAD_CHECKSUM=1 KT_PARALLEL_LOAD=1 $PY doc/zh/dsv4_single_npu/kt_load_checksum_driver.py $LAYERS \
  2>cksum_parallel.err 1>/dev/null
KT_LOAD_CHECKSUM=1 KT_PARALLEL_LOAD=0 $PY doc/zh/dsv4_single_npu/kt_load_checksum_driver.py $LAYERS \
  2>cksum_serial.err   1>/dev/null

grep KT_LOAD_CHECKSUM cksum_parallel.err | sort > a.txt
grep KT_LOAD_CHECKSUM cksum_serial.err   | sort > b.txt
diff a.txt b.txt && echo ">>> IDENTICAL：并行 == 串行，逐字节一致 <<<"
```

**期望输出**：`diff` 为空，打印 `>>> IDENTICAL ... <<<`。每层 8 行（8 个 TP），N 层共 8N 行。

---

## 6. 开关与计时一览（收尾 commit 将移除）

| 名称 | 位置 | 作用 | 默认 |
|---|---|---|---|
| `KT_ZEROCOPY_LOAD` | `loader.py` | `=0` 回退 P0 之前的 `.copy()` 路径 | 1（zero-copy） |
| `KT_PARALLEL_LOAD` | `moe.hpp` | `=0` 回退 P1 之前的串行重排 | 并行 |
| `KT_TIME_LOAD` | `llamafile.py` | 打印每层 read/construct/load 计时 + 累计 | 1（开） |
| `KT_LOAD_CHECKSUM` | （临时插桩，未入库） | 打印本地 buffer FNV-1a 校验和 | 关 |

收尾：`KT_TIME_LOAD` 计时与 `KT_ZEROCOPY_LOAD`/`KT_PARALLEL_LOAD` 开关在加速收益拿到、精度对齐并 commit 后，
**单独一个 commit 移除**（让 zero-copy / 并行无条件生效），保留干净演进记录。本文档即为该清理前的存档。

### 环境备注（沙箱）

- 本仓库 `third_party/sglang` 子模块为空、`kt-kernel/python/` 无预编 `.so`；拉起服务时需 sglang 可 import +
  对应 `.so` 就位。
- ⚠️ `/workspace/code/ktransformers-AK`（无 `-cpu-load-accel`）是**他人活跃工作区**（NPU 7 在跑 graph
  profiling、有未提交改动）：**只读 import 其 sglang（配 `PYTHONDONTWRITEBYTECODE=1`），切勿对其 git
  checkout/reset/clean、勿在其内重编/覆盖 `.so`、勿用 NPU 7**。需干净子模块编译请用 worktree / 单独 clone。
- 每次拉起服务前先 `npu-smi info` 选空闲卡、并查端口占用（冲突则换 `PORT`）。
