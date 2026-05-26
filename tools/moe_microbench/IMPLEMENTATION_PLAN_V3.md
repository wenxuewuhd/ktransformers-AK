# NPU MoE Microbench — 实施计划 v3

> **v2 → v3 更新（2026-05-26，二次评审反馈 N1-N9）**
> - **N1（必修）**: `run_loop_quant_matmul` 内 `.item()` 是 device→host 同步会污染 S2 对照；改为 warmup 阶段一次性 `.cpu().tolist()`
> - **N2（必修）**: 重新定义 `N`，从 `num_tokens × top_k` 改为 **`n_active × tokens_per_expert`**（NPU 上实际看到的 token 数），消除 sweep `n_active ∈ {1..6}` 与 `sum(group_list)==N` 断言的冲突；yaml 显式引入 `tokens_per_expert` 字段
> - **N3（必修，D4 前）**: `profile.py` 改 `schedule(wait/warmup/active) + prof.step()` 模式，否则单次 fn() 产空 trace
> - **N4**: yaml 拆 `hbm_peak_tb_s` (1.6) + `hbm_effective_tb_s` (1.0)；report 同时报 `util_vs_peak` 和 `util_vs_achievable`；D2 跑 `torch.empty.clone()` 校准 effective
> - **N5**: `act_quant` 默认是 post-dispatch 形态 `[N, H]`；bench_act_quant 加 `--pre-dispatch` flag 测 `[num_tokens, H]` 对照
> - **N6（必修）**: `roofline.py` 加 `shared_weight_bytes()`，否则 report shared 行 util 列空
> - **N7**: gate/up chunk 顺序加进 D2 smoke 检查清单（取真实权重对照 `silu(half[:I])*half[I:]` vs 反之）
> - **N8**: README 加 FP4 roofline 定量对比（76 MB / 48 μs vs 144 MB / 90 μs）
> - **N9**: `print_npu_smi` 顺手打印 `os.environ.get('ASCEND_RT_VISIBLE_DEVICES')`
>
> **v1 → v2 更新留存**
> - **A1**: `moe_intermediate_size` 128 → **实测 2048**
> - **A2**: 对齐 ckpt W8A8，不切 FP4（README 显式声明）
> - **A3**: silu_mul fused/unfused 双路径
> - **A4 + S8**: scale shape + A1 防回归 assert
> - **A5**: sweep `n_active ∈ {1..6}`
> - **A6/A7**: prefill / multi-token 延后到 D5
> - **A8**: shared‖routed 并行警示
> - **S1/S2/S3/S4/S5**: roofline + grouped vs loop + repeat=1000 + dispatch_overhead + profile dump
> - **S6**: README "已知偏离"
> - **S7**: 不抽 `tools/_common/`，改用 NOTE 同步

---

## 0. 范围与默认场景

### 0.1 论文 vs 实际 ckpt vs 本计划（基线对齐表）

| 字段 | 论文 (Flash, §4.2.1 / §5.2.1) | 本地 ckpt `DSv4-Flash-W8A8` 实测 | 本计划默认 | 备注 |
|------|------|------|------|------|
| `hidden_size` | 4096 | 4096 | 4096 | ✅ |
| `moe_intermediate_size` | 2048 | 2048 | **2048** | A1 修正 |
| `n_routed_experts` | 256 | 256 | 256 | ✅ |
| `num_experts_per_tok` | 6 | 6 | 6 | ✅ |
| `n_shared_experts` | 1 | 1 | 1 | ✅ |
| `shared expert 中间维` | 2048 | 2048 (= n_shared × moe_inter) | 2048 | ✅ |
| `num_hidden_layers` | — | 43 (全 MoE, `first_k_dense_replace=0`, `moe_layer_freq=1`) | — | 仅作信息 |
| **routed expert 权重精度** | MXFP4 (FP4) | **int8 per-channel** | **int8** | 对齐 ckpt, 不对齐论文 |
| **激活精度** | FP8 cast | **int8 per-token dynamic** | **int8** | 对齐 ckpt |
| **量化格式** | FP4 + FP8 mixed | `int-quantized` (compressed-tensors) | int8 W8A8 | 对齐 ckpt |
| 路由 | 前 3 层 Hash + 后续 expert | 实际 routing 与 per-expert 计算无关 | 不测 routing | — |

**关键判断**：用户场景是「CPU offload MoE 后所有热专家命中 NPU」，使用的是 **DSv4-Flash-W8A8 ckpt 的实际部署路径**，所以 bench 走 W8A8。README 必须明示这与论文 FP4×FP8 的差异（理论 HBM 流量约 2×）。

### 0.2 测试覆盖

**关键定义（N2 修正）**：
- `n_active_experts` = 落在 NPU 的 active expert 数（CPU offload 命中数，∈ [1, top_k]）
- `tokens_per_expert` = 每 expert 平均处理 token 数（默认 1，decode bs=1 + 命中均匀）
- **`N = n_active_experts × tokens_per_expert`** = NPU 上 grouped GEMM 实际看到的 token 数
- `group_list.sum() == N` ✅ 自洽

| 段 | 算子 | 默认 shape (decode, n_active=6, tokens/exp=1 → N=6, I=2048, H=4096) |
|----|------|---------------------------------------------------------------------|
| **act_quant** (post-dispatch, N5 默认) | `torch_npu.npu_dynamic_quant` (per-token) | bf16 `[N, 4096]` → int8 `[N, 4096]` + fp32 scale `[N]` |
| **act_quant** (pre-dispatch, N5 可选) | 同上 | bf16 `[num_tokens, 4096]` → int8 `[num_tokens, 4096]` (默认 num_tokens=1) |
| **gemm_up** | `torch_npu.npu_grouped_matmul` (W8A8) | int8 `[N, 4096]` × int8 `[E_act, 4096, 4096]` → bf16 `[N, 4096]` |
| **silu_mul** (unfused / fused, A3) | SiLU(gate) × up + act_quant_mid | bf16 `[N, 4096]` → bf16 `[N, 2048]` → int8 `[N, 2048]` |
| **gemm_down** | `npu_grouped_matmul` (W8A8) | int8 `[N, 2048]` × int8 `[E_act, 2048, 4096]` → bf16 `[N, 4096]` |
| **routed_full** | 上 4 段串联 | 端到端 (不含 dispatch/combine) |
| **shared_expert** | dense W8A8 MLP | bf16 `[num_tokens, 4096]` → `[num_tokens, 4096]`，中间维 2048 |
| **grouped_vs_loop** (S2) | 1×grouped vs N×quant_matmul 对比 | 同 gemm_up shape |

> ⚠ **act_quant 位置约定（N5）**：默认按「已 dispatch 完成、token 已展开到 N 份」测量，是上限。
> 若生产采用 pre-dispatch quant（CPU 只把 int8 + scale 拷到 NPU，再在 NPU 上 expand），
> 实际成本 ≈ post-dispatch 测量 / `n_active_experts`。`bench_act_quant --pre-dispatch` 同时跑两路径供对照。

**不测**：`npu_moe_gating_top_k`（gating）/ `npu_moe_init_routing` / `npu_moe_finalize_routing`（dispatch/combine）/ KT CPU MoE / all-to-all 通信。

### 0.3 默认参数 + sweep（N2 重定义 N 后）

| 参数 | 默认 | sweep | 说明 |
|------|------|-------|------|
| `num_tokens` | 1 | 1（主线，decode） | 用户 bs 中拿到了多少 token；KT 上文用 |
| `top_k` | 6 | 6（主线） | DSv4-Flash `num_experts_per_tok` |
| `n_active_experts` | 6 | **{1,2,3,4,5,6}** (A5) | NPU 上落了几个 active expert（CPU offload 热专家命中数） |
| `tokens_per_expert` | 1 | 1（主线） | 每个 active expert 处理几个 token；decode + 均匀命中 = 1 |
| **`N = n_active × tpe`** | **6** | sweep 时 `N ∈ {1..6}` | NPU grouped GEMM 实际 token 数；`group_list.sum() == N` |
| `group_list` | `[1]*n_active` | 同 | shape `(n_active,)`，sum = N |
| `dtype` | w8a8 | w8a8（主线），bf16（可选 baseline） | |
| `warmup` | 30 | — | |
| `repeat` | **1000** (S3) | `--quick-mode → 100` | |

**sweep 表（n_active 全覆盖）的 group_list 与 N**：

| n_active | tokens_per_expert | N | 默认 group_list | 物理意义 |
|---------:|------------------:|--:|-----------------|----------|
| 1 | 1 | 1 | `[1]` | 1/6 热命中：只 1 个 expert 在 NPU |
| 2 | 1 | 2 | `[1,1]` | 2/6 命中 |
| 3 | 1 | 3 | `[1,1,1]` | 3/6 命中 |
| 4 | 1 | 4 | `[1,1,1,1]` | 4/6 命中 |
| 5 | 1 | 5 | `[1,1,1,1,1]` | 5/6 命中 |
| 6 | 1 | 6 | `[1,1,1,1,1,1]` | 6/6 命中（100% 热） |

扩展（需手动配 `num_tokens`，D5）：`n_active ∈ {8,16,32}` 对应 multi-token / 不均衡 / batch > 1。

### 0.4 单条计算量参考（默认形状）

| 指标 | gemm_up | gemm_down | shared (gate_up+down) | 总 (routed) |
|------|---------|-----------|----------------------|-------------|
| weight bytes (n_active=6, W8) | 6 × 4096 × 4096 × 1B = **96 MB** | 6 × 2048 × 4096 × 1B = **48 MB** | 16 + 8 = **24 MB** (N6) | **144 MB** routed |
| GFLOPs (M=6, K, 2N) | 6×4096×4096×2 / 1e9 = 0.20 GF | 6×2048×4096×2 / 1e9 = 0.10 GF | M=1 同形 / 6 | 0.30 GF |
| roofline @ HBM peak 1.6 TB/s | 60 μs | 30 μs | 15 μs (N6) | **90 μs** |
| roofline @ HBM effective 1.0 TB/s (N4) | 96 μs | 48 μs | 24 μs | **144 μs** |

`report.py` 同时报：
- `util_vs_peak = actual / lb_peak`（< 1 表示带宽估高/算子未打满）
- `util_vs_achievable = actual / lb_effective`（最贴近真实优化空间；> 0.75 算打满）

---

## 1. 工作空间结构

```
tools/moe_microbench/
├── README.md                          (NPU 选卡 SOP + 用法 + 已知局限 §S6)
├── IMPLEMENTATION_PLAN.md             (本文)
├── env.sh                             (CANN + 性能调度)
├── run_all.sh                         (一键 + --profile + --quick-mode + n_active sweep)
├── config/
│   └── dsv4_flash_moe.yaml            (实测对齐)
└── moe_bench/                         (Python 包)
    ├── __init__.py
    ├── config.py                      (MoEConfig + sweep / roofline 配置加载)
    ├── init_npu.py                    (NPU select + npu-smi + version log)
    ├── synthetic.py                   (权重 + activation + group_list + 全套 assert §3.4)
    ├── ops_runner.py                  (quant / grouped_matmul / silu_mul + 自动探测 fused / shared mlp)
    ├── timing.py                      (Event + host wall; 顶部注释: 与 attn_bench/timing.py 同步)
    ├── sanity.py                      (NaN + 量级)
    ├── profile.py                     (S5: torch_npu.profiler 包装, dump 1 次到 OUT/profile/<seg>/)
    ├── roofline.py                    (S1: weight_bytes / HBM → lower_bound; utilization)
    ├── bench_act_quant.py
    ├── bench_gemm_up.py
    ├── bench_silu_mul.py              (含 --fused / --no-fused 探测)
    ├── bench_gemm_down.py
    ├── bench_routed_full.py
    ├── bench_shared_expert.py
    ├── bench_grouped_vs_loop.py       (S2: grouped vs N×独立 quant_matmul 对比)
    └── report.py                      (roofline + dispatch_overhead + shared‖routed 并行警示)
```

---

## 2. NPU 选卡 SOP（落 README，此处摘要）

```bash
npu-smi info                                # 看每张卡 Health / Memory-Usage / AICore-Util
export ASCEND_RT_VISIBLE_DEVICES=2          # 选空卡（物理 2 号）
python -c "import torch_npu, torch; print(torch.npu.current_device())"  # 进程内仍是 npu:0
```

`env.sh` **不自动选卡**；`init_npu.print_npu_smi()` 启动时打印简表提示。

---

## 3. 模块规格

### 3.1 `config/dsv4_flash_moe.yaml`（实测对齐版）

详见已落盘 yaml，要点：
- `moe_intermediate_size: 2048`（不是 128）
- `weight_scale_strategy: channel`, `act_scale_strategy: token`
- `sweep.n_active_experts: [1,2,3,4,5,6]`
- `bench.repeat: 1000`
- `roofline.hbm_bandwidth_tb_s: 1.6`（D2 实测后改）

### 3.2 `moe_bench/config.py`

新增字段（v3）：
- `tokens_per_expert: int = 1`（N2）
- `sweep_n_active: list[int]`
- `roofline_hbm_peak_tb_s: float`（N4，默认 1.6）
- `roofline_hbm_effective_tb_s: float`（N4，默认 1.0；D2 校准）
- `roofline_bytes_per_weight: int`（默认 1，W8）

`N` property 重定义（N2）：
```python
@property
def N(self) -> int:
    """NPU 上 grouped GEMM 实际看到的 token 总数 = n_active * tokens_per_expert"""
    return self.n_active_experts * self.tokens_per_expert
```

### 3.3 `moe_bench/init_npu.py`

`setup_pythonpath` / `init_custom_ops` / `require_npu` / `log_versions` / `print_npu_smi`，已落盘。

**N9 增强**：`print_npu_smi` 同时打印 `os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')`，
方便容器环境下用户判断是否已被外部 mount 限定（如单卡 davinci2 容器）。

### 3.4 `moe_bench/synthetic.py` — 全套断言（A4 + S8 + N2）

```python
def assert_shapes(cfg, t):
    N = cfg.N                                   # n_active * tokens_per_expert (N2)
    H = cfg.hidden_size; I = cfg.moe_intermediate_size
    S = cfg.shared_intermediate_size; E = cfg.n_active_experts

    # A1 + ckpt 防回归
    assert cfg.moe_intermediate_size == 2048, "DSv4-Flash 实测 moe_intermediate=2048; 不要回退"
    assert cfg.n_routed_experts == 256
    assert cfg.num_experts_per_tok == 6
    assert cfg.n_shared_experts == 1

    # N2: N 自洽
    assert N == cfg.n_active_experts * cfg.tokens_per_expert

    # Activation (post-dispatch 形态; N5 pre-dispatch 时另算 [num_tokens, H])
    assert t.x_bf16.shape == (N, H)
    assert t.x_int8.shape == (N, H) and t.x_int8.dtype == torch.int8
    assert t.x_scale.shape == (N,) and t.x_scale.dtype == torch.float32  # per-token (token strategy)

    # Routed weights
    assert t.w_gate_up.shape == (E, H, 2*I) and t.w_gate_up.dtype == torch.int8
    assert t.w_gate_up_scale.shape == (E, 2*I)                            # per-channel (A4)
    assert t.w_down.shape == (E, I, H) and t.w_down.dtype == torch.int8
    assert t.w_down_scale.shape == (E, H)                                 # per-channel (A4)

    # group_list 守恒 (N2 修正)
    assert t.group_list.shape == (E,)
    assert (t.group_list >= 0).all()
    assert int(t.group_list.sum().item()) == N

    # Shared (dense)
    assert t.w_shared_gate_up.shape == (H, 2*S)
    assert t.w_shared_gate_up_scale.shape == (2*S,)
    assert t.w_shared_down.shape == (S, H)
    assert t.w_shared_down_scale.shape == (H,)
```

默认 `_make_group_list()`：
```python
# 均匀分配: 每 active expert 拿 tokens_per_expert 个 token
counts = [cfg.tokens_per_expert] * cfg.n_active_experts
group_list = torch.tensor(counts, dtype=torch.int64, device=device)
# 校验: sum == N == n_active * tpe ✅
```

### 3.5 `moe_bench/ops_runner.py`

5 段算子封装 + `run_routed_full` + `run_shared_expert` + `run_loop_quant_matmul`（S2 对照）+ `make_silu_mul`（A3 fused 探测）。

#### 3.5.1 `make_silu_mul` (A3)

```python
def make_silu_mul(cfg):
    """返回 (silu_mul_fn, mode) where mode in {'fused', 'unfused'}."""
    fused = cfg.ops.get("swiglu_fused")
    if fused:
        try:
            import importlib
            mod, name = fused.rsplit(".", 1)
            fn = getattr(importlib.import_module(mod), name)
            return fn, "fused"
        except Exception:
            pass
    def unfused(gate_up_bf16):
        # ⚠ N7: gate/up chunk 顺序 D2 smoke 时需要拿真实权重核对
        # DSv4 实测顺序：(D2 smoke 后回填) — 当前默认 [gate | up]，即 chunk(2,-1) 返回 (gate, up)
        gate, up = gate_up_bf16.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up
    return unfused, "unfused"
```

#### 3.5.2 `run_loop_quant_matmul` (S2 + **N1 修正**)

⚠ **N1 关键修正**：原 v2 写法 `int(starts[e].item())` 是 device→host 同步，每次 launch 都强制等前一个 kernel 完成。
这会让 loop 路径系统性变慢，得到错误的"grouped 永远更快"结论。

```python
def make_loop_runner(t, cfg):  # S2 — 工厂模式: 在 warmup 之外预算 cumsum
    """返回一个无同步的 loop runner; 调用者只在 timing 区外 setup 一次."""
    import itertools

    # 这一次 .cpu().tolist() 是 device→host 同步，但只发生在 setup 阶段
    gl_cpu = t.group_list.cpu().tolist()
    bounds = list(itertools.accumulate(gl_cpu))   # cumsum 全 CPU 算

    def loop_fn():
        import torch_npu
        outs = []
        prev = 0
        for e, end in enumerate(bounds):
            if end == prev:
                continue
            x_e = t.x_int8[prev:end]
            s_e = t.x_scale[prev:end]
            outs.append(torch_npu.npu_quant_matmul(
                x_e, t.w_gate_up[e], t.w_gate_up_scale[e],
                per_token_scale=s_e, output_dtype=torch.bfloat16))
            prev = end
        return torch.cat(outs, dim=0) if outs else None

    return loop_fn
```

`bench_grouped_vs_loop.py` 使用：
```python
loop_fn = make_loop_runner(t, cfg)            # setup 一次
bench_op(loop_fn, cfg.warmup, cfg.repeat)     # timing 区内零同步
```

**重要 TBD（D2 smoke 时校验）**：
- `npu_grouped_matmul` 参数名 / `group_type` / `split_item` 在 CANN 版本间可能不同
- `group_list` 是 cumsum 还是 per-expert count
- 是否有 `swiglu` 或 `swiglu_quant` 融合算子（A3 探测落空也无所谓，只跑 unfused）
- **gate/up chunk 顺序（N7）**：D2 拿真实 ckpt 一层权重，分别跑 `silu(half[:I])*half[I:]` 和 `silu(half[I:])*half[:I]`，对比哪个更接近真实 reference 输出

### 3.6 `moe_bench/timing.py`

复用 `attn_bench/timing.py` 思路。**S7 决策**：不抽 `tools/_common/`，但两边文件顶部都加：

```python
# NOTE: keep in sync with tools/attn_microbench/attn_bench/timing.py
# Changes here should be mirrored manually (no shared module by design).
```

返回值字段同 attn 版（device mean/p50/p95/p99/max/std + host mean + n）。

### 3.7 `moe_bench/sanity.py`

`sanity_check(name, o)` 返回 NaN flag + mean/std/abs_max。

### 3.8 `moe_bench/profile.py` (S5 + **N3 修正**)

⚠ **N3 关键修正**：原 v2 写法 `with profile: fn()` 单次调用没有 `prof.step()`，
`tensorboard_trace_handler` 写盘是基于 step 边界的，单次 fn 可能产出空/极短 trace。

```python
def maybe_profile(out_dir, name, fn, *, n_steps: int = 8, warmup: int = 2):
    """跑 (warmup + n_steps) 次, 每次 prof.step(); dump 到 OUT_DIR/profile/<name>/."""
    import torch_npu
    target = Path(out_dir) / "profile" / name
    target.mkdir(parents=True, exist_ok=True)
    sched = torch_npu.profiler.schedule(
        wait=0, warmup=warmup, active=n_steps, repeat=1)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        schedule=sched,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(target)),
        record_shapes=True,
    ) as prof:
        for _ in range(warmup + n_steps):
            fn()
            prof.step()
        torch_npu.npu.synchronize()
```

`run_all.sh --profile` 时，每段 bench 跑一次（不与 timing 同时运行）。CANN 版本不同时 schedule/handler API 微差，D2 smoke 时若失败自动 fallback 到无 schedule 的 `with profile: fn()*N` 形态。

### 3.9 `moe_bench/roofline.py` (S1 + **N4 双带宽 + N6 shared**)

```python
def routed_weight_bytes(cfg):
    """grouped GEMM: up + down 总 weight bytes."""
    H, I, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_active_experts
    bpw = cfg.roofline_bytes_per_weight
    up   = E * H * (2 * I) * bpw
    down = E * I * H       * bpw
    return up, down

def shared_weight_bytes(cfg):  # N6 新增
    """dense shared expert MLP: gate_up + down 总 weight bytes."""
    H, S = cfg.hidden_size, cfg.shared_intermediate_size
    bpw = cfg.roofline_bytes_per_weight
    gate_up = H * (2 * S) * bpw
    down    = S * H       * bpw
    return gate_up, down

def lower_bound_us(weight_bytes, hbm_tb_s):
    return weight_bytes / (hbm_tb_s * 1e12) * 1e6

def utilizations(actual_us, weight_bytes, cfg):  # N4: 双带宽
    lb_peak = lower_bound_us(weight_bytes, cfg.roofline_hbm_peak_tb_s)
    lb_eff  = lower_bound_us(weight_bytes, cfg.roofline_hbm_effective_tb_s)
    return {
        "lb_peak_us":      lb_peak,
        "lb_effective_us": lb_eff,
        "util_vs_peak":        actual_us / lb_peak if lb_peak  else None,
        "util_vs_achievable":  actual_us / lb_eff  if lb_eff   else None,
    }

def measure_hbm_effective_tb_s(device, size_mb: int = 256) -> float:
    """D2 校准用: torch.empty(size).clone() 大块拷贝实测可达 HBM 带宽."""
    import torch, torch_npu, time
    n = size_mb * 1024 * 1024 // 2     # bf16 elem
    x = torch.empty(n, dtype=torch.bfloat16, device=device)
    torch.npu.synchronize()
    for _ in range(5): _ = x.clone()       # warmup
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): _ = x.clone()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / 20
    bytes_moved = 2 * size_mb * 1024 * 1024     # read + write
    return bytes_moved / dt / 1e12               # TB/s
```

D2 smoke 时调一次 `measure_hbm_effective_tb_s`，把数字回填 yaml `roofline.hbm_effective_tb_s`。

### 3.10 CLI: `bench_*.py`

公共 argparse：
```
--config / --num-tokens / --top-k / --n-active-experts
--tokens-per-expert (N2)
--warmup / --repeat / --quick-mode
--out / --sanity / --dry-run
--fused / --no-fused          (仅 bench_silu_mul, A3)
--pre-dispatch                 (仅 bench_act_quant, N5: 同时跑 [num_tokens, H] 对照)
```

输出 JSON 示例（`bench_routed_full`）：
```json
{
  "kind": "routed_full",
  "cfg": {"N":6,"H":4096,"I":2048,"E_act":6,"top_k":6,"num_tokens":1},
  "timing": {"device_mean_us":..., "host_mean_us":..., "p95":..., "p99":..., "std":...},
  "derived": {
    "dispatch_overhead_us": host_mean - device_mean    // S4
  }
}
```

### 3.11 `moe_bench/report.py` (S1 + S4 + N4 + N6)

`summary_decode.md` 列：

```markdown
| segment             | dev_mean_us | p99 | host_mean_us | dispatch_overhead_us (S4) | lb_peak | util_peak | lb_eff (N4) | util_eff |
|---------------------|-------------|-----|--------------|---------------------------|---------|-----------|-------------|----------|
| act_quant (post)    | ...         | ... | ...          | host - device             | (n/a)   | (n/a)     | (n/a)       | (n/a)    |
| act_quant (pre, N5) | ...         | ... | ...          | ...                       | (n/a)   | (n/a)     | (n/a)       | (n/a)    |
| gemm_up             | ...         | ... | ...          | ...                       | 60 μs   | x.xx      | 96 μs       | x.xx     |
| silu_mul (unfused)  | ...         | ... | ...          | ...                       | (n/a)   |           |             |          |
| silu_mul (fused, A3)| ...         | ... | ...          | ...                       | (n/a)   |           |             |          |
| gemm_down           | ...         | ... | ...          | ...                       | 30 μs   | x.xx      | 48 μs       | x.xx     |
| routed_full         | ...         | ... | ...          | ...                       | 90 μs   | x.xx      | 144 μs      | x.xx     |
| shared_expert (N6)  | ...         | ... | ...          | ...                       | 15 μs   | x.xx      | 24 μs       | x.xx     |

> ⚠ **shared_expert ‖ routed_full 并行** (A8)：生产中走不同 stream，
>   端到端 latency ≈ max(shared, routed)，**不能简单相加**。
> ⚠ **util_vs_peak < 1** 说明 yaml.hbm_peak 估高，或算子未达带宽上限；
>   util_vs_achievable > 0.75 算打满 (N4)。
> ⚠ **act_quant pre vs post** (N5)：post-dispatch 是上限；
>   若生产用 pre-dispatch quant，实际 ≈ post / n_active。

### Grouped vs Loop (S2, N1 修正后)
| path                              | dev_mean_us | host_mean_us | dispatch_overhead_us |
|-----------------------------------|-------------|--------------|----------------------|
| grouped_matmul (1 call, E=n_act)  | ...         | ...          | ...                  |
| loop quant_matmul (n_act calls, 无 .item()) | ... | ...          | ...                  |
→ 若 loop ≤ grouped × 1.2，decode bs=1 下应考虑切 loop 路径。
→ 若差值随 n_active 增加而扩大，grouped 的优势随 E 增长才显现。

### n_active sweep (CPU offload 热专家命中率, A5 + N2)
| n_active | tpe | N | gemm_up_us | gemm_down_us | routed_full_us | util_eff |
| 1 | 1 | 1 | ... |
| 2 | 1 | 2 | ... |
| 3 | 1 | 3 | ... |
| 4 | 1 | 4 | ... |
| 5 | 1 | 5 | ... |
| 6 | 1 | 6 | ... |

### HBM 带宽实测 (N4 校准)
- hbm_peak_tb_s     = 1.6 (yaml)
- hbm_effective_tb_s= x.xx (D2 measure_hbm_effective_tb_s)
```

---

## 4. 环境变量（`env.sh`，已落盘）

从 `tools/p27_launch_ds4flash_npu.sh` 抽 MoE 必需子集：`ASCEND_TOOLKIT_HOME`, `TASK_QUEUE_ENABLE`, `STREAMS_PER_DEVICE`, `PYTORCH_NPU_ALLOC_CONF`, `IS_DEEPSEEK_V4`, `USE_NPU_MOE_GATING_TOP_K`。不自动选卡。

---

## 5. 已知风险与 fallback

| 风险 | fallback |
|------|----------|
| `npu_grouped_matmul` API signature 不一致 | `ops_runner.py` 单点改；yaml `ops.grouped_matmul` 字符串切换 |
| `group_list` cumsum vs per-expert count | `_as_kernel_group_list()` 一行切换 (`cfg.ops.group_list_mode`) |
| `npu_dynamic_quant` 输出 layout (`[N,H]+[N]` vs `[N,H]+[N,1]`) | sanity 阶段校验 |
| W8A8 dtype 与 kernel 不匹配 | yaml 提供 `act_dtype=bf16` baseline（跳过 quant 段） |
| 单步 GEMM M=1 太小，Event 抖动 | `repeat=1000`（S3）；报 p95+std；`--quick-mode` 100 |
| `npu-smi info` 容器内不可执行 | `print_npu_smi` 捕获异常 |
| `swiglu_fused` 算子在当前 sgl_kernel_npu 不存在 | `make_silu_mul` 静默降级到 unfused (A3) |
| `roofline.hbm_bandwidth_tb_s` 估错 | yaml 单点改；D2 smoke 时若 util < 1 说明带宽估高了 |
| **与论文 FP4×FP8 偏离** (W8A8) | README "已知偏离"段显式声明 ~2× HBM 流量差；不切 FP4 路径（生产 ckpt 就是 W8A8） |
| `timing.py` 在 attn / moe 两侧漂移 | 文件顶部 NOTE 注释提醒手动同步（不实施 S7 的共用层） |

---

## 6. 里程碑

| 阶段 | 主要交付 | 验收 |
|------|----------|------|
| **D0 配置校验** ✅ | yaml 已用 config.json 实测值填实；量化方案声明；双 HBM 带宽字段 (N4) | `moe_intermediate==2048`, scale strategy 已写, hbm_peak/effective 占位 |
| **D1 synthetic + shape** | `config.py` (含 `tokens_per_expert` N2) / `synthetic.py` (assert sum==N) / `sanity.py` / `timing.py` / `roofline.py` (含 shared N6)；`DRY_RUN=1` | dry-run 全部 assert 通过，sweep `n_active ∈ {1..6}` 每个 size 都自洽 (N2) |
| **D2 NPU smoke** | `init_npu` (含 N9 env 打印) + `ops_runner` (含 N1 无同步 loop + A3 fused 探测) + `bench_gemm_up.py`；`--sanity --repeat 10`；调 `measure_hbm_effective_tb_s` 回填 yaml (N4)；拿真实 ckpt 一层 weight 核对 gate/up 顺序 (N7) | 输出无 NaN；fused/unfused 都跑；effective HBM 数字回填；gate/up 顺序确认并注释到 ops_runner |
| **D3 五段 + 端到端 + shared + grouped_vs_loop** | 全部 `bench_*.py`；`bench_act_quant --pre-dispatch` (N5)；`bench_grouped_vs_loop` 用 `make_loop_runner` 工厂 (N1) | 8 段 JSON：act_quant×2 / gemm_up / silu_mul×2 / gemm_down / routed_full / shared_expert / grouped_vs_loop |
| **D4 报告 + roofline + profile + sweep** | `report.py` (双 util 列 N4 + shared N6 + 并行警示 A8 + pre/post 对照 N5) + `run_all.sh --profile` (N3 schedule+step) + `n_active {1..6}` sweep | `summary_decode.md` 含 util_peak/util_eff 双列、sweep 表、并行警示；profile dump 非空 |
| **D5（可选）** | A6 (num_tokens, top_k) 组合 sweep / A7 prefill (num_tokens=512) / bf16 baseline / N8 FP4 roofline 估算放 README | sweep CSV / md |

---

## 7. 首跑命令

```bash
cd tools/moe_microbench
source env.sh

# D0: 看 npu-smi、选卡
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=2

# D1: dry-run（无 NPU 也跑，校验形状 + assert）
DRY_RUN=1 bash run_all.sh

# D2: 单段 smoke
python -m moe_bench.bench_gemm_up --sanity --repeat 10 --quick-mode

# D3-D4: 正式 + sweep + profile
SWEEP_N_ACTIVE="1 2 3 4 5 6" PROFILE=1 bash run_all.sh
# 产物: results/summary_decode.md + results/profile/<seg>/
```

---

## 8. 与 attention microbench 的对照

| 维度 | attn_microbench | moe_microbench v2 |
|------|-----------------|---------------------|
| 主要 op | `npu_sparse_attn_sharedkv` | `npu_grouped_matmul`(×2) + `npu_quant_matmul` (shared) |
| 输入主要张量 | `q + paged KV` | `act bf16/int8 + grouped W8` |
| 子类型 | SWA / CSA / HCA | act_quant / gemm_up / silu_mul / gemm_down / shared / grouped_vs_loop |
| Synthetic 难点 | page_table 物理 id + sinks + topk 索引 | weight scale (E, 2I)/(E, H) + group_list cumsum 语义 |
| 排除范围 | KT MoE / QKV proj | gating / routing / dispatch / combine / CPU MoE / all-to-all |
| 共用方法 | Event + host wall, sanity, p95/p99/std, dry-run | 同（NOTE 手动同步，不抽共用层 / S7） |
| Roofline | 未做 | **本计划新增（S1）** |
| Dispatch overhead | 隐含在 host_mean | **本计划独立报（S4）** |

---

## 9. 关于 Claude 建议的处理记录

### v1 → v2（首轮 A1-A8, S1-S8）

| 编号 | 决策 | 实施位置 |
|------|------|----------|
| A1 (moe_inter=2048) | ✅ 必修 | yaml + synthetic.py assert |
| A2 (W4A8 vs W8A8) | ⚠️ 不切 FP4，README 声明 | README §"已知偏离" |
| A3 (silu fused 双路径) | ✅ 加 fused/unfused | `make_silu_mul` + bench_silu_mul --fused |
| A4 (scale shape assert) | ✅ 必修 | synthetic.assert_shapes |
| A5 (n_active {1..6}) | ✅ 必修 | yaml.sweep + run_all.sh |
| A6 (num_tokens, top_k) sweep | 🟡 D5 可选 | 计划保留 |
| A7 (prefill) | 🟡 D5 可选 + README 声明 | README §局限 |
| A8 (shared‖routed) | ✅ 必修 | report.py 警示 |
| S1 (roofline) | ✅ 必加 | roofline.py + report 列 |
| S2 (grouped vs loop) | ✅ 必加 | bench_grouped_vs_loop.py |
| S3 (repeat=1000 + quick-mode) | ✅ 合理 | yaml + CLI |
| S4 (dispatch_overhead) | ✅ 必加 | report.py 列 |
| S5 (profile dump) | ✅ 合理 | profile.py + run_all.sh --profile |
| S6 (README 局限) | ✅ 必修 | README 顶部段 |
| S7 (共用 _common/) | ❌ 不实施 | timing.py 顶部 NOTE 替代 |
| S8 (assertion) | ✅ 必修 | synthetic.assert_shapes（含 A1 防回归） |

### v2 → v3（二轮 N1-N9）

| 编号 | 决策 | 实施位置 | 阻塞阶段 |
|------|------|----------|----------|
| **N1** (loop `.item()` 同步污染) | ✅ 必修 | `make_loop_runner` 工厂，cumsum 提到 setup | D1 前 |
| **N2** (N 与 sum==N 冲突) | ✅ 必修 | 引入 `tokens_per_expert`，`N = n_active × tpe`；assert / yaml / sweep 全改 | D1 前 |
| **N3** (profile 单次空 trace) | ✅ 必修 | `schedule + step` 模式 | D4 前 |
| **N4** (HBM peak 过乐观) | ✅ | yaml 双带宽字段；`utilizations()` 双列；D2 跑 `measure_hbm_effective_tb_s` 校准 | D4 前 |
| **N5** (act_quant pre/post 位置) | ✅ | `bench_act_quant --pre-dispatch`；IMPL_PLAN §0.2 注释 | D3 |
| **N6** (shared roofline 漏算) | ✅ 必修 | `shared_weight_bytes()` | D4 前 |
| **N7** (gate/up 顺序) | 🟢 D2 smoke 检查 | ops_runner 注释；D2 拿真实 weight 核对 | D2 |
| **N8** (FP4 roofline 定量) | ✅ | README "已知偏离" 加 144MB/90μs vs 76MB/48μs | D4 |
| **N9** (容器内 ENV 已设) | ✅ | `print_npu_smi` 打印 `ASCEND_RT_VISIBLE_DEVICES` 当前值 | D1 |

---

*v3 — 2026-05-26：在 v2 基础上修复 N1（loop sync 污染）+ N2（N 定义冲突）+ N6（shared roofline 漏算）必修项；补强 N3-N5、N7-N9 polish。CPU offload 场景下纯 NPU MoE 计算路径。*
