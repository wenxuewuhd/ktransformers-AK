#!/usr/bin/env python3
"""P2.11 实验 B+：CPU MoE 单层离线数值对账。

把 ``KTMoEWrapper(LLAMAFILE, GGUF Q8_0)`` 在 **同一组输入** 下的输出与一份
**纯 PyTorch dequant(W8A8) + SwiGLU MoE** 参考实现对比，判定 CPU MoE 本身是
否数值正确。**与 sglang server 无关**，几分钟跑完。

诊断价值（与 ``Handoff §6.11`` 联动）：

  | A | B (--kt-num-gpu-experts 0) | B+（本工具） | 锁定子系统      |
  |---|---|---|---|
  | 仍乱 | 仍乱 | **不一致** | **CPU MoE (KT)** |
  | 仍乱 | 仍乱 | 一致      | shared 路径      |

接口对齐（与 ``tools/phase12_llamafile_moe_smoke.py`` 完全一致）：
  - hidden:        (B, H) bf16
  - topk_ids:      (B, K) long
  - topk_weights:  (B, K) fp32 (softmax 归一化)
  - 输出 out:      (B, H)

阈值：cosine_sim >= 0.99 / max_rel_err <= 5%（Q8_0 量化已知 ~1–2% 误差，
加上 W8A8 per-out-channel scale 与 Q8_0 per-32-block scale 不同步可能再多一
点，5% 是工程容忍上限；模型生成不至于退化成 padding）。

用法::

  ${PYTHON_BIN} tools/p27_cpu_moe_reference_check.py \\
    --w8a8 /workspace/models/DeepSeek-V4-Flash-W8A8 \\
    --gguf /workspace/models/cache/dsv4_layer3.gguf \\
    --layer-idx 3 --batch 4 --seed 1

环境要求：
  - **不要** 设置 ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` —— 那个开关会跳过 torch_npu
    自加载，导致 ``KTMoEWrapper`` 初始化时 ``torch.empty(..., pin_memory=True)``
    报 "Need to provide pin_memory allocator to use pin memory."。
  - 默认 ``--device npu``：input/output 都放到 NPU 上，stream 用真实
    ``torch.npu.current_stream(device).npu_stream``。这是与 sglang server 里完全
    一致的调用路径。

为什么不能用 ``--device cpu`` + ``stream=0``：
  ``cpu_infer.submit_with_cuda_stream(stream, task)`` 是依赖 GPU/NPU stream callback
  把 CPU 任务真正触发起来的；stream=0 时 task 永远不会执行，``output_cpu`` 保持
  初始 zeros，forward 返回全 0（``isfinite`` 仍 True，但语义已坏）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 提前 try-import torch_npu，让它注册 host pin allocator（如果环境正常的话）。
# 这必须在 `_ensure_pin_memory_or_patch()` 之前，否则 pin_memory 自检会误以为
# allocator 不可用并降级 monkey-patch。
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def _ensure_pin_memory_or_patch() -> None:
    """Probe whether host pin_memory allocator is available; if not, monkey-patch.

    ``KTMoEWrapper.__init__`` calls ``torch.empty(..., device='cpu', pin_memory=True)``,
    which requires CUDA *or* torch_npu to be loaded and to have registered a host
    pin allocator. When neither is registered (e.g. user set
    ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``), torch raises::

        RuntimeError: Need to provide pin_memory allocator to use pin memory.

    For this offline-numeric-check tool we don't actually need the memory pinned
    (we never D2H-copy it to a device), so it's safe to fall back to a plain CPU
    tensor.
    """
    try:
        _ = torch.empty(1, dtype=torch.bool, device="cpu", pin_memory=True)
        return
    except RuntimeError as e:
        if "pin_memory" not in str(e).lower():
            raise

    _orig_empty = torch.empty

    def _empty_no_pin(*args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.pop("pin_memory", False) and kwargs.get("device", None) in (None, "cpu", torch.device("cpu")):
            return _orig_empty(*args, **kwargs)
        return _orig_empty(*args, **kwargs)

    torch.empty = _empty_no_pin  # type: ignore[assignment]
    print(
        "[warn] pin_memory allocator 不可用（torch_npu 未加载？），"
        "已 monkey-patch torch.empty 把 pin_memory=True 降级为 False；"
        "建议取消 TORCH_DEVICE_BACKEND_AUTOLOAD=0 让 torch_npu 正常加载。",
        file=sys.stderr,
    )


_ensure_pin_memory_or_patch()

# 复用 convert 脚本里已经写好的安全 W8A8 加载工具
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from convert_w8a8_to_gguf_q8_0 import (  # noqa: E402
    _detect_experts_uri,
    _dequant_int8,
    _load_weight_map,
    _open_shard,
)


def load_expert_weights_fp32(
    model_dir: Path,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 HF W8A8 safetensors 读 256 个 expert 的 w1/w3/w2，dequant 到 fp32。

    Returns
    -------
    w1 : (E, I, H) fp32  — gate proj weight
    w3 : (E, I, H) fp32  — up proj weight
    w2 : (E, H, I) fp32  — down proj weight
    """
    weight_map = _load_weight_map(model_dir)
    experts_prefix, (gate_n, up_n, down_n) = _detect_experts_uri(weight_map, layer_idx)
    cache: dict = {}

    w1_list: list[torch.Tensor] = []
    w3_list: list[torch.Tensor] = []
    w2_list: list[torch.Tensor] = []
    for e in range(num_experts):
        for proj_name, target in (
            (gate_n, w1_list),
            (up_n, w3_list),
            (down_n, w2_list),
        ):
            wk = f"{experts_prefix}.{e}.{proj_name}.weight"
            sk = f"{experts_prefix}.{e}.{proj_name}.weight_scale"
            h = _open_shard(model_dir, weight_map, cache, wk)
            target.append(_dequant_int8(h.get_tensor(wk), h.get_tensor(sk)))

    w1 = torch.stack(w1_list, dim=0)
    w3 = torch.stack(w3_list, dim=0)
    w2 = torch.stack(w2_list, dim=0)

    assert w1.shape == (num_experts, moe_intermediate_size, hidden_size), (
        f"w1 shape mismatch: got {tuple(w1.shape)}, "
        f"expected ({num_experts}, {moe_intermediate_size}, {hidden_size})"
    )
    assert w3.shape == (num_experts, moe_intermediate_size, hidden_size), (
        f"w3 shape mismatch: got {tuple(w3.shape)}, "
        f"expected ({num_experts}, {moe_intermediate_size}, {hidden_size})"
    )
    assert w2.shape == (num_experts, hidden_size, moe_intermediate_size), (
        f"w2 shape mismatch: got {tuple(w2.shape)}, "
        f"expected ({num_experts}, {hidden_size}, {moe_intermediate_size})"
    )
    return w1, w3, w2


def reference_moe_forward(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch DSv4 MoE forward (fp32 accumulation, slow but exact)。

    For each token i, for each k ∈ topk:
        expert e  = topk_ids[i, k]
        gate      = hidden[i] @ w1[e].T          # (I,)
        up        = hidden[i] @ w3[e].T          # (I,)
        out_i    += topk_weights[i, k] * (SiLU(gate) * up) @ w2[e].T
    """
    B, H = hidden.shape
    K = topk_ids.shape[1]
    h = hidden.float()
    out = torch.zeros(B, H, dtype=torch.float32)
    for i in range(B):
        for k in range(K):
            e = int(topk_ids[i, k].item())
            w_eff = float(topk_weights[i, k].item())
            gate = h[i] @ w1[e].t()
            up = h[i] @ w3[e].t()
            act = F.silu(gate) * up
            d = act @ w2[e].t()
            out[i].add_(w_eff * d)
    return out.to(hidden.dtype)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b) / (a.norm() * b.norm() + 1e-12)


def _resolve_device_and_stream(want: str, npu_id: int) -> tuple[torch.device, int]:
    """Pick a torch.device and matching stream handle for cpu_infer.

    cpu_infer.submit_with_cuda_stream relies on a GPU/NPU stream callback to fire
    the CPU task. A stream handle of 0 on a non-cpu device is invalid and would
    leave the task uneexecuted (output buffer stays all-zero).
    """
    if want == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available()=False")
        device = torch.device("cuda", 0)
        return device, int(torch.cuda.current_stream(device).cuda_stream)

    if want == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "--device npu but `import torch_npu` failed. 检查 torch / torch_npu "
                "版本（必须主版本一致，如同为 2.8.x），或 unset TORCH_DEVICE_BACKEND_AUTOLOAD=0."
            ) from e
        if not torch.npu.is_available():
            raise RuntimeError("--device npu but torch.npu.is_available()=False")
        device = torch.device("npu", npu_id)
        return device, int(torch.npu.current_stream(device).npu_stream)

    return torch.device("cpu"), 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--w8a8",
        type=Path,
        required=True,
        help="HF W8A8 模型目录（含 model.safetensors.index.json）",
    )
    ap.add_argument(
        "--method",
        type=str,
        default="LLAMAFILE",
        choices=("LLAMAFILE", "MOE_INT8"),
        help=(
            "KT CPU MoE backend："
            "LLAMAFILE 走 GGUF Q8_0/BF16 → llamafile_sgemm（aarch64 上 Q8_0 NaN/BF16 throw，**已知坏**）；"
            "MOE_INT8 走 cblas_gemm_s8s8s32（OpenBLAS/KleidiAI）+ kt-kernel merged safetensor，已知 aarch64 健全。"
        ),
    )
    ap.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="LLAMAFILE 时必填：该层 GGUF Q8_0/BF16 文件，例如 /workspace/models/cache/dsv4_layer3.gguf",
    )
    ap.add_argument(
        "--kt-int8-dir",
        type=Path,
        default=None,
        help=(
            "MOE_INT8 时必填：包含 ``blk.<L>.safetensors`` 的目录，"
            "由 ``tools/build_kt_int8_merged_safetensor.py`` 生成。"
        ),
    )
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-experts-per-tok", type=int, default=6)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--cpuinfer-threads", type=int, default=24)
    ap.add_argument("--threadpool-count", type=int, default=8)
    ap.add_argument(
        "--numa-nodes",
        type=str,
        default=None,
        help="逗号分隔的 NUMA node id 列表，例如 ``0`` 或 ``0,1``；不传则 autodetect（受 cpu_infer "
        "singleton 缓存影响，第一次创建后整个进程内固定）。降为单 NUMA 可绕 TP merge 路径调试。",
    )
    ap.add_argument("--chunked-prefill-size", type=int, default=8)
    ap.add_argument(
        "--device",
        type=str,
        default="npu",
        choices=("npu", "cuda", "cpu"),
        help=(
            "input/output 张量与 cpu_infer stream 所在设备。**强烈建议保持默认 npu**："
            "cpu_infer 用 stream callback 触发 CPU 任务，stream=0 (cpu 模式) 会让 task "
            "永远不执行 → forward 全 0（仍 isfinite，但语义已坏）。"
        ),
    )
    ap.add_argument(
        "--npu-id",
        type=int,
        default=0,
        help="NPU device id（受 ASCEND_RT_VISIBLE_DEVICES 重映射，单卡通常是 0）",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--deterministic-input",
        action="store_true",
        help="使用全 1 输入 + identity routing (调试 NaN 来源)",
    )
    ap.add_argument(
        "--cos-min",
        type=float,
        default=0.99,
        help="cosine_sim 下限（与 ref 一致性）",
    )
    ap.add_argument(
        "--rel-tol",
        type=float,
        default=0.05,
        help="max_rel_err 上限（Q8_0 量化典型 ~1-2%，工程容忍 5%）",
    )
    args = ap.parse_args()

    w8a8_dir = args.w8a8.expanduser().resolve()
    if not (w8a8_dir / "model.safetensors.index.json").is_file():
        print(f"ERROR: W8A8 dir 缺 index.json: {w8a8_dir}", file=sys.stderr)
        return 2

    if args.method == "LLAMAFILE":
        if args.gguf is None:
            print("ERROR: --method LLAMAFILE 需要 --gguf <file>", file=sys.stderr)
            return 2
        gguf_file = args.gguf.expanduser().resolve()
        if not gguf_file.is_file():
            print(f"ERROR: GGUF 不存在: {gguf_file}", file=sys.stderr)
            return 2
        kt_int8_dir = None
    elif args.method == "MOE_INT8":
        if args.kt_int8_dir is None:
            print("ERROR: --method MOE_INT8 需要 --kt-int8-dir <dir>", file=sys.stderr)
            return 2
        kt_int8_dir = args.kt_int8_dir.expanduser().resolve()
        expected_blk = kt_int8_dir / f"blk.{args.layer_idx}.safetensors"
        if not expected_blk.is_file():
            print(
                f"ERROR: MOE_INT8 缺 layer 文件: {expected_blk}\n"
                f"先跑：tools/build_kt_int8_merged_safetensor.py --input {w8a8_dir} "
                f"--output {kt_int8_dir} --layer-idx {args.layer_idx}",
                file=sys.stderr,
            )
            return 2
        gguf_file = None
    else:
        print(f"ERROR: 未知 method {args.method!r}", file=sys.stderr)
        return 2

    # ---- device & stream ----
    device, stream_handle = _resolve_device_and_stream(args.device, args.npu_id)
    print(f"[env] device={device} stream_handle={stream_handle}")
    if stream_handle == 0 and device.type != "cpu":
        print(
            "ERROR: 拿到 stream_handle=0，但 device 非 cpu。这会让 cpu_infer task "
            "永远不执行 → forward 全 0。请检查 torch_npu / torch.cuda 是否正常。",
            file=sys.stderr,
        )
        return 2
    if device.type == "cpu":
        print(
            "[warn] device=cpu → stream=0 → KTMoEWrapper 的 cpu_infer 任务不会被触发，"
            "forward 输出必然全 0。仅用于回归这个已知失败模式；正确做法是 --device npu。",
            file=sys.stderr,
        )

    # ---- ref ----
    print(
        f"[ref] 加载 W8A8 layer {args.layer_idx} 的 {args.num_experts} 个 expert "
        f"(dequant int8 → fp32)..."
    )
    w1, w3, w2 = load_expert_weights_fp32(
        w8a8_dir,
        args.layer_idx,
        args.num_experts,
        args.hidden_size,
        args.moe_intermediate_size,
    )
    print(
        f"[ref] w1={tuple(w1.shape)} w3={tuple(w3.shape)} w2={tuple(w2.shape)} "
        f"({w1.dtype})"
    )

    # ---- cand ----
    cand_weight_path = str(gguf_file) if args.method == "LLAMAFILE" else str(kt_int8_dir)
    print(f"[cand] 加载 KTMoEWrapper({args.method}) from {cand_weight_path}...")
    try:
        from kt_kernel import KTMoEWrapper  # noqa: WPS433
    except ImportError as e:
        print(
            "ERROR: 无法 import kt_kernel。请先 `cd kt-kernel && pip install -e . --no-deps`。\n"
            f"原始错误：{e}",
            file=sys.stderr,
        )
        return 2

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)  # 全 CPU expert
    numa_nodes: list[int] | None = None
    if args.numa_nodes:
        numa_nodes = [int(x) for x in args.numa_nodes.split(",") if x.strip()]
        if len(numa_nodes) != args.threadpool_count:
            print(
                f"[warn] len(--numa-nodes)={len(numa_nodes)} != --threadpool-count={args.threadpool_count}; "
                f"adjust threadpool_count to {len(numa_nodes)} to match numa_nodes.",
                file=sys.stderr,
            )
            args.threadpool_count = len(numa_nodes)
    wrapper = KTMoEWrapper(
        layer_idx=args.layer_idx,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        weight_path=cand_weight_path,
        chunked_prefill_size=args.chunked_prefill_size,
        method=args.method,
        numa_nodes=numa_nodes,
    )
    # GeneralMoEWrapper(MOE_INT8).load_weights(physical_to_logical_map_cpu) 是必填，
    # LlamafileMoEWrapper.load_weights() 也接受同名 kwarg（默认 identity）。
    if args.method == "MOE_INT8":
        physical_to_logical = torch.arange(args.num_experts, dtype=torch.int32, device="cpu")
        wrapper.load_weights(physical_to_logical_map_cpu=physical_to_logical)
    else:
        wrapper.load_weights()

    # ---- inputs (一次性在 CPU 上构造作为 ref 输入；再 .to(device) 喂给 cand) ----
    torch.manual_seed(args.seed)
    if args.deterministic_input:
        # 极简输入用于排查 NaN：所有 token routing 走前 K 个 expert，weight 均分。
        hidden_cpu = torch.ones(args.batch, args.hidden_size, dtype=torch.bfloat16) * 0.01
        topk_ids_cpu = torch.arange(args.num_experts_per_tok, dtype=torch.long).expand(args.batch, -1).contiguous()
        topk_weights_cpu = torch.full(
            (args.batch, args.num_experts_per_tok),
            1.0 / args.num_experts_per_tok,
            dtype=torch.float32,
        )
    else:
        hidden_cpu = torch.randn(args.batch, args.hidden_size, dtype=torch.bfloat16)
        topk_ids_cpu = torch.randint(
            0,
            args.num_experts,
            (args.batch, args.num_experts_per_tok),
            dtype=torch.long,
        )
        tw = torch.randn(args.batch, args.num_experts_per_tok, dtype=torch.float32)
        topk_weights_cpu = torch.softmax(tw, dim=-1)

    print(
        f"[run] hidden={tuple(hidden_cpu.shape)} topk_ids={tuple(topk_ids_cpu.shape)} "
        f"topk_weights={tuple(topk_weights_cpu.shape)}"
    )
    print(
        f"[run] sample routing: topk_ids[0]={topk_ids_cpu[0].tolist()} "
        f"weights[0]={[round(x, 3) for x in topk_weights_cpu[0].tolist()]}"
    )

    # ---- ref forward (pure pytorch, fp32, on CPU) ----
    print("[ref]  forward (pure pytorch, fp32 accumulation)...")
    ref_out = reference_moe_forward(hidden_cpu, topk_ids_cpu, topk_weights_cpu, w1, w3, w2)

    # ---- cand forward (KTMoEWrapper, on `device` with real stream) ----
    print(f"[cand] forward (KTMoEWrapper, {args.method}) on {device} stream={stream_handle}...")
    hidden = hidden_cpu.to(device)
    topk_ids = topk_ids_cpu.to(device)
    topk_weights = topk_weights_cpu.to(device)
    cand_out = wrapper.forward(hidden, topk_ids, topk_weights, stream_handle)
    if isinstance(cand_out, (list, tuple)):
        cand_out = cand_out[0]
    # 等 stream 上 D2H/H2D 拷贝完成
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)

    # ---- compare ----
    a = cand_out.detach().cpu().float()
    b = ref_out.float()
    if a.shape != b.shape:
        print(
            f"FAIL: shape mismatch — cand={tuple(a.shape)} ref={tuple(b.shape)}",
            file=sys.stderr,
        )
        return 1

    import math

    cand_finite = bool(torch.isfinite(a).all().item())
    ref_finite = bool(torch.isfinite(b).all().item())

    cos = float(cosine_sim(a, b).item())
    max_abs = float((a - b).abs().max().item())
    ref_max = float(b.abs().max().item())
    max_rel = max_abs / (ref_max + 1e-12)

    print()
    print("=== 数值对账 ===")
    print(f"  cosine_sim   = {cos:.6f}     (期望 >= {args.cos_min:.3f})")
    print(f"  max_abs_err  = {max_abs:.4e}")
    print(f"  max_rel_err  = {max_rel:.4%}   (期望 <= {args.rel_tol:.2%})")
    print(f"  ref  L2/max  = {b.norm():.4e} / {b.abs().max():.4e}")
    print(f"  cand L2/max  = {a.norm():.4e} / {a.abs().max():.4e}")
    print(f"  finite(ref)  = {ref_finite}")
    print(f"  finite(cand) = {cand_finite}")
    with torch.no_grad():
        print(f"  ref [0,:8]   = {[round(float(x), 6) for x in b[0, :8].tolist()]}")
        print(f"  cand[0,:8]   = {[round(float(x), 6) for x in a[0, :8].tolist()]}")
        diff = (a - b)
        print(f"  diff[0,:8]   = {[round(float(x), 6) for x in diff[0, :8].tolist()]}")
        print(f"  ratio[0,:8]  = {[round(float(a[0,i]/b[0,i]) if abs(float(b[0,i]))>1e-9 else float('nan'), 4) for i in range(8)]}")
        token1_cos = float(cosine_sim(a[0], b[0]).item()) if a.shape[0] > 0 else float('nan')
        print(f"  cosine[token0] = {token1_cos:.6f}")
        if a.shape[0] > 1:
            token2_cos = float(cosine_sim(a[1], b[1]).item())
            print(f"  cosine[token1] = {token2_cos:.6f}")
        if a.shape[0] > 0:
            tok0_a = a[0].abs()
            tok0_b = b[0].abs()
            print(
                f"  token0  cand: nonzero={(tok0_a > 1e-7).sum().item()}/{tok0_a.numel()}  "
                f"max={tok0_a.max().item():.4e}  mean={tok0_a.mean().item():.4e}"
            )
            print(
                f"  token0   ref: nonzero={(tok0_b > 1e-7).sum().item()}/{tok0_b.numel()}  "
                f"max={tok0_b.max().item():.4e}  mean={tok0_b.mean().item():.4e}"
            )
    if not cand_finite:
        nan_count = int(torch.isnan(a).sum().item())
        inf_count = int(torch.isinf(a).sum().item())
        total = a.numel()
        print(f"  cand stats   : NaN={nan_count}/{total}  Inf={inf_count}/{total}")
        finite_cand = a[torch.isfinite(a)]
        if finite_cand.numel() > 0:
            print(
                f"  cand finite  : min={finite_cand.min():.4e} "
                f"max={finite_cand.max():.4e} mean={finite_cand.mean():.4e}"
            )
        print(f"  cand sample  : {a.flatten()[:16].tolist()}")
    print()

    # Treat non-finite cand as a hard FAIL (NaN<x is False so naive cos check would silently PASS).
    if (not cand_finite) or math.isnan(cos) or math.isnan(max_rel) or cos < args.cos_min or max_rel > args.rel_tol:
        print(
            "RESULT: FAIL — CPU MoE 数值与参考实现不一致。\n"
            "       → 锁定 CPU MoE / GGUF Q8_0 / KTMoEWrapper.forward 这一路。\n"
            "       检查方向：\n"
            "         1) GGUF 张量名 / shape 是否与 kt-kernel 期望对齐\n"
            "            (blk.{L}.ffn_{gate,up,down}_exps.weight, (n_embd, n_ff, E))\n"
            "         2) Q8_0 沿哪个维度分 block，与 forward kernel 一致？\n"
            "         3) KTMoEWrapper.forward 在非 pin_memory 输入上的 stride/dtype 处理\n"
            "         4) numa_nodes=None 是否真按 8 NUMA 切分；可临时改 list(range(8))",
            file=sys.stderr,
        )
        return 1

    print(
        "RESULT: PASS — CPU MoE 与参考实现一致。\n"
        "       → 排除 CPU MoE 嫌疑。\n"
        "       端到端 B 仍乱说明问题在 **shared 路径**：\n"
        "         attention / MLA / RoPE (ComplexExp) / NSA Compressor / W8A8 装载 / embed / lm_head。\n"
        "       下一步走 §6.11.5 实验 D（中间张量 dump，定位是哪一层先发散）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
