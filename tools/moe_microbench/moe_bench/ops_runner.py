# Gate/up chunk order (N7): DSv4-Flash-W8A8 ckpt uses w1=gate, w3=up (gate-first fused as [gate|up] on dim=0).
# Weight layout: [out, in] per expert (w1/w3: [I,H], fused gate_up: [2I,H], w2/down: [H,I]).

from __future__ import annotations

import importlib
import itertools
from typing import Callable, Literal, Tuple

import torch
import torch.nn.functional as F

from moe_bench.config import MoEConfig
from moe_bench.synthetic import MoESyntheticTensors

GATE_UP_ORDER = "gate-first"


def _out_dtype(cfg: MoEConfig) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.out_dtype, torch.bfloat16)


def _squeeze_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 2 and scale.shape[-1] == 1:
        return scale.squeeze(-1)
    return scale


def _split_by_group_list(
    x: torch.Tensor, scale: torch.Tensor, group_list: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gl_cpu = group_list.cpu().tolist()
    xs, ss = [], []
    prev = 0
    for c in gl_cpu:
        end = prev + c
        xs.append(x[prev:end])
        ss.append(scale[prev:end])
        prev = end
    return xs, ss


def _grouped_matmul_loop_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    cfg: MoEConfig,
) -> torch.Tensor:
    """Fallback when npu_grouped_matmul unavailable (CANN 161002 on W8A8 M=1)."""
    gl_cpu = group_list.cpu().tolist()
    bounds = list(itertools.accumulate(gl_cpu))
    outs = []
    prev = 0
    for e, end in enumerate(bounds):
        if end == prev:
            continue
        outs.append(
            _quant_matmul(
                x[prev:end].clone(),
                weight[e],
                weight_scale[e],
                per_token_scale[prev:end].clone(),
                cfg,
            )
        )
        prev = end
    return torch.cat(outs, dim=0)


def _grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    cfg: MoEConfig,
) -> torch.Tensor:
    import torch_npu

    xs, act_scales = _split_by_group_list(x, per_token_scale, group_list)
    E = weight.shape[0]
    ws = [weight[i] for i in range(E)]
    wss = [_squeeze_scale(weight_scale[i]) for i in range(E)]
    act_scales_f32 = act_scales
    act_scales_gmm = [s.to(torch.bfloat16) for s in act_scales_f32]
    out_dtype = _out_dtype(cfg)
    gl = group_list.cpu().tolist()
    try:
        outs = torch_npu.npu_grouped_matmul(
            xs,
            ws,
            scale=wss,
            per_token_scale=act_scales_gmm,
            group_list=torch.tensor(gl, device=x.device, dtype=torch.int64),
            split_item=3,
            group_type=0,
            output_dtype=out_dtype,
        )
        if isinstance(outs, list):
            return torch.cat(outs, dim=0)
        return outs
    except RuntimeError:
        return _grouped_matmul_loop_fallback(x, weight, weight_scale, per_token_scale, group_list, cfg)


def _quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    cfg: MoEConfig,
) -> torch.Tensor:
    import torch_npu

    w = weight.clone()
    ws = _squeeze_scale(weight_scale).clone()
    return torch_npu.npu_quant_matmul(
        x,
        w,
        ws,
        pertoken_scale=per_token_scale,
        output_dtype=_out_dtype(cfg),
    )


def run_act_quant(
    t: MoESyntheticTensors, cfg: MoEConfig, *, pre_dispatch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    x = t.x_bf16[: cfg.num_tokens] if pre_dispatch else t.x_bf16
    return torch_npu.npu_dynamic_quant(x)


def run_gemm_up(t: MoESyntheticTensors, cfg: MoEConfig) -> torch.Tensor:
    return _grouped_matmul(
        t.x_int8, t.w_gate_up, t.w_gate_up_scale, t.x_scale, t.group_list, cfg
    )


def run_gemm_down(t: MoESyntheticTensors, cfg: MoEConfig) -> torch.Tensor:
    return _grouped_matmul(
        t.mid_int8, t.w_down, t.w_down_scale, t.mid_scale, t.group_list, cfg
    )


def _unfused_silu_mul(gate_up_bf16: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up_bf16.chunk(2, dim=-1)
    return F.silu(gate) * up


def make_silu_mul(
    cfg: MoEConfig,
    *,
    force_fused: bool | None = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    fused_path = cfg.ops.get("swiglu_fused")

    if force_fused is False:
        return _unfused_silu_mul, "unfused"

    if force_fused is True:
        if not fused_path:
            raise RuntimeError("--fused requested but ops.swiglu_fused is null in yaml")
        mod_name, fn_name = fused_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(mod_name), fn_name)
        return fn, "fused"

    if fused_path:
        try:
            mod_name, fn_name = fused_path.rsplit(".", 1)
            fn = getattr(importlib.import_module(mod_name), fn_name)
            return fn, "fused"
        except Exception:
            pass
    return _unfused_silu_mul, "unfused"


def run_act_quant_mid(
    t: MoESyntheticTensors, cfg: MoEConfig, mid_bf16: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    return torch_npu.npu_dynamic_quant(mid_bf16)


def run_routed_compute_only(t: MoESyntheticTensors, cfg: MoEConfig) -> torch.Tensor:
    up = run_gemm_up(t, cfg)
    mid = _unfused_silu_mul(up)
    mid_int8, mid_scale = run_act_quant_mid(t, cfg, mid)
    t.mid_int8 = mid_int8
    t.mid_scale = mid_scale
    return run_gemm_down(t, cfg)


def run_routed_post_dispatch(t: MoESyntheticTensors, cfg: MoEConfig) -> torch.Tensor:
    x_int8, x_scale = run_act_quant(t, cfg, pre_dispatch=False)
    t.x_int8 = x_int8
    t.x_scale = x_scale
    return run_routed_compute_only(t, cfg)


def run_shared_expert(t: MoESyntheticTensors, cfg: MoEConfig) -> torch.Tensor:
    import torch_npu

    x = t.x_bf16[: cfg.num_tokens]
    x_int8, x_scale = torch_npu.npu_dynamic_quant(x)
    up = _quant_matmul(x_int8, t.w_shared_gate_up, t.w_shared_gate_up_scale, x_scale, cfg)
    mid = _unfused_silu_mul(up)
    mid_int8, mid_scale = torch_npu.npu_dynamic_quant(mid)
    return _quant_matmul(mid_int8, t.w_shared_down, t.w_shared_down_scale, mid_scale, cfg)


def make_loop_runner(
    t: MoESyntheticTensors,
    cfg: MoEConfig,
    target: Literal["up", "down"] = "up",
) -> Callable[[], torch.Tensor | None]:
    gl_cpu = t.group_list.cpu().tolist()
    bounds = list(itertools.accumulate(gl_cpu))
    x_parts: list[torch.Tensor] = []
    s_parts: list[torch.Tensor] = []
    prev = 0
    for end in bounds:
        if target == "up":
            x_parts.append(t.x_int8[prev:end].clone())
            s_parts.append(t.x_scale[prev:end].clone())
        else:
            x_parts.append(t.mid_int8[prev:end].clone())
            s_parts.append(t.mid_scale[prev:end].clone())
        prev = end

    def loop_fn() -> torch.Tensor | None:
        outs = []
        for e, end in enumerate(bounds):
            if gl_cpu[e] == 0:
                continue
            if target == "up":
                outs.append(
                    _quant_matmul(
                        x_parts[e], t.w_gate_up[e], t.w_gate_up_scale[e], s_parts[e], cfg
                    )
                )
            else:
                outs.append(
                    _quant_matmul(
                        x_parts[e], t.w_down[e], t.w_down_scale[e], s_parts[e], cfg
                    )
                )
        return torch.cat(outs, dim=0) if outs else None

    return loop_fn
