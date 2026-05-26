from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_bench.config import MoEConfig


@dataclass
class MoESyntheticTensors:
    x_bf16: torch.Tensor
    x_int8: torch.Tensor
    x_scale: torch.Tensor
    w_gate_up: torch.Tensor
    w_gate_up_scale: torch.Tensor
    w_down: torch.Tensor
    w_down_scale: torch.Tensor
    group_list: torch.Tensor
    mid_bf16: torch.Tensor
    mid_int8: torch.Tensor
    mid_scale: torch.Tensor
    w_shared_gate_up: torch.Tensor
    w_shared_gate_up_scale: torch.Tensor
    w_shared_down: torch.Tensor
    w_shared_down_scale: torch.Tensor


def _act_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


def _quantize_weight_per_channel(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel W8 from bf16 master weight [out, in]."""
    absmax = w_bf16.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = (absmax / 127.0).to(torch.float32)
    w_int8 = (w_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    return w_int8, scale


def _rand_weight(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    w_bf16 = torch.randn(shape, dtype=dtype, device=device, generator=gen)
    return _quantize_weight_per_channel(w_bf16)


def _make_group_list(cfg: MoEConfig, device: torch.device) -> torch.Tensor:
    counts = [cfg.tokens_per_expert] * cfg.n_active_experts
    return torch.tensor(counts, dtype=torch.int64, device=device)


def build_synthetic(
    cfg: MoEConfig,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> MoESyntheticTensors:
    dev = torch.device(device if device is not None else cfg.device)
    adt = _act_dtype(cfg.act_dtype)
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed if seed is None else seed)

    N = cfg.N
    H = cfg.hidden_size
    I = cfg.moe_intermediate_size
    S = cfg.shared_intermediate_size
    E = cfg.n_active_experts

    x_bf16 = torch.randn((N, H), dtype=adt, device=dev, generator=gen)
    x_int8 = torch.randint(-127, 128, (N, H), dtype=torch.int8, device=dev, generator=gen)
    x_scale = torch.full((N,), 0.1, dtype=torch.float32, device=dev)

    gate_up_parts: list[torch.Tensor] = []
    gate_up_scale_parts: list[torch.Tensor] = []
    down_parts: list[torch.Tensor] = []
    down_scale_parts: list[torch.Tensor] = []
    for _ in range(E):
        wi, ws = _rand_weight((2 * I, H), dtype=adt, device=dev, gen=gen)
        gate_up_parts.append(wi)
        gate_up_scale_parts.append(ws)
        w_hi, ws = _rand_weight((H, I), dtype=adt, device=dev, gen=gen)
        down_parts.append(w_hi.t().contiguous())
        down_scale_parts.append(ws)
    w_gate_up = torch.stack(gate_up_parts, dim=0)
    w_gate_up_scale = torch.stack(gate_up_scale_parts, dim=0)
    w_down = torch.stack(down_parts, dim=0)
    w_down_scale = torch.stack(down_scale_parts, dim=0)

    mid_bf16 = torch.randn((N, I), dtype=adt, device=dev, generator=gen)
    mid_int8 = torch.randint(-127, 128, (N, I), dtype=torch.int8, device=dev, generator=gen)
    mid_scale = torch.full((N,), 0.1, dtype=torch.float32, device=dev)

    w_shared_gate_up, w_shared_gate_up_scale = _rand_weight((2 * S, H), dtype=adt, device=dev, gen=gen)
    w_shared_hi, w_shared_down_scale = _rand_weight((H, S), dtype=adt, device=dev, gen=gen)
    w_shared_down = w_shared_hi.t().contiguous()

    group_list = _make_group_list(cfg, dev)

    return MoESyntheticTensors(
        x_bf16=x_bf16,
        x_int8=x_int8,
        x_scale=x_scale,
        w_gate_up=w_gate_up,
        w_gate_up_scale=w_gate_up_scale,
        w_down=w_down,
        w_down_scale=w_down_scale,
        group_list=group_list,
        mid_bf16=mid_bf16,
        mid_int8=mid_int8,
        mid_scale=mid_scale,
        w_shared_gate_up=w_shared_gate_up,
        w_shared_gate_up_scale=w_shared_gate_up_scale,
        w_shared_down=w_shared_down,
        w_shared_down_scale=w_shared_down_scale,
    )


def assert_shapes(cfg: MoEConfig, t: MoESyntheticTensors) -> None:
    N = cfg.N
    H = cfg.hidden_size
    I = cfg.moe_intermediate_size
    S = cfg.shared_intermediate_size
    E = cfg.n_active_experts

    assert cfg.moe_intermediate_size == 2048, "DSv4-Flash 实测 moe_intermediate=2048; 不要回退"
    assert cfg.n_routed_experts == 256
    assert cfg.num_experts_per_tok == 6
    assert cfg.n_shared_experts == 1

    assert N == cfg.n_active_experts * cfg.tokens_per_expert

    assert t.x_bf16.shape == (N, H)
    assert t.x_int8.shape == (N, H) and t.x_int8.dtype == torch.int8
    assert t.x_scale.shape == (N,) and t.x_scale.dtype == torch.float32

    # Routed weights: CANN layout [out, in] per expert (matches ckpt w1/w2/w3)
    assert t.w_gate_up.shape == (E, 2 * I, H) and t.w_gate_up.dtype == torch.int8
    assert t.w_gate_up_scale.shape == (E, 2 * I, 1) or t.w_gate_up_scale.shape == (E, 2 * I)
    assert t.w_down.shape == (E, I, H) and t.w_down.dtype == torch.int8
    assert t.w_down_scale.shape == (E, H, 1) or t.w_down_scale.shape == (E, H)

    assert t.group_list.shape == (E,)
    assert bool((t.group_list >= 0).all().item())
    assert int(t.group_list.sum().item()) == N

    assert t.mid_bf16.shape == (N, I)
    assert t.mid_int8.shape == (N, I) and t.mid_int8.dtype == torch.int8
    assert t.w_shared_gate_up.shape == (2 * S, H)
    assert t.w_shared_gate_up_scale.shape == (2 * S, 1) or t.w_shared_gate_up_scale.shape == (2 * S,)
    assert t.w_shared_down.shape == (S, H)
    assert t.w_shared_down_scale.shape == (H, 1) or t.w_shared_down_scale.shape == (H,)


def _selftest() -> None:
    from moe_bench.config import load_config

    for n_active in [1, 2, 3, 4, 5, 6]:
        cfg = apply_overrides_local(load_config(), n_active_experts=n_active)
        t = build_synthetic(cfg, seed=42, device="cpu")
        assert_shapes(cfg, t)
        print(
            f"n_active={n_active} N={cfg.N} group_list={t.group_list.tolist()} OK"
        )


def apply_overrides_local(cfg, **kwargs):
    from moe_bench.config import apply_overrides

    return apply_overrides(cfg, **kwargs)


if __name__ == "__main__":
    _selftest()
