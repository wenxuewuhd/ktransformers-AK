"""W8A8 dequantization helper for DeepSeek-V4-Flash routed-expert weights.

The W8A8 quantization config for DSv4-Flash routed experts is:
  - weights:           int8, channel-wise symmetric, no zero-point, static
  - input_activations: int8, per-token symmetric, dynamic (runtime, not on disk)
  - scale dtype:       fp32, shape (out_features, 1) — one scale per output channel

So dequantization is the linear formula:

    fp32_weight[i, j] = int8_weight[i, j] * scale[i, 0]
"""

from __future__ import annotations

import numpy as np
import torch


def dequant_w8a8(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize a single W8A8 routed-expert weight tensor to fp32.

    Args:
        weight: int8 tensor, shape (out_features, in_features).
        scale:  fp32 tensor, shape (out_features, 1) — one scale per output channel.

    Returns:
        fp32 tensor, shape (out_features, in_features).
    """
    if weight.dtype != torch.int8:
        raise TypeError(f"weight must be int8, got {weight.dtype}")
    if scale.dtype != torch.float32:
        raise TypeError(f"scale must be fp32, got {scale.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if scale.ndim != 2 or scale.shape[1] != 1:
        raise ValueError(
            f"scale must have shape (out_features, 1), got {tuple(scale.shape)}"
        )
    if scale.shape[0] != weight.shape[0]:
        raise ValueError(
            f"scale.shape[0]={scale.shape[0]} must match weight.shape[0]={weight.shape[0]}"
        )

    return weight.to(torch.float32) * scale


def dequant_w8a8_numpy(
    weight: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Numpy variant; same semantics as :func:`dequant_w8a8`."""
    if weight.dtype != np.int8:
        raise TypeError(f"weight must be int8, got {weight.dtype}")
    if scale.dtype != np.float32:
        raise TypeError(f"scale must be fp32, got {scale.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {weight.shape}")
    if scale.ndim != 2 or scale.shape[1] != 1:
        raise ValueError(f"scale must have shape (out_features, 1), got {scale.shape}")
    if scale.shape[0] != weight.shape[0]:
        raise ValueError(
            f"scale.shape[0]={scale.shape[0]} must match weight.shape[0]={weight.shape[0]}"
        )

    return weight.astype(np.float32) * scale
