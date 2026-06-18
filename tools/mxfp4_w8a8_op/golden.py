#!/usr/bin/env python3
"""Self-contained TORCH GOLDEN for the MXFP4 -> W8A8 fused dequant/requant operator.

This is the accuracy reference (validated end-to-end against the real network; eager and slow,
which is exactly why we want a fast NPU kernel). It defines the EXACT semantics the kernel must
match. No project dependencies beyond numpy/torch.

Layouts (one MoE projection, per expert):
  codes  uint8 [OUT, IN/2]   each byte = 2 nibbles, CONSECUTIVE order:
                             byte b -> IN pos 2b (low nibble = b&0x0F), 2b+1 (high = b>>4)
  scale  uint8 [OUT, IN/32]  e8m0 (8-bit exponent); 32 IN elements share one scale.
                             value = 2^(e-127), computed EXACTLY as bits=(e<<23) reinterpreted f32.

Output:
  q_int8 int8 [OUT, IN]      per-output-channel symmetric int8
  oscale       [OUT]         per-output-channel scale = amax/127  (bf16 for npu_fused_experts)
"""
import numpy as np
import torch

# FP4 e2m1 code table: nibble 0..15 -> value
FP4_TABLE = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6,
                      0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def e8m0_to_f32(e_u8: np.ndarray) -> np.ndarray:
    """e8m0 byte -> 2^(e-127), exact via integer bit pattern."""
    return ((e_u8.astype(np.uint32)) << 23).view(np.float32)


def dequant(codes_u8: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """[OUT,IN/2] u8 codes + [OUT,IN/32] u8 e8m0  ->  [OUT,IN] f32 (true weight)."""
    OUT, half = codes_u8.shape
    IN = half * 2
    lo = FP4_TABLE[codes_u8 & 0x0F]          # [OUT, IN/2] -> IN pos 0,2,4,...
    hi = FP4_TABLE[codes_u8 >> 4]            # [OUT, IN/2] -> IN pos 1,3,5,...
    w = np.empty((OUT, IN), np.float32)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    scale = np.repeat(e8m0_to_f32(scale_u8), 32, axis=1)   # [OUT, IN]
    return w * scale


def mxfp4_to_w8a8_golden(codes_u8: np.ndarray, scale_u8: np.ndarray):
    """The golden conversion. Returns (q_int8 [OUT,IN], oscale_bf16 [OUT]).

    Requant is PER OUTPUT CHANNEL (per row, amax over IN). The bf16 master mirrors the
    production path; a kernel computing internally in fp32 is equally acceptable (accuracy is
    judged end-to-end, see acceptance.py, not by bit-identical int8)."""
    w = torch.from_numpy(dequant(codes_u8, scale_u8)).to(torch.bfloat16)   # [OUT, IN]
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    oscale = amax / 127.0
    q = (w / oscale).round().clamp(-127, 127).to(torch.int8)
    return q, oscale.squeeze(-1).to(torch.bfloat16)


if __name__ == "__main__":
    # self-test on random data
    OUT, IN = 64, 4096
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, (OUT, IN // 2), dtype=np.uint8)
    scale = rng.integers(118, 136, (OUT, IN // 32), dtype=np.uint8)
    q, s = mxfp4_to_w8a8_golden(codes, scale)
    w = dequant(codes, scale)
    rec = q.float().numpy() * s.float().numpy()[:, None]
    cos = float((rec.reshape(-1) @ w.reshape(-1)) /
                (np.linalg.norm(rec) * np.linalg.norm(w) + 1e-9))
    print(f"golden self-test: q{tuple(q.shape)} int8, oscale{tuple(s.shape)}; "
          f"reconstruction cos(q*scale, true) = {cos:.6f}  (expect ~0.9999)")
