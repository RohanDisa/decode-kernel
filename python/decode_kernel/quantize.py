"""Symmetric grouped 4-bit weight quantization, group size 128.

Format
------
Each group of 128 fp16/fp32 weights shares one fp16 scale:

    scale = max(abs(W_group)) / 7
    q     = clamp(round(W / scale), -7, 7)     # symmetric signed 4-bit

-8 is representable in the nibble encoding (two's complement) but is not
emitted by this quantizer, because |W|/scale is at most 7.

Eight consecutive q values are packed into one little-endian int32 word,
low nibble first. The stored nibble is the two's-complement bit pattern.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 128
NIBBLES_PER_WORD = 8
QMIN, QMAX = -7, 7  # -8 is a valid nibble; this quantizer never emits it


def quantize_and_pack(
    weight: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize W[M, K] to packed int32 [M, K/8] and fp16 scales [M, K/group]."""
    if weight.dim() != 2:
        raise ValueError("weight must be [M, K]")
    m, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} is not divisible by group_size={group_size}")
    if k % NIBBLES_PER_WORD != 0:
        raise ValueError(f"K={k} is not divisible by {NIBBLES_PER_WORD}")

    w = weight.float()
    groups = k // group_size
    grouped = w.view(m, groups, group_size)
    amax = grouped.abs().amax(dim=-1)
    scale = torch.clamp(amax / float(QMAX), min=1e-8)
    q = torch.round(grouped / scale.unsqueeze(-1)).clamp(QMIN, QMAX).to(torch.int32)
    q = q.view(m, k)

    q8 = (q.view(m, k // NIBBLES_PER_WORD, NIBBLES_PER_WORD) & 0xF).to(torch.int32)
    packed = torch.zeros(m, k // NIBBLES_PER_WORD, dtype=torch.int32, device=w.device)
    for n in range(NIBBLES_PER_WORD):
        packed |= q8[:, :, n] << (4 * n)

    return packed.contiguous(), scale.half().contiguous()


def unpack_to_int(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 words to signed int32 q-values in [-8, 7], shape [M, K]."""
    m, packed_cols = packed.shape
    parts = []
    for n in range(NIBBLES_PER_WORD):
        nibble = (packed >> (4 * n)) & 0xF
        q = torch.where(nibble >= 8, nibble - 16, nibble)
        parts.append(q)
    q = torch.stack(parts, dim=-1).reshape(m, packed_cols * NIBBLES_PER_WORD)
    return q.to(torch.int32)


def dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize the dequantized weight matrix. For tests only — the kernel
    never does this."""
    q = unpack_to_int(packed).to(dtype)
    m, k = q.shape
    groups = k // group_size
    scale = scales.to(dtype).reshape(m, groups, 1)
    return (q.view(m, groups, group_size) * scale).reshape(m, k)
