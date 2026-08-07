"""W4A16 GEMV: 4-bit grouped weights, fp16 activations, fused dequant."""

from __future__ import annotations

from typing import Optional

import torch

from .quantize import GROUP_SIZE, dequantize, quantize_and_pack

__all__ = [
    "GROUP_SIZE",
    "cuda_available",
    "dequantize",
    "gemv",
    "quantize_and_pack",
    "w4a16_linear",
]


def cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from . import _cuda  # noqa: F401

        return True
    except Exception:
        return False


def gemv(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    version: int = 5,
    split_k: int = 4,
    reuse_weights: bool = False,
    warps_per_row: int = 1,
) -> torch.Tensor:
    """Y[B, M] = W_dequant[M, K] @ X[B, K].T, fused in one kernel.

    packed: int32 [M, K/8]
    scales: fp16 [M, K/128]
    x:      fp16 [B, K]
    """
    from . import _cuda

    if x.dim() == 1:
        x = x.unsqueeze(0)
    return _cuda.w4a16_gemv(
        packed,
        scales,
        x,
        int(version),
        int(split_k),
        bool(reuse_weights),
        int(warps_per_row),
    )


def w4a16_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    version: int = 5,
    split_k: Optional[int] = None,
    reuse_weights: Optional[bool] = None,
) -> torch.Tensor:
    """Drop-in for a quantized F.linear without a bias."""
    if split_k is None:
        split_k = 4 if version == 4 else 1
    if reuse_weights is None:
        reuse_weights = x.dim() == 2 and x.size(0) > 1
    return gemv(
        packed,
        scales,
        x,
        version=version,
        split_k=split_k,
        reuse_weights=reuse_weights,
        warps_per_row=1,
    )
