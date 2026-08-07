"""fp32 dequantize-then-matmul reference.

The kernel accumulates in fp32 from fp16 activations and fp16 scales, so we
cannot expect bitwise equality. The reference is the same quantized weights
dequantized in fp32, multiplied in fp32. Residual error is rounding, not
quantization vs the original dense matrix.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .quantize import dequantize


def linear_fp32(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Y = X @ W_dequant.T in fp32. x may be [K] or [B, K]."""
    w = dequantize(packed.cpu(), scales.cpu(), dtype=torch.float32)
    xv = x.detach().cpu().float()
    if xv.dim() == 1:
        xv = xv.unsqueeze(0)
    y = F.linear(xv, w)
    return y


def max_abs_err(y_kernel: torch.Tensor, y_ref: torch.Tensor) -> float:
    return (y_kernel.float().cpu() - y_ref.float().cpu()).abs().max().item()
