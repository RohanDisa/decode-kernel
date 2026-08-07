"""CPU tests for the packing format. No GPU required."""

from __future__ import annotations

import pytest
import torch

from decode_kernel.quantize import (
    GROUP_SIZE,
    dequantize,
    quantize_and_pack,
    unpack_to_int,
)


@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (32, 512)])
def test_pack_unpack_q_roundtrip(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape)
    packed, scales = quantize_and_pack(w)
    q = unpack_to_int(packed)
    assert int(q.min()) >= -7 and int(q.max()) <= 7
    recon = dequantize(packed, scales)
    scale = scales.float().repeat_interleave(GROUP_SIZE, dim=1)
    assert torch.allclose(recon, q.float() * scale, atol=1e-6)


@pytest.mark.parametrize("shape", [(8, 128), (16, 256)])
def test_reconstruction_within_half_step(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape)
    packed, scales = quantize_and_pack(w)
    recon = dequantize(packed, scales)
    scale = scales.float().repeat_interleave(GROUP_SIZE, dim=1)
    # Symmetric quantizer uses scale = amax/7 stored as fp16, so allow a
    # little more than half a step for the scale rounding.
    err = (recon - w.float()).abs()
    assert torch.all(err <= 0.5 * scale + 1e-3)


def test_pack_layout_low_nibble_first():
    # amax = 7 → scale = 1, so q is exact for ±7.
    w = torch.zeros(1, 128)
    w[0, 0] = 7.0
    w[0, 1] = -7.0
    packed, scales = quantize_and_pack(w)
    word = int(packed[0, 0].item()) & 0xFFFFFFFF
    n0 = word & 0xF
    n1 = (word >> 4) & 0xF
    assert n0 == 7
    assert n1 == 9  # two's complement of -7
    assert scales.shape == (1, 1)
    assert abs(float(scales[0, 0]) - 1.0) < 1e-3


def test_zero_group_does_not_nan():
    w = torch.zeros(2, 128)
    packed, scales = quantize_and_pack(w)
    recon = dequantize(packed, scales)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(scales.float()).all()
