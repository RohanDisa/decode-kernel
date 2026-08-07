"""Correctness gate: kernel vs fp32 dequantize-then-matmul.

Tolerance is 1e-2 max abs error. This is *not* vs the original dense matrix;
it is vs the same quantized weights, dequantized and multiplied in fp32.
Residual is fp16 rounding of activations/scales plus reduction order.

Quantized kernels fail silently and plausibly. This test is the whole gate.
"""

from __future__ import annotations

import os

import pytest
import torch

from decode_kernel.quantize import quantize_and_pack
from decode_kernel.reference import linear_fp32, max_abs_err

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)

SHAPES = [(256, 256), (4096, 4096)]
if os.environ.get("DK_FULL_SHAPES") == "1":
    SHAPES += [(4096, 11008), (4096, 14336)]

VERSIONS = [0, 1, 2, 3, 4, 5]
MAX_ABS = 1e-2
N_RANDOM = int(os.environ.get("DK_N_RANDOM", "3"))
if os.environ.get("DK_FULL_TEST") == "1":
    N_RANDOM = 100


def _need_cuda_ext():
    try:
        import decode_kernel._cuda  # noqa: F401
    except Exception as exc:
        pytest.skip(f"CUDA extension not built: {exc}")


def _run(m, k, batch, version, seed, reuse=False):
    _need_cuda_ext()
    from decode_kernel import gemv

    torch.manual_seed(seed)
    # Weight scale similar to a linearized transformer projection.
    w = torch.randn(m, k, device="cuda") * 0.05
    x = torch.randn(batch, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w)
    y = gemv(
        packed,
        scales,
        x,
        version=version,
        split_k=4 if version == 4 else 1,
        reuse_weights=reuse,
    )
    y_ref = linear_fp32(x, packed, scales)
    err = max_abs_err(y, y_ref)
    assert err < MAX_ABS, (
        f"max abs err {err:.4e} >= {MAX_ABS} for v{version} "
        f"M={m} K={k} B={batch} seed={seed} reuse={reuse}"
    )
    return err


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("batch", [1, 4])
def test_versions_against_fp32_reference(version, shape, batch):
    m, k = shape
    for i in range(N_RANDOM):
        _run(m, k, batch, version, seed=1000 + i)


def test_reuse_weights_matches_streaming():
    _need_cuda_ext()
    from decode_kernel import gemv

    torch.manual_seed(1)
    m, k, b = 256, 512, 8
    w = torch.randn(m, k, device="cuda") * 0.05
    x = torch.randn(b, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w)
    y_stream = gemv(packed, scales, x, version=5, reuse_weights=False)
    y_reuse = gemv(packed, scales, x, version=5, reuse_weights=True)
    err = (y_stream.float() - y_reuse.float()).abs().max().item()
    assert err < 1e-3, err


def test_splitk_matches_v3():
    _need_cuda_ext()
    from decode_kernel import gemv

    torch.manual_seed(2)
    m, k = 256, 1024
    w = torch.randn(m, k, device="cuda") * 0.05
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w)
    y3 = gemv(packed, scales, x, version=3)
    y4 = gemv(packed, scales, x, version=4, split_k=4)
    err = (y3.float() - y4.float()).abs().max().item()
    assert err < 1e-3, err


def test_kernel_stamps_are_distinct():
    _need_cuda_ext()
    from decode_kernel import gemv
    from decode_kernel import _cuda

    torch.manual_seed(4)
    m, k = 256, 512
    w = torch.randn(m, k, device="cuda") * 0.05
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w)
    seen = {}
    for v in range(6):
        gemv(packed, scales, x, version=v, split_k=4 if v == 4 else 1)
        seen[v] = int(_cuda.last_kernel_stamp())
    assert seen == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, seen


def test_v1_multiwarp_matches_v1():
    _need_cuda_ext()
    from decode_kernel import gemv
    from decode_kernel import _cuda

    torch.manual_seed(3)
    m, k = 256, 1024
    w = torch.randn(m, k, device="cuda") * 0.05
    x = torch.randn(1, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w)
    y1 = gemv(packed, scales, x, version=1, warps_per_row=1)
    s1 = int(_cuda.last_kernel_stamp())
    y2 = gemv(packed, scales, x, version=1, warps_per_row=2)
    s2 = int(_cuda.last_kernel_stamp())
    y4 = gemv(packed, scales, x, version=1, warps_per_row=4)
    s4 = int(_cuda.last_kernel_stamp())
    assert s1 == 1 and s2 == 12 and s4 == 14, (s1, s2, s4)
    assert (y1.float() - y2.float()).abs().max().item() < 1e-2
    assert (y1.float() - y4.float()).abs().max().item() < 1e-2
