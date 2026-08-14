"""Launch each GEMV stage once and record the device stamp it wrote.

Confirms the version argument reaches a distinct kernel. Expected stamps:
  v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, reuse=15,
  v1 with 2/4 warps per row = 12 / 14.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bench.common import RESULTS_DIR, collect_meta, save_json
from decode_kernel.quantize import quantize_and_pack


def _launch(packed, scales, x, **kwargs) -> int:
    from decode_kernel import gemv
    from decode_kernel import _cuda

    gemv(packed, scales, x, **kwargs)
    torch.cuda.synchronize()
    return int(_cuda.last_kernel_stamp())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    m, k, b = 256, 512, 1
    w = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.05
    x = torch.randn(b, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w.float())

    stamps = {}
    for v in range(6):
        stamps[f"v{v}"] = _launch(
            packed, scales, x, version=v, split_k=4 if v == 4 else 1
        )
    stamps["v5_reuse"] = _launch(
        packed, scales, x, version=5, reuse_weights=True
    )
    stamps["v1_wpr2"] = _launch(
        packed, scales, x, version=1, warps_per_row=2
    )
    stamps["v1_wpr4"] = _launch(
        packed, scales, x, version=1, warps_per_row=4
    )

    expected = {
        "v0": 0,
        "v1": 1,
        "v2": 2,
        "v3": 3,
        "v4": 4,
        "v5": 5,
        "v5_reuse": 15,
        "v1_wpr2": 12,
        "v1_wpr4": 14,
    }
    mismatches = {k: {"got": stamps[k], "expected": expected[k]} for k in expected if stamps[k] != expected[k]}
    distinct_v025 = len({stamps[f"v{v}"] for v in range(6)}) == 6

    payload = {
        "meta": {**collect_meta().__dict__},
        "stamps": stamps,
        "expected": expected,
        "mismatches": mismatches,
        "distinct_v0_v5": distinct_v025,
        "dispatch_ok": not mismatches,
        "note": (
            "Identical v2/v3/v5 *latencies* are not a dispatch bug if stamps "
            "are 2, 3, 5: those kernels share dequant_dot32 on int4 loads."
        ),
    }
    gpu = payload["meta"]["gpu"]["name"].replace(" ", "_")
    out = args.out or (RESULTS_DIR / f"stamp_{gpu}.json")
    save_json(out, payload)
    print("stamps:", stamps)
    print("distinct v0-v5:", distinct_v025)
    if mismatches:
        print("MISMATCHES:", mismatches)
    else:
        print("dispatch ok — each launch wrote its own stamp")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
