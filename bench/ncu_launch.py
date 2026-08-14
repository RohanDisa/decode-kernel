"""Tiny launcher so `ncu` can attach to a single kernel."""

from __future__ import annotations

import argparse

import torch

from decode_kernel import gemv
from decode_kernel.quantize import quantize_and_pack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, default=5)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()

    w = torch.randn(args.m, args.k, device="cuda") * 0.05
    x = torch.randn(args.batch, args.k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w)
    for _ in range(5):
        gemv(
            packed,
            scales,
            x,
            version=args.version,
            split_k=4 if args.version == 4 else 1,
            reuse_weights=args.reuse,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
