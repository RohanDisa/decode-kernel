"""Benchmark harness.

Warmup 50, time 1000 CUDA events, median + IQR, L2 flushed between iterations.
Compares v0-v5 against PyTorch F.linear fp16, cuBLAS-backed torch.mm/mv, and
optionally Marlin if it is installed.

Cross-precision comparisons (fp16 vs W4) are labeled as such. Marlin is the
same-precision peer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from bench.common import (
    PREDICTION_PATH,
    RESULTS_DIR,
    arithmetic_intensity_w4,
    bytes_moved_fp16,
    bytes_moved_w4,
    collect_meta,
    flop_gemv,
    load_json,
    make_l2_flush,
    save_json,
    time_cuda,
)
from decode_kernel.quantize import quantize_and_pack

SHAPES = [(4096, 4096), (4096, 11008), (4096, 14336)]
BATCHES = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]


def _try_marlin():
    try:
        import marlin  # noqa: F401

        return True
    except Exception:
        return False


def bench_one(
    packed,
    scales,
    x,
    w_fp16,
    version: int,
    reuse: bool,
    split_k: int,
    warmup: int,
    iters: int,
    flush,
    warps_per_row: int = 1,
) -> dict:
    from decode_kernel import gemv
    from decode_kernel import _cuda

    def fn():
        gemv(
            packed,
            scales,
            x,
            version=version,
            split_k=split_k,
            reuse_weights=reuse,
            warps_per_row=warps_per_row,
        )

    # Touch once so the first timed call is not a lazy init.
    fn()
    torch.cuda.synchronize()
    stamp = int(_cuda.last_kernel_stamp())
    stats = time_cuda(fn, warmup=warmup, iters=iters, flush=flush)
    launched = int(_cuda.last_kernel_stamp())
    b, k = x.shape
    m = packed.shape[0]
    moved = bytes_moved_w4(m, k, b)
    seconds = stats["median_us"] / 1e6
    impl = f"v{version}" + ("_reuse" if reuse else "")
    if warps_per_row > 1:
        impl = f"v{version}_wpr{warps_per_row}"
    return {
        "impl": impl,
        "version": version,
        "reuse_weights": reuse,
        "split_k": split_k,
        "warps_per_row": warps_per_row,
        "kernel_stamp": launched,
        "launched_version": launched,
        "stamp_after_touch": stamp,
        "precision": "w4a16",
        "gbps": (moved / 1e9) / seconds,
        "tflops": (flop_gemv(m, k, b) / seconds) / 1e12,
        "bytes_moved": moved,
        "ai_flop_per_byte": arithmetic_intensity_w4(m, k, b),
        **stats,
    }


def bench_linear_fp16(w_fp16, x, warmup, iters, flush) -> dict:
    def fn():
        F.linear(x, w_fp16)

    fn()
    torch.cuda.synchronize()
    stats = time_cuda(fn, warmup=warmup, iters=iters, flush=flush)
    b, k = x.shape
    m = w_fp16.shape[0]
    moved = bytes_moved_fp16(m, k, b)
    seconds = stats["median_us"] / 1e6
    return {
        "impl": "pytorch_linear_fp16",
        "precision": "fp16",
        "cross_precision_vs_w4": True,
        "gbps": (moved / 1e9) / seconds,
        "tflops": (flop_gemv(m, k, b) / seconds) / 1e12,
        "bytes_moved": moved,
        **stats,
    }


def bench_cublas_fp16(w_fp16, x, warmup, iters, flush) -> dict:
    # B=1 uses torch.mv (cublasHgemv). Larger B uses torch.mm (GEMM).
    wt = w_fp16  # [M, K]
    if x.shape[0] == 1:

        def fn():
            torch.mv(wt, x.squeeze(0))
    else:

        def fn():
            torch.mm(x, wt.t())

    fn()
    torch.cuda.synchronize()
    stats = time_cuda(fn, warmup=warmup, iters=iters, flush=flush)
    b, k = x.shape
    m = w_fp16.shape[0]
    moved = bytes_moved_fp16(m, k, b)
    seconds = stats["median_us"] / 1e6
    return {
        "impl": "cublas_fp16_gemv" if b == 1 else "cublas_fp16_gemm",
        "precision": "fp16",
        "cross_precision_vs_w4": True,
        "gbps": (moved / 1e9) / seconds,
        "tflops": (flop_gemv(m, k, b) / seconds) / 1e12,
        "bytes_moved": moved,
        **stats,
    }


def run_case(
    m: int,
    k: int,
    batch: int,
    versions: list[int],
    reuse: bool,
    split_k: int,
    warmup: int,
    iters: int,
    with_baselines: bool,
    flush,
    warps_per_row: int = 1,
) -> dict:
    w = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.05
    x = torch.randn(batch, k, device="cuda", dtype=torch.float16)
    packed, scales = quantize_and_pack(w.float())
    results = []
    for v in versions:
        sk = split_k if v == 4 else 1
        use_reuse = reuse and v >= 3
        wpr = warps_per_row if v == 1 else 1
        results.append(
            bench_one(
                packed,
                scales,
                x,
                w,
                v,
                use_reuse,
                sk,
                warmup,
                iters,
                flush,
                warps_per_row=wpr,
            )
        )
    if with_baselines:
        results.append(bench_linear_fp16(w, x, warmup, iters, flush))
        results.append(bench_cublas_fp16(w, x, warmup, iters, flush))
        if _try_marlin():
            results.append(
                {
                    "impl": "marlin",
                    "precision": "w4a16",
                    "skipped": False,
                    "note": "Marlin present but packing conversion not wired; see RESULTS.md",
                }
            )
        else:
            results.append(
                {
                    "impl": "marlin",
                    "precision": "w4a16",
                    "skipped": True,
                    "note": "Marlin not installed. pip install from IST-DASLab/marlin to enable.",
                }
            )
    return {
        "M": m,
        "K": k,
        "batch": batch,
        "kernels": results,
    }


def parse_shapes(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        a, b = part.lower().split("x")
        out.append((int(a), int(b)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", default="4096x4096")
    p.add_argument("--versions", default="0,1,2,3,4,5")
    p.add_argument("--batch", default="1")
    p.add_argument("--split-k", type=int, default=4)
    p.add_argument("--reuse", action="store_true", help="smem-resident W (batch sweep)")
    p.add_argument("--sweep", action="store_true", help="batch 1..256 on v5 reuse")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--quick", action="store_true", help="warmup 5, iters 20")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument(
        "--warps-per-row",
        type=int,
        default=1,
        help="v1 only: split K across this many warps per row",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    if args.quick:
        args.warmup, args.iters = 5, 20

    versions = [int(v) for v in args.versions.split(",") if v]
    shapes = parse_shapes(args.shapes)
    batches = [int(b) for b in args.batch.split(",")]
    if args.sweep:
        versions = [5]
        args.reuse = True
        batches = BATCHES
        if args.shapes == "4096x4096" and len(shapes) == 1:
            pass

    meta = collect_meta()
    flush = make_l2_flush()
    prediction = load_json(PREDICTION_PATH)

    cases = []
    for m, k in shapes:
        for b in batches:
            print(
                f"bench M={m} K={k} B={b} versions={versions} "
                f"reuse={args.reuse} warps_per_row={args.warps_per_row}"
            )
            cases.append(
                run_case(
                    m,
                    k,
                    b,
                    versions,
                    args.reuse,
                    args.split_k,
                    args.warmup,
                    args.iters,
                    with_baselines=not args.no_baselines,
                    flush=flush,
                    warps_per_row=args.warps_per_row,
                )
            )

    payload = {
        "meta": {**meta.__dict__},
        "prediction": prediction,
        "cases": cases,
        "methodology": {
            "warmup": args.warmup,
            "iters": args.iters,
            "timer": "cuda_events",
            "l2_flush_bytes": int(flush.numel() * 4),
            "statistic": "median_and_iqr",
            "clocks_locked": meta.gpu.get("clocks_locked"),
            "warps_per_row": args.warps_per_row,
            "kernel_stamp": "last_kernel_stamp after launch",
        },
    }
    gpu = meta.gpu["name"].replace(" ", "_")
    default_name = "sweep" if args.sweep else "stages"
    out = args.out or (RESULTS_DIR / f"{default_name}_{gpu}.json")
    save_json(out, payload)
    _print_table(cases)
    print(f"wrote {out}")


def _print_table(cases):
    print()
    print(f"{'shape':<18} {'B':>4} {'impl':<22} {'median us':>10} {'GB/s':>8}")
    for case in cases:
        shape = f"{case['M']}x{case['K']}"
        for k in case["kernels"]:
            if "median_us" not in k:
                continue
            print(
                f"{shape:<18} {case['batch']:>4} {k['impl']:<22} "
                f"{k['median_us']:10.1f} {k.get('gbps', 0):8.1f}"
            )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
