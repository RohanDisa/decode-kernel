"""Establish hardware ceilings *before* optimizing the GEMV.

Three STREAM-style probes, all over a buffer well past L2 (default 512 MB,
which clears A100's 40 MB L2):

- copy: device-to-device, each byte is read once and written once (factor 2)
- read: grid-stride sum-reduction, read-only (factor 1)
- write: grid-stride fill, write-only (factor 1)

A decode GEMV is read-dominated (weights in, tiny Y out), so hit rates use
the read-only figure as the denominator. Copy is kept for reference; using it
as the GEMV roof understates bandwidth and overstates every %HBM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bench.common import (
    RESULTS_DIR,
    collect_meta,
    make_l2_flush,
    save_json,
    time_cuda,
)

# 512 MB clears T4 (4 MB) and A100 (40 MB) L2 with plenty of margin.
DEFAULT_STREAM_BYTES = 512 << 20


def _measure(fn, nbytes: int, factor: float, warmup: int, iters: int, flush) -> dict:
    stats = time_cuda(fn, warmup=warmup, iters=iters, flush=flush, flush_every=False)
    seconds = stats["median_us"] / 1e6
    gb = (factor * nbytes) / 1e9
    return {"nbytes": nbytes, "bytes_factor": factor, "gbps": gb / seconds, **stats}


def measure_copy(nbytes: int, warmup: int, iters: int, flush) -> dict:
    from decode_kernel import _cuda

    src = torch.empty(nbytes, device="cuda", dtype=torch.uint8)
    dst = torch.empty_like(src)

    def fn():
        _cuda.d2d_copy(src, dst)

    out = _measure(fn, nbytes, 2.0, warmup, iters, flush)
    out["kind"] = "d2d_copy"
    return out


def measure_read(nbytes: int, warmup: int, iters: int, flush) -> dict:
    from decode_kernel import _cuda

    src = torch.empty(nbytes, device="cuda", dtype=torch.uint8)
    src.fill_(1)
    blocks = int(_cuda.stream_recommended_blocks())
    partials = torch.empty(blocks, device="cuda", dtype=torch.float32)

    def fn():
        _cuda.stream_read(src, partials)

    out = _measure(fn, nbytes, 1.0, warmup, iters, flush)
    out["kind"] = "stream_read_sum"
    out["blocks"] = blocks
    # Sink so a future compiler cannot DCE the reduction across the timed region.
    out["partial_checksum"] = float(partials[0].item())
    return out


def measure_write(nbytes: int, warmup: int, iters: int, flush) -> dict:
    from decode_kernel import _cuda

    dst = torch.empty(nbytes, device="cuda", dtype=torch.uint8)

    def fn():
        _cuda.stream_write(dst)

    out = _measure(fn, nbytes, 1.0, warmup, iters, flush)
    out["kind"] = "stream_write_fill"
    return out


def measure_cublas_fp16(n: int = 8192, warmup: int = 20, iters: int = 50) -> dict:
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    flush = make_l2_flush()

    def fn():
        torch.mm(a, b)

    stats = time_cuda(fn, warmup=warmup, iters=iters, flush=flush, flush_every=False)
    seconds = stats["median_us"] / 1e6
    flops = 2.0 * n * n * n
    tflops = (flops / seconds) / 1e12
    return {"n": n, "tflops": tflops, **stats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--bytes", type=int, default=DEFAULT_STREAM_BYTES)
    p.add_argument("--copy-bytes", type=int, default=None,
                   help="Deprecated alias for --bytes")
    p.add_argument("--gemm-n", type=int, default=4096)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    nbytes = args.copy_bytes if args.copy_bytes is not None else args.bytes
    if nbytes % 16 != 0:
        raise SystemExit("--bytes must be a multiple of 16")

    meta = collect_meta()
    flush = make_l2_flush()
    copy = measure_copy(nbytes, warmup=10, iters=50, flush=flush)
    read = measure_read(nbytes, warmup=10, iters=50, flush=flush)
    write = measure_write(nbytes, warmup=10, iters=50, flush=flush)
    gemm = measure_cublas_fp16(n=args.gemm_n)

    read_gbps = read["gbps"]
    copy_gbps = copy["gbps"]
    ridge_read = (gemm["tflops"] * 1e12) / (read_gbps * 1e9)
    ridge_copy = (gemm["tflops"] * 1e12) / (copy_gbps * 1e9)
    vs_copy = (read_gbps / copy_gbps - 1.0) * 100.0 if copy_gbps else None

    payload = {
        "meta": {**meta.__dict__},
        "hbm_copy": copy,
        "hbm_read": read,
        "hbm_write": write,
        "hbm_denominator": "read",
        "cublas_gemm_fp16": gemm,
        "ridge_flop_per_byte": ridge_read,
        "ridge_flop_per_byte_copy": ridge_copy,
        "read_vs_copy_pct": vs_copy,
        "note": (
            "Copy moves each byte twice (read+write). GEMV is read-dominated, so "
            "hit rates and the GEMV ridge use hbm_read. hbm_copy is reference only."
        ),
    }
    gpu = meta.gpu["name"].replace(" ", "_")
    out = args.out or (RESULTS_DIR / f"ceilings_{gpu}.json")
    save_json(out, payload)
    print(f"GPU: {meta.gpu['name']}")
    print(f"STREAM buffer: {nbytes / (1 << 20):.0f} MiB")
    print(f"HBM copy  (read+write, factor 2): {copy_gbps:.1f} GB/s  [{copy['kind']}]")
    print(f"HBM read  (sum-reduction, factor 1): {read_gbps:.1f} GB/s  [{read['kind']}]")
    print(f"HBM write (fill, factor 1): {write['gbps']:.1f} GB/s  [{write['kind']}]")
    if vs_copy is not None:
        print(f"Read vs copy: {vs_copy:+.1f}%")
    print(f"cuBLAS fp16 GEMM {args.gemm_n}x{args.gemm_n}: {gemm['tflops']:.2f} TFLOP/s")
    print(f"Ridge (read denom): {ridge_read:.1f} FLOP/byte")
    print(f"Ridge (copy denom, reference): {ridge_copy:.1f} FLOP/byte")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
