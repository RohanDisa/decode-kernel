# decode-kernel

A hand-written W4A16 GEMV for the LLM decode path: 4-bit grouped weights, fp16 activations, fused dequantization, and no materialization of the fp16 weight matrix at any point.

Decode is memory-bound. The ceiling is HBM bandwidth, not tensor cores, so this is built and measured as a bandwidth problem.

---

## Results

Measured on Tesla T4 and NVIDIA A100-SXM4-40GB via Modal, 2026-09-10. Full tables in [`RESULTS.md`](RESULTS.md).

**4096x4096, batch 1:**

| | T4 | A100 |
|---|---|---|
| Latency | 77.5 us | 27.6 us |
| Achieved read bandwidth | 111 GB/s | 313 GB/s |
| Percent of measured read roof | **41%** | **22%** |

The warp-per-row kernel improved bandwidth utilization **4.1x** over an uncoalesced one-thread-per-row baseline. Coalescing was the entire gap.

At 4096x11008, T4 holds at 41 percent of the read roof, so the result is not specific to a square shape.

## Ceilings

All hit rates are computed against a measured read-only roof, not a datasheet figure and not a device-to-device copy. A copy moves each byte twice; this kernel reads weights and writes a small output vector, so a copy roof understates the available read bandwidth.

| | T4 | A100 |
|---|---|---|
| STREAM read (denominator) | 270 GB/s | 1406 GB/s |
| STREAM copy, read+write | 243 GB/s | 1380 GB/s |
| cuBLAS fp16 GEMM 4096 | 23.5 TFLOP/s | 251 TFLOP/s |
| Ridge point | 87 FLOP/byte | 178 FLOP/byte |

A W4A16 GEMV runs at 3.87 FLOP/byte including scales and vectors, which puts it 22x below the T4 ridge and 46x below the A100 ridge. It is nowhere near compute-bound, which is why tensor cores are not used here and would not help.

## The A100 shortfall

The kernel holds 41 percent of the read roof on T4 but only 22 percent on A100, despite A100 having 5.2x the read bandwidth. The obvious hypothesis is insufficient memory-level parallelism: one warp per row over 4096 rows gives far fewer warps per SM on A100's 108 SMs than on T4's 40.

That was tested and rejected. Running M=11008 on A100 restores warps-per-SM to roughly the T4 figure and leaves utilization at 22.8 percent, against 22.3 percent at 4096 rows. Assigning 2 and 4 warps per row at 4096x4096 made it worse, at 17 percent.

The remaining explanation is dequantization cost in the inner loop. Confirming it directly requires Nsight Compute, which is unavailable on Modal; a load-only ablation would also settle it and has not been run.

## Peer comparison

The peer comparison is Marlin, which was not installed on these runs. Until it is, this kernel has been measured against a bandwidth ceiling and against itself, and not against a production W4A16 implementation.

cuBLAS fp16 GEMV is included in the harness as a reference point, but it is cross-precision and should not be read as a peer result.

---

## Variants tested

Each variant is a separate kernel; device-side stamps confirm distinct code paths.

| Variant | Question being tested | Result |
|---|---|---|
| Scalar row | What is the naive baseline? | Slow. One thread per row leaves loads uncoalesced. |
| Warp-per-row | Does coalescing fix the main bottleneck? | Yes. Best variant, 4.1x over the baseline. |
| Vectorized packed load | Do wider loads reduce overhead? | No. Regressed on both GPUs. |
| Shared-scale variant | Does caching group scales in shared memory help? | No measurable change. |
| Split-K | Does more parallelism help at this shape? | No. Slightly worse at batch 1. |
| Compiler-hint variant | Do unrolling and load hints help? | No change. |

The last four share a 32-wide unpack inner loop, so they are one direction tested several ways rather than four independent attempts. Coalescing was verified intact, which places the cost on the instruction side rather than the access pattern. Register pressure is the leading suspect and has not been confirmed with `-Xptxas -v`.

**Format:** symmetric signed 4-bit, group size 128, eight consecutive quantized values packed low-nibble-first into an int32. Products accumulate fp16 x fp16 into fp32.

**Batch sweep:** run on a weight-reuse path where each warp stages its packed row into shared memory once and streams every batch item. Latency tracks O(B) on both GPUs while cuBLAS stays flat, which reflects re-dequantization cost rather than a memory-to-compute crossover. No crossover batch size is claimed; establishing one requires SM and DRAM counters.

---

## Correctness

Every variant is checked against a PyTorch fp32 dequantize-then-matmul of the same packed weights, not against the original dense matrix. Quantization error is out of scope for a kernel test; rounding and reduction order are in scope.

Tolerance is 1e-2 max absolute error. Activations and scales enter the kernel in fp16 while the reference is fp32, and K runs to thousands of terms, so bitwise equality is neither achievable nor claimed.

```bash
pytest tests/test_quantize.py -q                              # CPU, packing format
pytest tests/test_correctness.py -q                           # GPU, every variant
DK_FULL_TEST=1 DK_FULL_SHAPES=1 pytest tests/test_correctness.py
```

The packing gate runs in CI. The GPU gate runs in the Kaggle and A100 scripts.

---

## Measurement protocol

50 warmup iterations discarded, 1000 timed iterations using CUDA events, median and interquartile range reported. Best-of-N is never used.

L2 is flushed between iterations with a 64 MB fill. At 4096x4096 the fp16 weights are 33 MB, which misses T4's 4 MB L2 but fits A100's 40 MB; without the flush the A100 numbers would measure cache, not HBM. STREAM roofs use a 512 MB buffer for the same reason.

```bash
python -m bench.ceilings                  # STREAM copy / read / write, cuBLAS GEMM
python -m bench.bench --shapes 4096x4096 --versions 0,1,2,3,4,5
python -m bench.bench --sweep --shapes 4096x4096
python -m bench.apply_hit_rates           # hit rates against the read-only roof
python -m bench.ncu_collect               # --print-only if ncu is unavailable
python -m bench.plot
```

---

## Running it

Requires Linux, the CUDA toolkit, and PyTorch with CUDA.

```bash
pip install -e .
pytest tests/ -q
bash scripts/run_all.sh
```

`setup.py` builds `sm_75` through `sm_90` unless it can detect the current GPU, in which case it builds only that architecture. Override with `DK_CUDA_ARCH=75,80`, or skip the extension entirely with `DK_SKIP_CUDA=1` for CPU packing tests.

Remote GPUs through Modal, same JSON contract:

```bash
pip install modal && modal setup
modal run bench/modal_runner.py --gpu T4 --mode ceilings
modal run bench/modal_runner.py --gpu A100 --mode all
python -m bench.plot
```

Both paths write GPU-tagged JSON to `bench/results/`. Nsight Compute is not available under Modal.

Shapes are taken from real models: 4096x4096, 4096x11008, and 4096x14336 correspond to Llama 7B and 8B attention and MLP projections.

Out of scope: training, tensor cores, activation quantization, multi-GPU, and end-to-end model integration.
