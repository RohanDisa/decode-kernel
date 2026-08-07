#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace decode_kernel {

constexpr int kGroupSize = 128;
constexpr int kNibblesPerWord = 8;
constexpr int kWeightsPerInt4 = 32;

enum class KernelVersion {
  V0 = 0,  // one thread per row, scalar loads
  V1 = 1,  // one warp per row, shuffle reduction
  V2 = 2,  // vectorized int4 packed-weight loads
  V3 = 3,  // fused dequant in registers, scales in smem
  V4 = 4,  // split-K across the reduction dimension
  V5 = 5,  // occupancy / register / unroll tuning
};

struct GemvLaunchConfig {
  KernelVersion version = KernelVersion::V5;
  int split_k = 4;           // used by v4; ignored otherwise
  bool reuse_weights = false;  // smem-resident W for batch sweep
  int warps_per_row = 1;     // v1 only: split K across warps in one row
};

// packed:  [M, K/8] uint32, 8 consecutive signed nibbles per word, low first
// scales:  [M, K/128] fp16, symmetric group scale
// x:       [B, K] fp16
// y:       [B, M] fp16
// workspace: fp32 [B, M], required for v4 when split_k > 1
void w4a16_gemv(const uint32_t* packed, const __half* scales, const __half* x,
                __half* y, int M, int K, int B, const GemvLaunchConfig& cfg,
                float* workspace, cudaStream_t stream);

// STREAM-style HBM probes. `partials` is one float per block for the read kernel.
void d2d_copy(const void* src, void* dst, size_t bytes, cudaStream_t stream);
void stream_read(const void* src, float* partials, size_t bytes, int blocks,
                 cudaStream_t stream);
void stream_write(void* dst, size_t bytes, int blocks, cudaStream_t stream);
int stream_recommended_blocks();

// Device stamp written by the GEMV that actually launched. -1 if none.
int last_kernel_stamp();

}  // namespace decode_kernel
