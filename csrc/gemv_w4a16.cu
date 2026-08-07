#include "decode_kernel/gemv_w4a16.h"

#include <algorithm>
#include <cstdio>

namespace decode_kernel {
namespace {

#define DK_LAUNCH_CHECK()                                                      \
  do {                                                                         \
    cudaError_t err_ = cudaGetLastError();                                     \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "decode-kernel launch error: %s (%s:%d)\n",         \
                   cudaGetErrorString(err_), __FILE__, __LINE__);              \
    }                                                                          \
  } while (0)

// Written by whichever GEMV actually launched. Host reads via last_kernel_stamp().
__device__ int d_kernel_stamp = -1;

__device__ __forceinline__ void stamp_kernel(int v) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    d_kernel_stamp = v;
  }
}

__device__ __forceinline__ int s4(uint32_t packed, int n) {
  return static_cast<int>(packed << (28 - (n << 2))) >> 28;
}

__device__ __forceinline__ float dequant_dot8(uint32_t packed, const __half* x,
                                              float scale) {
  float acc = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    acc += scale * static_cast<float>(s4(packed, n)) * __half2float(x[n]);
  }
  return acc;
}

__device__ __forceinline__ float dequant_dot32(int4 pw, const __half* x,
                                               float scale) {
  float acc = 0.f;
  acc += dequant_dot8(static_cast<uint32_t>(pw.x), x + 0, scale);
  acc += dequant_dot8(static_cast<uint32_t>(pw.y), x + 8, scale);
  acc += dequant_dot8(static_cast<uint32_t>(pw.z), x + 16, scale);
  acc += dequant_dot8(static_cast<uint32_t>(pw.w), x + 24, scale);
  return acc;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ int4 load_int4(const uint32_t* ptr) {
  return *reinterpret_cast<const int4*>(ptr);
}

__device__ __forceinline__ int4 load_int4_ldg(const uint32_t* ptr) {
  return __ldg(reinterpret_cast<const int4*>(ptr));
}

// ---------------------------------------------------------------------------
// v0: one thread per output row, scalar nibble loads. Intentionally terrible.
// ---------------------------------------------------------------------------
__global__ void gemv_v0(const uint32_t* __restrict__ packed,
                        const __half* __restrict__ scales,
                        const __half* __restrict__ x, __half* __restrict__ y,
                        int M, int K, int B) {
  stamp_kernel(0);
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;

  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      const uint32_t word = row_w[k >> 3];
      const int n = k & 7;
      const float scale = __half2float(row_s[k >> 7]);
      acc += scale * static_cast<float>(s4(word, n)) * __half2float(xb[k]);
    }
    y[static_cast<size_t>(b) * M + row] = __float2half(acc);
  }
}

// ---------------------------------------------------------------------------
// v1: one warp per row, coalesced uint32 loads, shuffle reduction.
// ---------------------------------------------------------------------------
__global__ void gemv_v1(const uint32_t* __restrict__ packed,
                        const __half* __restrict__ scales,
                        const __half* __restrict__ x, __half* __restrict__ y,
                        int M, int K, int B) {
  stamp_kernel(1);
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= M) return;

  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
    for (int pk = lane; pk < packed_cols; pk += 32) {
      const uint32_t word = row_w[pk];
      const float scale = __half2float(row_s[(pk << 3) >> 7]);
      acc += dequant_dot8(word, xb + (pk << 3), scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      y[static_cast<size_t>(b) * M + row] = __float2half(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// v1, K split across warps of one row. One block per row.
// Diagnostic: more warps on GPUs where M / #SM underfills the memory pipe.
// ---------------------------------------------------------------------------
__global__ void gemv_v1_multiwarp(const uint32_t* __restrict__ packed,
                                  const __half* __restrict__ scales,
                                  const __half* __restrict__ x,
                                  __half* __restrict__ y, int M, int K, int B) {
  const int wpr = blockDim.x >> 5;
  stamp_kernel(10 + wpr);  // 12 or 14 for 2/4 warps per row
  const int row = blockIdx.x;
  if (row >= M) return;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  __shared__ float smem_red[8];

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
    for (int pk = warp * 32 + lane; pk < packed_cols; pk += wpr * 32) {
      const uint32_t word = row_w[pk];
      const float scale = __half2float(row_s[(pk << 3) >> 7]);
      acc += dequant_dot8(word, xb + (pk << 3), scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) smem_red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
      float tot = (lane < wpr) ? smem_red[lane] : 0.f;
      tot = warp_reduce_sum(tot);
      if (lane == 0) {
        y[static_cast<size_t>(b) * M + row] = __float2half(tot);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// v2: 128-bit packed-weight loads (32 weights per thread per iter).
//
// Access: lane v loads int4 at row_w + (v << 2). Consecutive lanes therefore
// issue consecutive 16-byte transactions — coalesced. x is 32 fp16s at
// k0 = v << 5 (64 B/lane). A 32-wide unpack that broke coalescing is not
// what this loop does. Matching v2/v3/v5 latency is the shared dequant_dot32
// body, not a dispatch fallthrough.
// ---------------------------------------------------------------------------
__global__ void gemv_v2(const uint32_t* __restrict__ packed,
                        const __half* __restrict__ scales,
                        const __half* __restrict__ x, __half* __restrict__ y,
                        int M, int K, int B) {
  stamp_kernel(2);
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= M) return;

  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const int nvec = K >> 5;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
    for (int v = lane; v < nvec; v += 32) {
      const int4 pw = load_int4(row_w + (v << 2));
      const int k0 = v << 5;
      const float scale = __half2float(row_s[k0 >> 7]);
      acc += dequant_dot32(pw, xb + k0, scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      y[static_cast<size_t>(b) * M + row] = __float2half(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// v3: v2 + group scales staged in shared memory, dequant in registers.
// ---------------------------------------------------------------------------
__global__ void gemv_v3(const uint32_t* __restrict__ packed,
                        const __half* __restrict__ scales,
                        const __half* __restrict__ x, __half* __restrict__ y,
                        int M, int K, int B) {
  stamp_kernel(3);
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= M) return;

  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const int nvec = K >> 5;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  extern __shared__ __half smem_scales[];
  __half* my_scales = smem_scales + warp_in_block * groups;
  for (int g = lane; g < groups; g += 32) {
    my_scales[g] = row_s[g];
  }
  __syncwarp();

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
    for (int v = lane; v < nvec; v += 32) {
      const int4 pw = load_int4(row_w + (v << 2));
      const int k0 = v << 5;
      const float scale = __half2float(my_scales[k0 >> 7]);
      acc += dequant_dot32(pw, xb + k0, scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      y[static_cast<size_t>(b) * M + row] = __float2half(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// v4: split-K. Each block-y owns a group-aligned K slice; partials atomicAdd
// into an fp32 workspace. Helps at batch 1 (more CTAs on the memory pipe),
// hurts once the problem already saturates HBM.
// ---------------------------------------------------------------------------
__global__ void gemv_v4(const uint32_t* __restrict__ packed,
                        const __half* __restrict__ scales,
                        const __half* __restrict__ x, float* __restrict__ acc_out,
                        int M, int K, int B) {
  stamp_kernel(4);
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= M) return;

  const int groups = K >> 7;
  const int splits = gridDim.y;
  const int split = blockIdx.y;
  const int groups_per_split = (groups + splits - 1) / splits;
  const int g0 = split * groups_per_split;
  if (g0 >= groups) return;
  const int g1 = min(g0 + groups_per_split, groups);

  const int packed_cols = K >> 3;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  const int pk0 = g0 << 4;  // 128 weights = 16 uint32
  const int pk1 = g1 << 4;
  const int v0 = pk0 >> 2;  // int4 index
  const int v1 = pk1 >> 2;

  extern __shared__ __half smem_scales[];
  __half* my_scales = smem_scales + warp_in_block * groups;
  for (int g = g0 + lane; g < g1; g += 32) {
    my_scales[g] = row_s[g];
  }
  __syncwarp();

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
    for (int v = v0 + lane; v < v1; v += 32) {
      const int4 pw = load_int4(row_w + (v << 2));
      const int k0 = v << 5;
      const float scale = __half2float(my_scales[k0 >> 7]);
      acc += dequant_dot32(pw, xb + k0, scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      atomicAdd(&acc_out[static_cast<size_t>(b) * M + row], acc);
    }
  }
}

// ---------------------------------------------------------------------------
// v5: v3 plus launch_bounds, __ldg, unrolled k-stride, vector-friendly x.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 8)
gemv_v5(const uint32_t* __restrict__ packed, const __half* __restrict__ scales,
        const __half* __restrict__ x, __half* __restrict__ y, int M, int K,
        int B) {
  stamp_kernel(5);
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= M) return;

  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const int nvec = K >> 5;
  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  extern __shared__ __half smem_scales[];
  __half* my_scales = smem_scales + warp_in_block * groups;
  for (int g = lane; g < groups; g += 32) {
    my_scales[g] = __ldg(row_s + g);
  }
  __syncwarp();

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
#pragma unroll 4
    for (int v = lane; v < nvec; v += 32) {
      const int4 pw = load_int4_ldg(row_w + (v << 2));
      const int k0 = v << 5;
      const float scale = __half2float(my_scales[k0 >> 7]);
      acc += dequant_dot32(pw, xb + k0, scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      y[static_cast<size_t>(b) * M + row] = __float2half(acc);
    }
  }
}

// ---------------------------------------------------------------------------
// Weight-reuse path for the batch-size sweep.
// Stage each warp's packed row into smem once, then stream all batch items.
// HBM traffic is W once + X + Y, so arithmetic intensity scales with B.
// ---------------------------------------------------------------------------
__global__ void gemv_reuse(const uint32_t* __restrict__ packed,
                           const __half* __restrict__ scales,
                           const __half* __restrict__ x, __half* __restrict__ y,
                           int M, int K, int B) {
  stamp_kernel(15);
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int row = blockIdx.x * warps_per_block + warp_in_block;
  if (row >= M) return;

  const int packed_cols = K >> 3;
  const int groups = K >> 7;
  const int nvec = K >> 5;

  extern __shared__ char smem_raw[];
  uint32_t* smem_w = reinterpret_cast<uint32_t*>(smem_raw);
  __half* smem_s =
      reinterpret_cast<__half*>(smem_w + warps_per_block * packed_cols);

  uint32_t* my_w = smem_w + warp_in_block * packed_cols;
  __half* my_s = smem_s + warp_in_block * groups;

  const uint32_t* row_w = packed + static_cast<size_t>(row) * packed_cols;
  const __half* row_s = scales + static_cast<size_t>(row) * groups;

  for (int i = lane; i < packed_cols; i += 32) {
    my_w[i] = __ldg(row_w + i);
  }
  for (int g = lane; g < groups; g += 32) {
    my_s[g] = __ldg(row_s + g);
  }
  __syncwarp();

  for (int b = 0; b < B; ++b) {
    const __half* xb = x + static_cast<size_t>(b) * K;
    float acc = 0.f;
#pragma unroll 4
    for (int v = lane; v < nvec; v += 32) {
      const int4 pw = load_int4(my_w + (v << 2));
      const int k0 = v << 5;
      const float scale = __half2float(my_s[k0 >> 7]);
      acc += dequant_dot32(pw, xb + k0, scale);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      y[static_cast<size_t>(b) * M + row] = __float2half(acc);
    }
  }
}

__global__ void fp32_to_fp16(const float* __restrict__ in,
                             __half* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

__global__ void d2d_copy_kernel(const float4* __restrict__ src,
                                float4* __restrict__ dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

int warps_per_block_for(int K, bool reuse) {
  if (reuse) {
    // packed row is K/2 bytes; keep the block under ~48 KB smem.
    const int row_bytes = (K >> 1) + (K / kGroupSize) * static_cast<int>(sizeof(__half));
    return row_bytes > 4096 ? 4 : 8;
  }
  return 4;  // 128 threads; matches v5 launch_bounds
}

void launch_streaming(KernelVersion ver, const uint32_t* packed,
                      const __half* scales, const __half* x, __half* y,
                      float* workspace, int M, int K, int B, int split_k,
                      int warps_per_row, cudaStream_t stream) {
  const int wpb = warps_per_block_for(K, false);
  const int threads = wpb * 32;
  const int groups = K / kGroupSize;
  const size_t smem_scales = static_cast<size_t>(wpb) * groups * sizeof(__half);

  if (ver == KernelVersion::V0) {
    const int t = 256;
    const int blocks = (M + t - 1) / t;
    gemv_v0<<<blocks, t, 0, stream>>>(packed, scales, x, y, M, K, B);
    DK_LAUNCH_CHECK();
    return;
  }

  if (ver == KernelVersion::V1 && warps_per_row > 1) {
    const int wpr = std::max(2, std::min(warps_per_row, 8));
    gemv_v1_multiwarp<<<M, wpr * 32, 0, stream>>>(packed, scales, x, y, M, K,
                                                   B);
    DK_LAUNCH_CHECK();
    return;
  }

  const int blocks = (M + wpb - 1) / wpb;

  switch (ver) {
    case KernelVersion::V1:
      gemv_v1<<<blocks, threads, 0, stream>>>(packed, scales, x, y, M, K, B);
      break;
    case KernelVersion::V2:
      gemv_v2<<<blocks, threads, 0, stream>>>(packed, scales, x, y, M, K, B);
      break;
    case KernelVersion::V3:
      gemv_v3<<<blocks, threads, smem_scales, stream>>>(packed, scales, x, y, M,
                                                        K, B);
      break;
    case KernelVersion::V4: {
      const int splits = std::max(split_k, 1);
      dim3 grid(blocks, splits);
      gemv_v4<<<grid, threads, smem_scales, stream>>>(packed, scales, x,
                                                      workspace, M, K, B);
      const int n = M * B;
      fp32_to_fp16<<<(n + 255) / 256, 256, 0, stream>>>(workspace, y, n);
      break;
    }
    case KernelVersion::V5:
      gemv_v5<<<blocks, threads, smem_scales, stream>>>(packed, scales, x, y, M,
                                                        K, B);
      break;
    default:
      break;
  }
  DK_LAUNCH_CHECK();
}

constexpr int kStreamThreads = 256;
constexpr int kReadAccs = 8;

__global__ void stream_read_kernel(const float4* __restrict__ src,
                                   float* __restrict__ partials, size_t n) {
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  float ax[kReadAccs] = {};
  float ay[kReadAccs] = {};
  float az[kReadAccs] = {};
  float aw[kReadAccs] = {};

  size_t i = tid;
  const size_t step = stride * kReadAccs;
  for (; i + (kReadAccs - 1) * stride < n; i += step) {
#pragma unroll
    for (int k = 0; k < kReadAccs; ++k) {
      const float4 v = src[i + static_cast<size_t>(k) * stride];
      ax[k] += v.x;
      ay[k] += v.y;
      az[k] += v.z;
      aw[k] += v.w;
    }
  }
  for (; i < n; i += stride) {
    const float4 v = src[i];
    ax[0] += v.x;
    ay[0] += v.y;
    az[0] += v.z;
    aw[0] += v.w;
  }

  float local = 0.f;
#pragma unroll
  for (int k = 0; k < kReadAccs; ++k) {
    local += ax[k] + ay[k] + az[k] + aw[k];
  }

  __shared__ float smem[kStreamThreads];
  smem[threadIdx.x] = local;
  __syncthreads();
  for (int s = kStreamThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];
}

__global__ void stream_write_kernel(float4* __restrict__ dst, size_t n) {
  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const float4 z = make_float4(1.f, 1.f, 1.f, 1.f);
  for (size_t i = tid; i < n; i += stride) {
    dst[i] = z;
  }
}

void reset_kernel_stamp(cudaStream_t stream) {
  const int neg = -1;
  cudaMemcpyToSymbolAsync(d_kernel_stamp, &neg, sizeof(int), 0,
                          cudaMemcpyHostToDevice, stream);
}

int last_kernel_stamp_impl() {
  int h = -1;
  cudaDeviceSynchronize();
  cudaMemcpyFromSymbol(&h, d_kernel_stamp, sizeof(int));
  return h;
}

}  // namespace

void w4a16_gemv(const uint32_t* packed, const __half* scales, const __half* x,
                __half* y, int M, int K, int B, const GemvLaunchConfig& cfg,
                float* workspace, cudaStream_t stream) {
  if (M <= 0 || K <= 0 || B <= 0) return;

  reset_kernel_stamp(stream);

  if (cfg.reuse_weights) {
    const int wpb = warps_per_block_for(K, true);
    const int threads = wpb * 32;
    const int blocks = (M + wpb - 1) / wpb;
    const int packed_cols = K / kNibblesPerWord;
    const int groups = K / kGroupSize;
    const size_t smem =
        static_cast<size_t>(wpb) * packed_cols * sizeof(uint32_t) +
        static_cast<size_t>(wpb) * groups * sizeof(__half);
    gemv_reuse<<<blocks, threads, smem, stream>>>(packed, scales, x, y, M, K, B);
    DK_LAUNCH_CHECK();
    return;
  }

  launch_streaming(cfg.version, packed, scales, x, y, workspace, M, K, B,
                   cfg.split_k, cfg.warps_per_row, stream);
}

void d2d_copy(const void* src, void* dst, size_t bytes, cudaStream_t stream) {
  const size_t n = bytes / sizeof(float4);
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  d2d_copy_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const float4*>(src), static_cast<float4*>(dst), n);
  DK_LAUNCH_CHECK();
}

int stream_recommended_blocks() {
  int sms = 1;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int blocks = sms * 32;
  if (blocks < 1024) blocks = 1024;
  if (blocks > 8192) blocks = 8192;
  return blocks;
}

void stream_read(const void* src, float* partials, size_t bytes, int blocks,
                 cudaStream_t stream) {
  const size_t n = bytes / sizeof(float4);
  if (blocks < 1) blocks = stream_recommended_blocks();
  stream_read_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const float4*>(src), partials, n);
  DK_LAUNCH_CHECK();
}

void stream_write(void* dst, size_t bytes, int blocks, cudaStream_t stream) {
  const size_t n = bytes / sizeof(float4);
  if (blocks < 1) blocks = stream_recommended_blocks();
  stream_write_kernel<<<blocks, 256, 0, stream>>>(static_cast<float4*>(dst),
                                                   n);
  DK_LAUNCH_CHECK();
}

int last_kernel_stamp() { return last_kernel_stamp_impl(); }

}  // namespace decode_kernel
