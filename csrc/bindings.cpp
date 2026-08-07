#include "decode_kernel/gemv_w4a16.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

void check_gemv_inputs(const torch::Tensor& packed, const torch::Tensor& scales,
                       const torch::Tensor& x) {
  TORCH_CHECK(packed.is_cuda() && scales.is_cuda() && x.is_cuda(),
              "packed, scales, and x must be CUDA tensors");
  TORCH_CHECK(packed.scalar_type() == torch::kInt32,
              "packed must be int32 (uint32 bit pattern)");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be fp16");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
  TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2 && x.dim() == 2,
              "packed [M,K/8], scales [M,K/128], x [B,K]");
  TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous() &&
                  x.is_contiguous(),
              "packed, scales, and x must be contiguous");

  const int64_t M = packed.size(0);
  const int64_t packed_cols = packed.size(1);
  const int64_t K = packed_cols * decode_kernel::kNibblesPerWord;
  TORCH_CHECK(K % decode_kernel::kGroupSize == 0,
              "K must be divisible by group size 128");
  TORCH_CHECK(scales.size(0) == M &&
                  scales.size(1) == K / decode_kernel::kGroupSize,
              "scales shape must be [M, K/128]");
  TORCH_CHECK(x.size(1) == K, "x.shape[1] must equal K");
}

}  // namespace

torch::Tensor w4a16_gemv(torch::Tensor packed, torch::Tensor scales,
                         torch::Tensor x, int version, int split_k,
                         bool reuse_weights, int warps_per_row) {
  check_gemv_inputs(packed, scales, x);
  TORCH_CHECK(version >= 0 && version <= 5, "version must be 0..5");
  TORCH_CHECK(split_k >= 1 && split_k <= 32, "split_k must be in [1, 32]");
  TORCH_CHECK(warps_per_row >= 1 && warps_per_row <= 8,
              "warps_per_row must be in [1, 8]");
  if (warps_per_row > 1) {
    TORCH_CHECK(version == 1 && !reuse_weights,
                "warps_per_row > 1 is implemented for streaming v1 only");
  }

  const int M = static_cast<int>(packed.size(0));
  const int K =
      static_cast<int>(packed.size(1) * decode_kernel::kNibblesPerWord);
  const int B = static_cast<int>(x.size(0));

  auto y = torch::empty({B, M}, x.options());

  decode_kernel::GemvLaunchConfig cfg;
  cfg.version = static_cast<decode_kernel::KernelVersion>(version);
  cfg.split_k = split_k;
  cfg.reuse_weights = reuse_weights;
  cfg.warps_per_row = warps_per_row;

  torch::Tensor workspace;
  float* ws_ptr = nullptr;
  if (!reuse_weights && cfg.version == decode_kernel::KernelVersion::V4) {
    workspace = torch::zeros({B, M}, x.options().dtype(torch::kFloat32));
    ws_ptr = workspace.data_ptr<float>();
  }

  decode_kernel::w4a16_gemv(
      reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
      reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), M, K, B, cfg, ws_ptr,
      at::cuda::getCurrentCUDAStream());

  return y;
}

void d2d_copy(torch::Tensor src, torch::Tensor dst) {
  TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src and dst must be CUDA");
  TORCH_CHECK(src.nbytes() == dst.nbytes(), "src/dst size mismatch");
  TORCH_CHECK(src.nbytes() % 16 == 0, "copy size must be a multiple of 16");
  decode_kernel::d2d_copy(src.data_ptr(), dst.data_ptr(), src.nbytes(),
                          at::cuda::getCurrentCUDAStream());
}

void stream_read(torch::Tensor src, torch::Tensor partials) {
  TORCH_CHECK(src.is_cuda() && partials.is_cuda(), "src and partials must be CUDA");
  TORCH_CHECK(src.nbytes() % 16 == 0, "buffer size must be a multiple of 16");
  TORCH_CHECK(partials.scalar_type() == torch::kFloat32, "partials must be fp32");
  TORCH_CHECK(partials.is_contiguous(), "partials must be contiguous");
  const int blocks = static_cast<int>(partials.numel());
  TORCH_CHECK(blocks >= 1, "partials must have one float per block");
  decode_kernel::stream_read(src.data_ptr(), partials.data_ptr<float>(),
                             src.nbytes(), blocks,
                             at::cuda::getCurrentCUDAStream());
}

void stream_write(torch::Tensor dst) {
  TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
  TORCH_CHECK(dst.nbytes() % 16 == 0, "buffer size must be a multiple of 16");
  const int blocks = decode_kernel::stream_recommended_blocks();
  decode_kernel::stream_write(dst.data_ptr(), dst.nbytes(), blocks,
                              at::cuda::getCurrentCUDAStream());
}

int stream_recommended_blocks() {
  return decode_kernel::stream_recommended_blocks();
}

int last_kernel_stamp() { return decode_kernel::last_kernel_stamp(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a16_gemv", &w4a16_gemv, "W4A16 GEMV", py::arg("packed"),
        py::arg("scales"), py::arg("x"), py::arg("version") = 5,
        py::arg("split_k") = 4, py::arg("reuse_weights") = false,
        py::arg("warps_per_row") = 1);
  m.def("d2d_copy", &d2d_copy, "Aligned device-to-device copy", py::arg("src"),
        py::arg("dst"));
  m.def("stream_read", &stream_read,
        "Read-only STREAM sum-reduction (one float partial per block)",
        py::arg("src"), py::arg("partials"));
  m.def("stream_write", &stream_write, "Write-only STREAM fill", py::arg("dst"));
  m.def("stream_recommended_blocks", &stream_recommended_blocks,
        "Block count used by STREAM probes");
  m.def("last_kernel_stamp", &last_kernel_stamp,
        "Stage stamp written by the GEMV that actually launched");
}
