#include "gpu.hpp"
#include "value_bias.hpp"
#include <algorithm>
#include <cstddef>

namespace rwkv {
namespace {
int ceil_div(int n, int d) { return 1 + (n - 1) / d; }

// Match the four row accumulators and pairwise lane reductions of ATen sum.
// One thread handles each channel of ATen's four-channel vector separately.
template <bool final, typename T>
__global__ void reduce_rows(const T *terms, float *output, int rows,
                            int channels) {
  __shared__ float partial[512];
  const unsigned lane = threadIdx.y;
  const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned slot = lane * blockDim.x + threadIdx.x;
  const std::size_t stride = std::size_t(blockDim.y) * gridDim.y;
  float sums[4] = {};
  if (col < channels) {
    std::size_t row = lane + blockDim.y * blockIdx.y;
    if constexpr (final) {
      for (; row < rows; row += stride)
        sums[0] += static_cast<float>(terms[row * channels + col]);
    } else {
      for (; row < rows; row += 4 * stride) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const auto r = row + i * stride;
          if (r < rows)
            sums[i] += static_cast<float>(terms[r * channels + col]);
        }
      }
    }
  }
  partial[slot] = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  for (unsigned offset = blockDim.y / 2; offset; offset /= 2) {
    __syncthreads();
    if (lane < offset)
      partial[slot] += partial[slot + offset * blockDim.x];
  }
  if (lane == 0 && col < channels)
    output[std::size_t(blockIdx.y) * channels + col] = partial[slot];
}
} // namespace

template <typename T>
void value_bias_gradient(cudaStream_t stream, int rows, int channels,
                         int resident_threads, const T *terms,
                         float *gradient) {
  // Model widths are multiples of 64 and padded sequence lengths are multiples
  // of 16.
  const int columns = std::min(128, channels);
  const int lanes = 512 / columns;
  const int groups = ceil_div(channels, columns);
  const int row_lanes = rows >= 16 * lanes ? lanes : 1;
  const int per_lane = ceil_div(rows, row_lanes);
  const int target_blocks = resident_threads / 128;
  int parts = 1;
  if (row_lanes > 1 && per_lane >= 256 && groups <= target_blocks) {
    parts = std::max(
        ceil_div(per_lane, 256),
        std::min(ceil_div(target_blocks, groups), ceil_div(per_lane, 16)));
  }
  const dim3 threads(columns, row_lanes);
  if (parts == 1) {
    reduce_rows<false>
        <<<groups, threads, 0, stream>>>(terms, gradient, rows, channels);
    return;
  }
  // Limit the number of partials for very large batches.
  // CUDA grid.y has a limit of 65535; the row loop can consume the remainder.
  parts = std::min(parts, 65535);
  Buffer<float> partial(stream, std::size_t(parts) * channels);
  reduce_rows<false><<<dim3(groups, parts), threads, 0, stream>>>(
      terms, partial.get(), rows, channels);
  reduce_rows<true><<<groups, threads, 0, stream>>>(partial.get(), gradient,
                                                    parts, channels);
}
template void value_bias_gradient(cudaStream_t, int, int, int, const float *,
                                  float *);
template void value_bias_gradient(cudaStream_t, int, int, int,
                                  const __nv_bfloat16 *, float *);
} // namespace rwkv
