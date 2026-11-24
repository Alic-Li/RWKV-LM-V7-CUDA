#include "embedding.hpp"
#include "gpu.hpp"
#include <cub/device/device_radix_sort.cuh>

namespace rwkv {
namespace {
__global__ void row_numbers(int *positions, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows)
    positions[i] = i;
}

// Stable sorting keeps contributions to a token in input order. Each warp
// owns a token and channel tile, so repeated indices never race.
template <bool small>
__global__ void reduce_tokens(const int *tokens, const int *positions,
                              const __nv_bfloat16 *upstream, float *gradient,
                              int rows, int channels) {
  const int begin = blockIdx.x * blockDim.y + threadIdx.y;
  if (begin >= rows || (begin && tokens[begin - 1] == tokens[begin]))
    return;
  const int channel = blockIdx.y * warpSize + threadIdx.x;
  if (channel >= channels)
    return;
  const int token = tokens[begin];
  int i = begin;
  float sum = 0;
  do {
    const int tile = positions[i] / 32;
    float partial = 0;
    int count = 0;
    do {
      partial += __bfloat162float(
          upstream[std::size_t(positions[i]) * channels + channel]);
      ++i;
      ++count;
    } while (i < rows && tokens[i] == token &&
             (small ? positions[i] / 32 == tile : count < 10));
    // ATen writes a BF16 partial for each group of 32 source positions on
    // small inputs. The large-input path sums groups of ten matches in FP32.
    if constexpr (small)
      sum = __bfloat162float(__float2bfloat16_rn(
          sum + __bfloat162float(__float2bfloat16_rn(partial))));
    else
      sum += partial;
  } while (i < rows && tokens[i] == token);
  gradient[std::size_t(token) * channels + channel] =
      __bfloat162float(__float2bfloat16_rn(sum));
}
} // namespace

void embedding_gradient(cudaStream_t stream, int rows, int channels,
                        const int *tokens, const __nv_bfloat16 *upstream,
                        float *gradient) {
  Buffer<int> sorted(stream, rows), positions(stream, rows);
  Buffer<int> indices(stream, rows);
  row_numbers<<<(rows - 1) / 256 + 1, 256, 0, stream>>>(indices.get(), rows);
  std::size_t bytes = 0;
  auto sort = [&](void *scratch) {
    CUDA(cub::DeviceRadixSort::SortPairs(scratch, bytes, tokens, sorted.get(),
                                         indices.get(), positions.get(), rows,
                                         0, 31, stream));
  };
  sort(nullptr);
  Buffer<unsigned char> scratch(stream, bytes);
  sort(scratch.get());
  const dim3 blocks((rows - 1) / 8 + 1, (channels - 1) / 32 + 1);
  const dim3 threads(32, 8);
  if (rows <= 3072)
    reduce_tokens<true><<<blocks, threads, 0, stream>>>(
        sorted.get(), positions.get(), upstream, gradient, rows, channels);
  else
    reduce_tokens<false><<<blocks, threads, 0, stream>>>(
        sorted.get(), positions.get(), upstream, gradient, rows, channels);
}
} // namespace rwkv
