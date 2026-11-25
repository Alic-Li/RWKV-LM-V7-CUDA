#include "embedding.hpp"
#include "gpu.hpp"
#ifdef RWKV_ROCM
#include <hipcub/device/device_radix_sort.hpp>
#else
#include <cub/device/device_radix_sort.cuh>
#endif

namespace rwkv {
namespace {
#ifdef RWKV_ROCM
using hipcub::DeviceRadixSort;
#else
using cub::DeviceRadixSort;
#endif
constexpr int channels_per_tile = 32;

__global__ void row_numbers(int *positions, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows)
    positions[i] = i;
}

// Stable sorting keeps contributions to a token in input order. Each thread group
// owns a token and channel tile, so repeated indices never race.
template <bool small>
__global__ void reduce_tokens(const int *tokens, const int *positions,
                              const __nv_bfloat16 *upstream, float *gradient,
                              int rows, int channels) {
  const int begin = blockIdx.x * blockDim.y + threadIdx.y;
  if (begin >= rows || (begin && tokens[begin - 1] == tokens[begin]))
    return;
  const int channel = blockIdx.y * channels_per_tile + threadIdx.x;
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

std::size_t embedding_sort_bytes(cudaStream_t stream, int rows) {
  std::size_t bytes = 0;
  int *p = nullptr;
  CUDA(DeviceRadixSort::SortPairs(nullptr, bytes, static_cast<const int *>(p),
                                  p, static_cast<const int *>(p), p, rows, 0,
                                  31, stream));
  return bytes;
}

void embedding_gradient(Arena &arena, std::size_t scratch_bytes,
                        cudaStream_t stream, int rows, int channels,
                        const int *tokens, const __nv_bfloat16 *upstream,
                        float *gradient) {
  auto sorted = arena.take<int>(rows), positions = arena.take<int>(rows);
  auto indices = arena.take<int>(rows);
  row_numbers<<<(rows - 1) / 256 + 1, 256, 0, stream>>>(indices.get(), rows);
  auto scratch = arena.take<unsigned char>(scratch_bytes);
  CUDA(DeviceRadixSort::SortPairs(scratch.get(), scratch_bytes, tokens,
                                  sorted.get(), indices.get(), positions.get(),
                                  rows, 0, 31, stream));
  const dim3 blocks((rows - 1) / 8 + 1, (channels - 1) / channels_per_tile + 1);
  const dim3 threads(channels_per_tile, 8);
  if (rows <= 3072)
    reduce_tokens<true><<<blocks, threads, 0, stream>>>(
        sorted.get(), positions.get(), upstream, gradient, rows, channels);
  else
    reduce_tokens<false><<<blocks, threads, 0, stream>>>(
        sorted.get(), positions.get(), upstream, gradient, rows, channels);
}
} // namespace rwkv
