#include "gpu.hpp"
#include "layer_norm.hpp"
#include <algorithm>

namespace rwkv {
namespace {
// Match ATen's vectorized CUDA LayerNorm reduction tree (four warps,
// four consecutive channels per lane). Model widths are multiples of 64.
constexpr int row_threads = 128;
template <typename T>
__device__ T shuffle_down_32(T value, int step) {
#ifdef RWKV_ROCM
  return __shfl_down(value, step, 32);
#else
  return __shfl_down_sync(0xffffffff, value, step);
#endif
}

__device__ float shuffle_xor_32(float value, int step) {
#ifdef RWKV_ROCM
  return __shfl_xor(value, step, 32);
#else
  return __shfl_xor_sync(0xffffffff, value, step);
#endif
}

struct Moments {
  float mean, m2, count;
};

__device__ Moments merge(Moments lower, Moments upper) {
  const float count = upper.count + lower.count;
  if (count == 0)
    return {};
  const float reciprocal = 1.f / count;
  const float a = upper.count * reciprocal, b = lower.count * reciprocal;
  const float delta = lower.mean - upper.mean;
  return {a * upper.mean + b * lower.mean,
          upper.m2 + lower.m2 + delta * delta * upper.count * b, count};
}

__global__ void normalize_rows(int width, const BF *input, const BF *residual,
                               BF *sum, const BF *weight, const BF *bias,
                               BF *out, float *means, float *inv_stds) {
  __shared__ Moments warps[4];
  __shared__ float mean, inv_std;
  const int tid = threadIdx.x;
  const auto offset = size_t(blockIdx.x) * width;
  Moments local{};
  for (int j = tid * 4; j < width; j += row_threads * 4) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto i = offset + j + k;
      BF value = input[i];
      if (residual) {
        value = __float2bfloat16_rn(float(value) + float(residual[i]));
        sum[i] = value;
      }
      const float delta = float(value) - local.mean;
      local.count += 1.f;
      local.mean += delta * (1.f / local.count);
      local.m2 += delta * (float(value) - local.mean);
    }
  }
  for (int step = 16; step; step /= 2) {
    Moments other{shuffle_down_32(local.mean, step),
                  shuffle_down_32(local.m2, step),
                  shuffle_down_32(local.count, step)};
    local = merge(local, other);
  }
  if (tid % 32 == 0)
    warps[tid / 32] = local;
  __syncthreads();
  if (tid == 0) {
    const auto row =
        merge(merge(warps[0], warps[2]), merge(warps[1], warps[3]));
    means[blockIdx.x] = mean = row.mean;
    inv_stds[blockIdx.x] = inv_std = rsqrtf(row.m2 / width + 1e-5f);
  }
  __syncthreads();
  const BF *x = residual ? sum : input;
  for (int j = tid; j < width; j += row_threads)
    out[offset + j] = __float2bfloat16_rn(
        float(weight[j]) * (inv_std * (float(x[offset + j]) - mean)) +
        float(bias[j]));
}

__device__ float warp_sum(float value) {
  for (int step = 16; step; step /= 2)
    value += shuffle_down_32(value, step);
  return value;
}

__global__ void input_gradient(int width, const BF *x, const BF *dy,
                               const BF *weight, const float *means,
                               const float *inv_stds, BF *dx) {
  const int tid = threadIdx.x;
  const auto off = size_t(blockIdx.x) * width;
  const float mean = means[blockIdx.x], r = inv_stds[blockIdx.x];
  float s = 0, p = 0;
  for (int j = tid * 4; j < width; j += row_threads * 4) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const auto i = off + j + k;
      const float g = float(dy[i]) * float(weight[j + k]);
      s += float(dy[i]) * float(weight[j + k]);
      p += g * (float(x[i]) - mean) * r;
    }
  }
  __shared__ float sums[4], products[4];
  s = warp_sum(s);
  p = warp_sum(p);
  if (tid % 32 == 0) {
    sums[tid / 32] = s;
    products[tid / 32] = p;
  }
  __syncthreads();
  if (tid < 32) {
    s = warp_sum(tid < 4 ? sums[tid] : 0.f);
    p = warp_sum(tid < 4 ? products[tid] : 0.f);
    if (tid == 0) {
      sums[0] = s;
      products[0] = p;
    }
  }
  __syncthreads();
  for (int j = tid; j < width; j += row_threads) {
    const auto i = off + j;
    float g = float(width) * float(weight[j]) * float(dy[i]);
    g -= (float(x[i]) - mean) * r * products[0];
    dx[i] = __float2bfloat16_rn((g - sums[0]) * ((1.f / width) * r));
  }
}

__global__ void affine_gradient(int rows, int width, const BF *x, const BF *dy,
                                const float *mean, const float *inv_std,
                                float *dw, float *db) {
  __shared__ float weights[32][33], biases[32][33];
  const int col = blockIdx.x * 32 + threadIdx.x;
  float w = 0, b = 0;
  for (int start = threadIdx.y * 8; start < rows; start += blockDim.y * 8) {
    for (int row = start; row < min(rows, start + 8); ++row) {
      const auto i = size_t(row) * width + col;
      const float grad = float(dy[i]);
      w += grad * (float(x[i]) - mean[row]) * inv_std[row];
      b += grad;
    }
  }
  weights[threadIdx.y][threadIdx.x] = w;
  biases[threadIdx.y][threadIdx.x] = b;
  __syncthreads();
  for (int channel = threadIdx.y; channel < 32; channel += blockDim.y) {
    const int lane = threadIdx.x;
    w = lane < blockDim.y ? weights[lane][channel] : 0.f;
    b = lane < blockDim.y ? biases[lane][channel] : 0.f;
    for (int step = blockDim.y / 2; step; step /= 2) {
      w += shuffle_xor_32(w, step);
      b += shuffle_xor_32(b, step);
    }
    if (lane == 0) {
      // Autograd returns BF16 parameter gradients before FP32 accumulation.
      dw[blockIdx.x * 32 + channel] += float(__float2bfloat16_rn(w));
      db[blockIdx.x * 32 + channel] += float(__float2bfloat16_rn(b));
    }
  }
}

// For tall, narrow matrices ATen stores BF16 partial gradients over groups
// of 32 rows before reducing the partials. Preserve this rounding boundary.
__global__ void affine_partials(int rows, int width, const BF *x, const BF *dy,
                                const float *mean, const float *inv_std,
                                BF *weights, BF *biases) {
  const int col = blockIdx.x * 32 + threadIdx.x;
  float w = 0, b = 0;
  for (int start = blockIdx.y * 32; start < rows; start += gridDim.y * 32) {
    for (int row = start; row < min(start + 32, rows); ++row) {
      const auto i = size_t(row) * width + col;
      const float grad = float(dy[i]);
      w += grad * (float(x[i]) - mean[row]) * inv_std[row];
      b += grad;
    }
  }
  const auto i = blockIdx.y * width + col;
  weights[i] = __float2bfloat16_rn(w);
  biases[i] = __float2bfloat16_rn(b);
}

__global__ void finish_affine(int parts, int width, const BF *weights,
                              const BF *biases, float *dw, float *db) {
  const int col = blockIdx.x * 32 + threadIdx.x;
  __shared__ float ws[32][32], bs[32][32];
  float w = 0, b = 0;
  for (int part = threadIdx.y; part < parts; part += 32) {
    w += float(weights[part * width + col]);
    b += float(biases[part * width + col]);
  }
  ws[threadIdx.y][threadIdx.x] = w;
  bs[threadIdx.y][threadIdx.x] = b;
  __syncthreads();
  w = warp_sum(ws[threadIdx.x][threadIdx.y]);
  b = warp_sum(bs[threadIdx.x][threadIdx.y]);
  if (threadIdx.x == 0) {
    dw[blockIdx.x * 32 + threadIdx.y] += float(__float2bfloat16_rn(w));
    db[blockIdx.x * 32 + threadIdx.y] += float(__float2bfloat16_rn(b));
  }
}
} // namespace

void layer_norm(cudaStream_t stream, int rows, int width, const BF *input,
                const BF *residual, BF *sum, const BF *weight, const BF *bias,
                BF *output, float *mean, float *inv_std) {
  normalize_rows<<<rows, row_threads, 0, stream>>>(
      width, input, residual, sum, weight, bias, output, mean, inv_std);
}

void layer_norm_backward(cudaStream_t stream, int rows, int width,
                         const BF *input, const BF *dy, const BF *weight,
                         const float *mean, const float *inv_std, BF *dx,
                         float *dw, float *db) {
  input_gradient<<<rows, row_threads, 0, stream>>>(width, input, dy, weight,
                                                   mean, inv_std, dx);
  if (rows > 65536) {
    int device, sms;
    CUDA(cudaGetDevice(&device));
    CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const int columns = width / 32;
    if (columns < sms / 2) {
      const int parts = std::min((rows + 31) / 32, 32768 / columns);
      const size_t elements = size_t(parts) * width;
      Buffer<BF> scratch(stream, 2 * elements);
      BF *weights = scratch.get(), *biases = weights + elements;
      affine_partials<<<dim3(columns, parts), 32, 0, stream>>>(
          rows, width, input, dy, mean, inv_std, weights, biases);
      finish_affine<<<columns, dim3(32, 32), 0, stream>>>(parts, width, weights,
                                                          biases, dw, db);
      return;
    }
  }
  const int groups = rows < 64 ? 1 : rows < 128 ? 8 : rows < 256 ? 16 : 32;
  affine_gradient<<<width / 32, dim3(32, groups), 0, stream>>>(
      rows, width, input, dy, mean, inv_std, dw, db);
}
} // namespace rwkv
