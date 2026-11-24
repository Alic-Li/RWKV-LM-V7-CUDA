#include "rwkv7_training_ops.hpp"
#include <cstdlib>
namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

inline bool is_power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

__device__ inline float bf16_to_float(const __hip_bfloat16* ptr) {
    return __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(ptr));
}

__device__ inline void store_bf16(__hip_bfloat16* ptr, float value) {
    *reinterpret_cast<__hip_bfloat16*>(ptr) = __float2bfloat16(value);
}

__device__ inline float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void a_gate_forward_kernel(
    const __hip_bfloat16* __restrict__ a0,
    const __hip_bfloat16* __restrict__ a12,
    __hip_bfloat16* __restrict__ out,
    int64_t total,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    float pre = bf16_to_float(a0 + c) + bf16_to_float(a12 + idx);
    store_bf16(out + idx, sigmoidf_fast(pre));
}

__global__ void a_gate_forward_pow2c_kernel(
    const __hip_bfloat16* __restrict__ a0,
    const __hip_bfloat16* __restrict__ a12,
    __hip_bfloat16* __restrict__ out,
    int64_t total,
    int64_t c_mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx & c_mask;
    float pre = bf16_to_float(a0 + c) + bf16_to_float(a12 + idx);
    store_bf16(out + idx, sigmoidf_fast(pre));
}

constexpr int kBackwardTileM = 16;
constexpr int kDefaultThreads = 256;

__global__ void a_gate_backward_full_kernel(
    const __hip_bfloat16* __restrict__ grad_out,
    const __hip_bfloat16* __restrict__ a0,
    const __hip_bfloat16* __restrict__ a12,
    __hip_bfloat16* __restrict__ grad_a12,
    float* __restrict__ partial_a0,
    int64_t rows,
    int64_t c_size) {
    int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= c_size) {
        return;
    }

    int64_t row_base = static_cast<int64_t>(blockIdx.y) * kBackwardTileM;
    float a0v = bf16_to_float(a0 + c);
    float sum = 0.0f;
    for (int i = 0; i < kBackwardTileM; ++i) {
        int64_t row = row_base + i;
        if (row >= rows) {
            break;
        }
        int64_t idx = row * c_size + c;
        float go = bf16_to_float(grad_out + idx);
        float a = sigmoidf_fast(a0v + bf16_to_float(a12 + idx));
        float gp = go * a * (1.0f - a);
        store_bf16(grad_a12 + idx, gp);
        sum += gp;
    }

    partial_a0[static_cast<int64_t>(blockIdx.y) * c_size + c] = sum;
}

__global__ void a_gate_reduce_a0_kernel(
    const float* __restrict__ partial_a0,
    __hip_bfloat16* __restrict__ grad_a0,
    int64_t num_tiles,
    int64_t c_size) {
    int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= c_size) {
        return;
    }

    float sum = 0.0f;
    for (int64_t tile = 0; tile < num_tiles; ++tile) {
        sum += partial_a0[tile * c_size + c];
    }
    store_bf16(grad_a0 + c, sum);
}

}


namespace rwkv::hip_ops {
void alpha(hipStream_t s, int rows, int c, BF *a0, BF *a12, BF *out) {
  const int64_t n = int64_t(rows) * c;
  const int blocks = ceil_div(n, kDefaultThreads);
  if (is_power_of_two(c))
    a_gate_forward_pow2c_kernel<<<blocks, kDefaultThreads, 0, s>>>(a0, a12, out,
                                                                   n, c - 1);
  else
    a_gate_forward_kernel<<<blocks, kDefaultThreads, 0, s>>>(a0, a12, out, n,
                                                             c);
}
int alpha_tiles(int rows) { return ceil_div(rows, kBackwardTileM); }
void alpha_backward(hipStream_t s, int rows, int c, BF *dy, BF *a0, BF *a12,
                    BF *da12, float *partial, BF *da0) {
  const int tiles = alpha_tiles(rows);
  const int blocks = ceil_div(c, kDefaultThreads);
  const dim3 grid(blocks, tiles);
  a_gate_backward_full_kernel<<<grid, kDefaultThreads, 0, s>>>(
      dy, a0, a12, da12, partial, rows, c);
  a_gate_reduce_a0_kernel<<<blocks, kDefaultThreads, 0, s>>>(partial, da0,
                                                             tiles, c);
}
} // namespace rwkv::hip_ops
