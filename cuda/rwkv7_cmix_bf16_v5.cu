#include "atomic_pair.cuh"
#include "rwkv7_training_ops.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

namespace {
using KernelBF16 = __nv_bfloat16;


__device__ inline __nv_bfloat162 load_bf16x2(const KernelBF16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(KernelBF16* ptr, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = value;
}

using rwkv::atomic_add_float2;

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void cmix_mix_forward_vec2_kernel(
    const KernelBF16* __restrict__ x,
    const KernelBF16* __restrict__ x_k,
    KernelBF16* __restrict__ out,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = bt_size * (c_size / 2);
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t bt = pair_idx / (c_size / 2);
    int64_t c_pair = pair_idx % (c_size / 2);
    int64_t c = c_pair * 2;
    int64_t idx = bt * c_size + c;
    int64_t t = bt % t_size;

    __nv_bfloat162 x_now = load_bf16x2(x + idx);
    __nv_bfloat162 x_prev = __floats2bfloat162_rn(0.0f, 0.0f);
    if (t > 0) {
        x_prev = load_bf16x2(x + idx - c_size);
    }
    __nv_bfloat162 mix = load_bf16x2(x_k + c);
    store_bf16x2(out + idx, __hadd2(x_now, __hmul2(__hsub2(x_prev, x_now), mix)));
}

constexpr int CMIX_THREADS = 256;

template<int BT_TILE>
__global__ void cmix_mix_backward_fused_vec2_kernel(
    const KernelBF16* __restrict__ grad_out,
    const KernelBF16* __restrict__ x,
    const KernelBF16* __restrict__ x_k,
    KernelBF16* __restrict__ grad_x,
    float* __restrict__ grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size) {
    int64_t c_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_pairs = c_size / 2;
    if (c_pair >= total_pairs) {
        return;
    }

    int64_t c = c_pair * 2;
    int64_t bt_start = static_cast<int64_t>(blockIdx.y) * BT_TILE;
    int64_t bt_end = min(bt_start + static_cast<int64_t>(BT_TILE), bt_size);

    __nv_bfloat162 mix = load_bf16x2(x_k + c);
    __nv_bfloat162 one_minus_mix = __hsub2(__floats2bfloat162_rn(1.0f, 1.0f), mix);

    __nv_bfloat162 grad = __floats2bfloat162_rn(0.0f, 0.0f);
    bool have_grad = false;
    float2 prev_x = make_float2(0.0f, 0.0f);
    bool have_prev_x = false;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int64_t bt = bt_start; bt < bt_end; ++bt) {
        int64_t idx = bt * c_size + c;
        int64_t t = bt % t_size;
        if (!have_grad) {
            grad = load_bf16x2(grad_out + idx);
        }

        __nv_bfloat162 next_grad = __floats2bfloat162_rn(0.0f, 0.0f);
        bool have_next_grad = false;
        if (t + 1 < t_size) {
            next_grad = load_bf16x2(grad_out + idx + c_size);
            have_next_grad = true;
        }

        __nv_bfloat162 dx = __hmul2(grad, one_minus_mix);
        if (have_next_grad) {
            dx = __hadd2(dx, __hmul2(next_grad, mix));
        }
        store_bf16x2(grad_x + idx, dx);

        float2 x_now = __bfloat1622float2(load_bf16x2(x + idx));
        float2 x_prev = make_float2(0.0f, 0.0f);
        if (t > 0) {
            x_prev = have_prev_x ? prev_x : __bfloat1622float2(load_bf16x2(x + idx - c_size));
        }
        float2 g = __bfloat1622float2(grad);
        sum0 += g.x * (x_prev.x - x_now.x);
        sum1 += g.y * (x_prev.y - x_now.y);

        prev_x = x_now;
        have_prev_x = (t + 1 < t_size);
        grad = next_grad;
        have_grad = have_next_grad;
    }

    atomic_add_float2(grad_x_k + c, sum0, sum1);
}

template<int BT_TILE>
void launch_cmix_mix_backward_fused_vec2_kernel(
    const KernelBF16* grad_out,
    const KernelBF16* x,
    const KernelBF16* x_k,
    KernelBF16* grad_x,
    float* grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size,
    cudaStream_t stream) {
    dim3 blocks(
        static_cast<unsigned int>(ceil_div(c_size / 2, static_cast<int64_t>(CMIX_THREADS))),
        static_cast<unsigned int>(ceil_div(bt_size, static_cast<int64_t>(BT_TILE))),
        1);
    cmix_mix_backward_fused_vec2_kernel<BT_TILE><<<blocks, CMIX_THREADS, 0, stream>>>(
        grad_out,
        x,
        x_k,
        grad_x,
        grad_x_k,
        bt_size,
        t_size,
        c_size);
}


__global__ void relu_sq_inplace_vec2_kernel(KernelBF16* __restrict__ x, int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 value = __bfloat1622float2(load_bf16x2(x + idx));
    float x0 = value.x > 0.0f ? value.x : 0.0f;
    float x1 = value.y > 0.0f ? value.y : 0.0f;
    store_bf16x2(x + idx, __floats2bfloat162_rn(x0 * x0, x1 * x1));
}

__global__ void relu_sq_backward_from_output_inplace_vec2_kernel(
    KernelBF16* __restrict__ grad_out,
    const KernelBF16* __restrict__ out,
    int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 grad_v = __bfloat1622float2(load_bf16x2(grad_out + idx));
    float2 act_v = __bfloat1622float2(load_bf16x2(out + idx));
    float g0 = grad_v.x * (2.0f * sqrtf(fmaxf(act_v.x, 0.0f)));
    float g1 = grad_v.y * (2.0f * sqrtf(fmaxf(act_v.y, 0.0f)));
    store_bf16x2(grad_out + idx, __floats2bfloat162_rn(g0, g1));
}

} // namespace
namespace rwkv {
void cmix(cudaStream_t s, int rows, int t, int c, BF *x, BF *mix, BF *out) {
  cmix_mix_forward_vec2_kernel<<<
      ceil_div(int64_t(rows) * (c / 2), CMIX_THREADS), CMIX_THREADS, 0, s>>>(
      x, mix, out, rows, t, c);
}
void cmix_backward(cudaStream_t s, int rows, int t, int c, BF *dy, BF *x,
                   BF *mix, BF *dx, float *dm) {
  if (rows <= 256)
    launch_cmix_mix_backward_fused_vec2_kernel<16>(dy, x, mix, dx, dm, rows, t,
                                                   c, s);
  else if (rows >= 4096)
    launch_cmix_mix_backward_fused_vec2_kernel<128>(dy, x, mix, dx, dm, rows, t,
                                                    c, s);
  else if (rows >= 1024)
    launch_cmix_mix_backward_fused_vec2_kernel<64>(dy, x, mix, dx, dm, rows, t,
                                                   c, s);
  else
    launch_cmix_mix_backward_fused_vec2_kernel<32>(dy, x, mix, dx, dm, rows, t,
                                                   c, s);
}
void relu_square(cudaStream_t s, int64_t n, BF *x) {
  relu_sq_inplace_vec2_kernel<<<ceil_div(n / 2, CMIX_THREADS), CMIX_THREADS, 0,
                                s>>>(x, n / 2);
}
void relu_square_backward(cudaStream_t s, int64_t n, BF *dy, BF *out) {
  relu_sq_backward_from_output_inplace_vec2_kernel<<<
      ceil_div(n / 2, CMIX_THREADS), CMIX_THREADS, 0, s>>>(dy, out, n / 2);
}
} // namespace rwkv
