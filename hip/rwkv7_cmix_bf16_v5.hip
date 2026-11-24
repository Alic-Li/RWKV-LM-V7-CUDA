#include "rwkv7_training_ops.hpp"
#include <cstdlib>
namespace {

__device__ inline __hip_bfloat162 load_bf16x2(const __hip_bfloat16* ptr) {
    return *reinterpret_cast<const __hip_bfloat162*>(ptr);
}

__device__ inline __hip_bfloat162 make_bf16x2(float x0, float x1) {
    return __float22bfloat162_rn(make_float2(x0, x1));
}

__device__ inline void store_bf16x2(__hip_bfloat16* ptr, __hip_bfloat162 value) {
    *reinterpret_cast<__hip_bfloat162*>(ptr) = value;
}

__device__ inline void atomic_add_float2(float* ptr, float x0, float x1) {
    atomicAdd(ptr + 0, x0);
    atomicAdd(ptr + 1, x1);
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void cmix_mix_forward_vec2_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ x_k,
    __hip_bfloat16* __restrict__ out,
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

    __hip_bfloat162 x_now = load_bf16x2(x + idx);
    __hip_bfloat162 x_prev = make_bf16x2(0.0f, 0.0f);
    if (t > 0) {
        x_prev = load_bf16x2(x + idx - c_size);
    }
    __hip_bfloat162 mix = load_bf16x2(x_k + c);
    store_bf16x2(out + idx, __hadd2(x_now, __hmul2(__hsub2(x_prev, x_now), mix)));
}

constexpr int CMIX_THREADS = 256;

template<int BT_TILE>
__global__ void cmix_mix_backward_fused_vec2_kernel(
    const __hip_bfloat16* __restrict__ grad_out,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ x_k,
    __hip_bfloat16* __restrict__ grad_x,
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

    __hip_bfloat162 mix = load_bf16x2(x_k + c);
    __hip_bfloat162 one_minus_mix = __hsub2(make_bf16x2(1.0f, 1.0f), mix);

    __hip_bfloat162 grad = make_bf16x2(0.0f, 0.0f);
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

        __hip_bfloat162 next_grad = make_bf16x2(0.0f, 0.0f);
        bool have_next_grad = false;
        if (t + 1 < t_size) {
            next_grad = load_bf16x2(grad_out + idx + c_size);
            have_next_grad = true;
        }

        __hip_bfloat162 dx = __hmul2(grad, one_minus_mix);
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
    const __hip_bfloat16* grad_out,
    const __hip_bfloat16* x,
    const __hip_bfloat16* x_k,
    __hip_bfloat16* grad_x,
    float* grad_x_k,
    int64_t bt_size,
    int64_t t_size,
    int64_t c_size,
    hipStream_t stream) {
    dim3 blocks(
        static_cast<unsigned int>(ceil_div(c_size / 2, static_cast<int64_t>(CMIX_THREADS))),
        static_cast<unsigned int>(ceil_div(bt_size, static_cast<int64_t>(BT_TILE))),
        1);
   hipLaunchKernelGGL(( cmix_mix_backward_fused_vec2_kernel<BT_TILE>), dim3(blocks), dim3(CMIX_THREADS), 0, stream, 
        grad_out,
        x,
        x_k,
        grad_x,
        grad_x_k,
        bt_size,
        t_size,
        c_size);
}

__global__ void cast_float_to_bf16_vec2_kernel(
    const float* __restrict__ src,
    __hip_bfloat16* __restrict__ dst,
    int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }
    int64_t idx = pair_idx * 2;
    store_bf16x2(dst + idx, make_bf16x2(src[idx], src[idx + 1]));
}

__global__ void relu_sq_inplace_vec2_kernel(__hip_bfloat16* __restrict__ x, int64_t total_pairs) {
    int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    int64_t idx = pair_idx * 2;
    float2 value = __bfloat1622float2(load_bf16x2(x + idx));
    float x0 = value.x > 0.0f ? value.x : 0.0f;
    float x1 = value.y > 0.0f ? value.y : 0.0f;
    store_bf16x2(x + idx, make_bf16x2(x0 * x0, x1 * x1));
}

__global__ void relu_sq_backward_from_output_inplace_vec2_kernel(
    __hip_bfloat16* __restrict__ grad_out,
    const __hip_bfloat16* __restrict__ out,
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
    store_bf16x2(grad_out + idx, make_bf16x2(g0, g1));
}

}


namespace rwkv::hip_ops {
void cmix(hipStream_t s, int rows, int t, int c, BF *x, BF *mix, BF *out) {
  cmix_mix_forward_vec2_kernel<<<
      ceil_div(int64_t(rows) * (c / 2), CMIX_THREADS), CMIX_THREADS, 0, s>>>(
      x, mix, out, rows, t, c);
}
void cmix_backward(hipStream_t s, int rows, int t, int c, BF *dy, BF *x,
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
void relu_square(hipStream_t s, int64_t n, BF *x) {
  relu_sq_inplace_vec2_kernel<<<ceil_div(n / 2, CMIX_THREADS), CMIX_THREADS, 0,
                                s>>>(x, n / 2);
}
void relu_square_backward(hipStream_t s, int64_t n, BF *dy, BF *out) {
  relu_sq_backward_from_output_inplace_vec2_kernel<<<
      ceil_div(n / 2, CMIX_THREADS), CMIX_THREADS, 0, s>>>(dy, out, n / 2);
}
} // namespace rwkv::hip_ops
