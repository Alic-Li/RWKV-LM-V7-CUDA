#include "rwkv7_training_ops.hpp"
#include <cstdlib>
namespace {

constexpr int kHeadSize = 64;
constexpr int kWarpSize = 32;
#ifndef TMIX_LNX_RKVRES_XG_ROWS_PER_BLOCK
#define TMIX_LNX_RKVRES_XG_ROWS_PER_BLOCK 16
#endif
constexpr int kRowsPerBlock = TMIX_LNX_RKVRES_XG_ROWS_PER_BLOCK;
constexpr int kThreads = kWarpSize * kRowsPerBlock;
constexpr float kLnXEps = 64e-5f;

__device__ inline __hip_bfloat162 load_bf16x2(const __hip_bfloat16* ptr) {
    return *reinterpret_cast<const __hip_bfloat162*>(ptr);
}

__device__ inline void store_bf16(__hip_bfloat16* ptr, float value) {
    *reinterpret_cast<__hip_bfloat16*>(ptr) = __float2bfloat16(value);
}

__device__ inline __hip_bfloat162 make_bf16x2(float x0, float x1) {
    return __float22bfloat162_rn(make_float2(x0, x1));
}

__device__ inline void store_bf16x2(__hip_bfloat16* ptr, float x0, float x1) {
    *reinterpret_cast<__hip_bfloat162*>(ptr) = make_bf16x2(x0, x1);
}


__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset, 32);
    }
    return v;
}

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void tmix_lnx_rkvres_xg_v1_forward_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ r,
    const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v,
    const __hip_bfloat16* __restrict__ r_k,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ bias,
    const __hip_bfloat16* __restrict__ g,
    __hip_bfloat16* __restrict__ xg,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int64_t ngroups,
    int64_t nrows) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t group = static_cast<int64_t>(blockIdx.x);
    const int64_t row = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock + warp;
    if (row >= nrows) {
        return;
    }

    const int64_t c0 = group * kHeadSize;
    const int64_t base = row * (ngroups * kHeadSize) + c0;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;
    const int64_t ng = row * ngroups + group;

    const __hip_bfloat162 xv2 = load_bf16x2(x + idx);
    const float x0 = __low2float(xv2);
    const float x1 = __high2float(xv2);

    float sum = warp_sum(x0 + x1);
    const float mean_val = __shfl(sum, 0, 32) * (1.0f / kHeadSize);

    const float d0 = x0 - mean_val;
    const float d1 = x1 - mean_val;
    float sq = warp_sum(d0 * d0 + d1 * d1);
    const float var_val = __shfl(sq, 0, 32) * (1.0f / kHeadSize);
    const float rstd_val = rsqrtf(var_val + kLnXEps);

    const __hip_bfloat162 rv2 = load_bf16x2(r + idx);
    const __hip_bfloat162 kv2 = load_bf16x2(k + idx);
    const __hip_bfloat162 rk2 = load_bf16x2(r_k + c);
    float scale_sum =
        __low2float(rv2) * __low2float(kv2) * __low2float(rk2) +
        __high2float(rv2) * __high2float(kv2) * __high2float(rk2);
    scale_sum = warp_sum(scale_sum);
    const float scale_val = __shfl(scale_sum, 0, 32);

    if (lane == 0) {
        mean[ng] = mean_val;
        rstd[ng] = rstd_val;
    }

    const __hip_bfloat162 vv2 = load_bf16x2(v + idx);
    const __hip_bfloat162 w2 = load_bf16x2(weight + c);
    const __hip_bfloat162 b2 = load_bf16x2(bias + c);
    const __hip_bfloat162 g2 = load_bf16x2(g + idx);
    const float y0 = d0 * rstd_val * __low2float(w2) + __low2float(b2) + scale_val * __low2float(vv2);
    const float y1 = d1 * rstd_val * __high2float(w2) + __high2float(b2) + scale_val * __high2float(vv2);
    store_bf16x2(xg + idx, y0 * __low2float(g2), y1 * __high2float(g2));
}

__global__ void tmix_lnx_rkvres_xg_v1_backward_kernel(
    const __hip_bfloat16* __restrict__ grad_xg,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ r,
    const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v,
    const __hip_bfloat16* __restrict__ r_k,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ bias,
    const __hip_bfloat16* __restrict__ g,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    __hip_bfloat16* __restrict__ grad_x,
    __hip_bfloat16* __restrict__ grad_r,
    __hip_bfloat16* __restrict__ grad_k,
    __hip_bfloat16* __restrict__ grad_v,
    float* __restrict__ grad_r_k,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    __hip_bfloat16* __restrict__ grad_g,
    int64_t ngroups,
    int64_t nrows) {
    __shared__ float s_gw0[kRowsPerBlock][kWarpSize];
    __shared__ float s_gw1[kRowsPerBlock][kWarpSize];
    __shared__ float s_gb0[kRowsPerBlock][kWarpSize];
    __shared__ float s_gb1[kRowsPerBlock][kWarpSize];
    __shared__ float s_grk0[kRowsPerBlock][kWarpSize];
    __shared__ float s_grk1[kRowsPerBlock][kWarpSize];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t group = static_cast<int64_t>(blockIdx.x);
    const int64_t row = static_cast<int64_t>(blockIdx.y) * kRowsPerBlock + warp;
    const bool valid = row < nrows;

    const int64_t c0 = group * kHeadSize;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;
    const int64_t ng = row * ngroups + group;
    const int64_t base = row * (ngroups * kHeadSize) + c0;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;

    float gw0 = 0.0f;
    float gw1 = 0.0f;
    float gb0 = 0.0f;
    float gb1 = 0.0f;
    float grk0 = 0.0f;
    float grk1 = 0.0f;

    if (valid) {
        const float mean_val = mean[ng];
        const float rstd_val = rstd[ng];

        const __hip_bfloat162 xv2 = load_bf16x2(x + idx);
        const __hip_bfloat162 gxg2 = load_bf16x2(grad_xg + idx);
        const __hip_bfloat162 g2 = load_bf16x2(g + idx);
        const __hip_bfloat162 w2 = load_bf16x2(weight + c);
        const __hip_bfloat162 b2 = load_bf16x2(bias + c);
        const __hip_bfloat162 vv2 = load_bf16x2(v + idx);

        const float x0 = __low2float(xv2);
        const float x1 = __high2float(xv2);
        const float gxg0 = __low2float(gxg2);
        const float gxg1 = __high2float(gxg2);
        const float g0 = __low2float(g2);
        const float g1 = __high2float(g2);
        const float gy0 = gxg0 * g0;
        const float gy1 = gxg1 * g1;
        const float w0 = __low2float(w2);
        const float w1 = __high2float(w2);
        const float xhat0 = (x0 - mean_val) * rstd_val;
        const float xhat1 = (x1 - mean_val) * rstd_val;
        const float dxhat0 = gy0 * w0;
        const float dxhat1 = gy1 * w1;

        float total_dxhat = warp_sum(dxhat0 + dxhat1);
        float total_dxhat_xhat = warp_sum(dxhat0 * xhat0 + dxhat1 * xhat1);
        total_dxhat = __shfl(total_dxhat, 0, 32);
        total_dxhat_xhat = __shfl(total_dxhat_xhat, 0, 32);
        const float inv_m = 1.0f / kHeadSize;

        const float gx0 = (dxhat0 - total_dxhat * inv_m - xhat0 * total_dxhat_xhat * inv_m) * rstd_val;
        const float gx1 = (dxhat1 - total_dxhat * inv_m - xhat1 * total_dxhat_xhat * inv_m) * rstd_val;
        store_bf16x2(grad_x + idx, gx0, gx1);

        gw0 = gy0 * xhat0;
        gw1 = gy1 * xhat1;
        gb0 = gy0;
        gb1 = gy1;

        const __hip_bfloat162 rv2 = load_bf16x2(r + idx);
        const __hip_bfloat162 kv2 = load_bf16x2(k + idx);
        const __hip_bfloat162 rk2 = load_bf16x2(r_k + c);
        const float r0 = __low2float(rv2);
        const float r1 = __high2float(rv2);
        const float k0 = __low2float(kv2);
        const float k1 = __high2float(kv2);
        const float rk0 = __low2float(rk2);
        const float rk1 = __high2float(rk2);

        float scale_sum = r0 * k0 * rk0 + r1 * k1 * rk1;
        scale_sum = warp_sum(scale_sum);
        const float scale_val = __shfl(scale_sum, 0, 32);

        const float v0 = __low2float(vv2);
        const float v1 = __high2float(vv2);
        const float q = __shfl(warp_sum(gy0 * v0 + gy1 * v1), 0, 32);
        store_bf16x2(grad_v + idx, gy0 * scale_val, gy1 * scale_val);
        store_bf16x2(grad_r + idx, q * k0 * rk0, q * k1 * rk1);
        store_bf16x2(grad_k + idx, q * r0 * rk0, q * r1 * rk1);

        const float y0 = xhat0 * w0 + __low2float(b2) + scale_val * v0;
        const float y1 = xhat1 * w1 + __high2float(b2) + scale_val * v1;
        store_bf16x2(grad_g + idx, gxg0 * y0, gxg1 * y1);

        grk0 = q * r0 * k0;
        grk1 = q * r1 * k1;
    }

    s_gw0[warp][lane] = gw0;
    s_gw1[warp][lane] = gw1;
    s_gb0[warp][lane] = gb0;
    s_gb1[warp][lane] = gb1;
    s_grk0[warp][lane] = grk0;
    s_grk1[warp][lane] = grk1;
    __syncthreads();

    if (warp == 0) {
        float sum_gw0 = 0.0f;
        float sum_gw1 = 0.0f;
        float sum_gb0 = 0.0f;
        float sum_gb1 = 0.0f;
        float sum_grk0 = 0.0f;
        float sum_grk1 = 0.0f;
#pragma unroll
        for (int w = 0; w < kRowsPerBlock; ++w) {
            sum_gw0 += s_gw0[w][lane];
            sum_gw1 += s_gw1[w][lane];
            sum_gb0 += s_gb0[w][lane];
            sum_gb1 += s_gb1[w][lane];
            sum_grk0 += s_grk0[w][lane];
            sum_grk1 += s_grk1[w][lane];
        }
        atomicAdd(grad_weight + c + 0, sum_gw0);
        atomicAdd(grad_weight + c + 1, sum_gw1);
        atomicAdd(grad_bias + c + 0, sum_gb0);
        atomicAdd(grad_bias + c + 1, sum_gb1);
        atomicAdd(grad_r_k + c + 0, sum_grk0);
        atomicAdd(grad_r_k + c + 1, sum_grk1);
    }
}

__global__ void cast_float_to_bf16_kernel(
    const float* __restrict__ src,
    __hip_bfloat16* __restrict__ dst,
    int64_t size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    store_bf16(dst + idx, src[idx]);
}

}


namespace rwkv::hip_ops {
void norm_gate(hipStream_t s, int rows, int c, BF *x, BF *r, BF *k, BF *v,
               BF *rk, BF *weight, BF *bias, BF *g, BF *y, float *mean,
               float *rstd) {
  const int h = c / kHeadSize;
  const dim3 grid(h, ceil_div(rows, kRowsPerBlock));
  tmix_lnx_rkvres_xg_v1_forward_kernel<<<grid, kThreads, 0, s>>>(
      x, r, k, v, rk, weight, bias, g, y, mean, rstd, h, rows);
}
void norm_gate_backward(hipStream_t s, int rows, int c, BF *dy, BF *x, BF *r,
                        BF *k, BF *v, BF *rk, BF *weight, BF *bias, BF *g,
                        float *mean, float *rstd, BF *dx, BF *dr, BF *dk,
                        BF *dv, float *drk, float *dw, float *db, BF *dg) {
  const int h = c / kHeadSize;
  const dim3 grid(h, ceil_div(rows, kRowsPerBlock));
  tmix_lnx_rkvres_xg_v1_backward_kernel<<<grid, kThreads, 0, s>>>(
      dy, x, r, k, v, rk, weight, bias, g, mean, rstd, dx, dr, dk, dv, drk, dw,
      db, dg, h, rows);
}
} // namespace rwkv::hip_ops
