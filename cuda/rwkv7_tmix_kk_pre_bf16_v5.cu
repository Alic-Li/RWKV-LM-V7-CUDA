#include "atomic_pair.cuh"
#include "rwkv7_training_ops.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
using KernelBF16 = __nv_bfloat16;


constexpr float kNormalizeEps = 1.0e-12f;
constexpr float kInvNormalizeEps = 1.0e12f;
constexpr int kHeadSize = 64;
constexpr int kWarpsPerBlock = 4;

__device__ inline __nv_bfloat162 load_bf16x2(const KernelBF16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_bf16x2(KernelBF16* ptr, float x0, float x1) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = __floats2bfloat162_rn(x0, x1);
}


__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

using rwkv::atomic_add_float2;

inline int64_t ceil_div(int64_t n, int64_t d) {
    return (n + d - 1) / d;
}

__global__ void tmix_kk_pre_forward64_v5_kernel(
    const KernelBF16* __restrict__ k,
    const KernelBF16* __restrict__ k_k,
    const KernelBF16* __restrict__ a,
    const KernelBF16* __restrict__ k_a,
    KernelBF16* __restrict__ new_k,
    KernelBF16* __restrict__ neg_kk,
    KernelBF16* __restrict__ kka,
    float* __restrict__ inv_d_out,
    int64_t bth_size,
    int64_t h_size) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t bth = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (bth >= bth_size) {
        return;
    }

    const int64_t h = bth % h_size;
    const int64_t base = bth * kHeadSize;
    const int64_t c0 = h * kHeadSize;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;

    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 kk_scale2 = load_bf16x2(k_k + c);
    const float kv0 = __low2float(kv2);
    const float kv1 = __high2float(kv2);
    const float ks0 = __low2float(kk_scale2);
    const float ks1 = __high2float(kk_scale2);
    const float u0 = kv0 * ks0;
    const float u1 = kv1 * ks1;

    float sum_sq = u0 * u0 + u1 * u1;
    sum_sq = warp_sum(sum_sq);
    const float total_sum_sq = __shfl_sync(0xffffffffu, sum_sq, 0);
    const float inv_d = 1.0f / fmaxf(sqrtf(total_sum_sq), kNormalizeEps);
    if (lane == 0) {
        inv_d_out[bth] = inv_d;
    }

    const __nv_bfloat162 av2 = load_bf16x2(a + idx);
    const __nv_bfloat162 ka_scale2 = load_bf16x2(k_a + c);
    const float av0 = __low2float(av2);
    const float av1 = __high2float(av2);
    const float ka0 = __low2float(ka_scale2);
    const float ka1 = __high2float(ka_scale2);
    const float kk0 = u0 * inv_d;
    const float kk1 = u1 * inv_d;

    store_bf16x2(new_k + idx, kv0 * fmaf(av0, ka0, 1.0f - ka0), kv1 * fmaf(av1, ka1, 1.0f - ka1));
    store_bf16x2(neg_kk + idx, -kk0, -kk1);
    store_bf16x2(kka + idx, kk0 * av0, kk1 * av1);
}

__global__ void tmix_kk_pre_backward64_v5_kernel(
    const KernelBF16* __restrict__ grad_new_k,
    const KernelBF16* __restrict__ grad_neg_kk,
    const KernelBF16* __restrict__ grad_kka,
    const KernelBF16* __restrict__ k,
    const KernelBF16* __restrict__ k_k,
    const KernelBF16* __restrict__ a,
    const KernelBF16* __restrict__ k_a,
    const float* __restrict__ inv_d_buf,
    KernelBF16* __restrict__ grad_k,
    float* __restrict__ grad_k_k,
    KernelBF16* __restrict__ grad_a,
    float* __restrict__ grad_k_a,
    int64_t bth_size,
    int64_t h_size) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t bth = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (bth >= bth_size) {
        return;
    }

    const int64_t h = bth % h_size;
    const int64_t base = bth * kHeadSize;
    const int64_t c0 = h * kHeadSize;
    const int64_t idx = base + static_cast<int64_t>(lane) * 2;
    const int64_t c = c0 + static_cast<int64_t>(lane) * 2;
    const float inv_d = inv_d_buf[bth];
    const bool use_norm_branch = inv_d < kInvNormalizeEps;

    const __nv_bfloat162 kv2 = load_bf16x2(k + idx);
    const __nv_bfloat162 kk_scale2 = load_bf16x2(k_k + c);
    const __nv_bfloat162 av2 = load_bf16x2(a + idx);
    const __nv_bfloat162 gneg_kk2 = load_bf16x2(grad_neg_kk + idx);
    const __nv_bfloat162 gkka2 = load_bf16x2(grad_kka + idx);
    const float kv0 = __low2float(kv2);
    const float kv1 = __high2float(kv2);
    const float ks0 = __low2float(kk_scale2);
    const float ks1 = __high2float(kk_scale2);
    const float av0 = __low2float(av2);
    const float av1 = __high2float(av2);
    const float kk0 = kv0 * ks0 * inv_d;
    const float kk1 = kv1 * ks1 * inv_d;
    const float gkka0 = __low2float(gkka2);
    const float gkka1 = __high2float(gkka2);
    const float gkk_total0 = -__low2float(gneg_kk2) + gkka0 * av0;
    const float gkk_total1 = -__high2float(gneg_kk2) + gkka1 * av1;

    float dot = gkk_total0 * kk0 + gkk_total1 * kk1;
    dot = warp_sum(dot);
    const float dot_total = __shfl_sync(0xffffffffu, dot, 0);

    const __nv_bfloat162 ka_scale2 = load_bf16x2(k_a + c);
    const __nv_bfloat162 gnew2 = load_bf16x2(grad_new_k + idx);
    const float ka0 = __low2float(ka_scale2);
    const float ka1 = __high2float(ka_scale2);
    const float gnew0 = __low2float(gnew2);
    const float gnew1 = __high2float(gnew2);

    float gu0 = gkk_total0 * inv_d;
    float gu1 = gkk_total1 * inv_d;
    if (use_norm_branch) {
        gu0 = (gkk_total0 - kk0 * dot_total) * inv_d;
        gu1 = (gkk_total1 - kk1 * dot_total) * inv_d;
    }

    const float scale0 = fmaf(av0, ka0, 1.0f - ka0);
    const float scale1 = fmaf(av1, ka1, 1.0f - ka1);
    store_bf16x2(grad_k + idx, gnew0 * scale0 + gu0 * ks0, gnew1 * scale1 + gu1 * ks1);
    store_bf16x2(grad_a + idx, gnew0 * kv0 * ka0 + gkka0 * kk0, gnew1 * kv1 * ka1 + gkka1 * kk1);
    atomic_add_float2(grad_k_k + c, gu0 * kv0, gu1 * kv1);
    atomic_add_float2(grad_k_a + c, gnew0 * kv0 * (av0 - 1.0f), gnew1 * kv1 * (av1 - 1.0f));
}


} // namespace

namespace rwkv {
void key_prepare(cudaStream_t s, int rows, int c, BF *k, BF *kk, BF *a, BF *ka,
                 BF *key, BF *neg, BF *kka, float *inv) {
  const int h = c / kHeadSize;
  const int64_t n = int64_t(rows) * h;
  tmix_kk_pre_forward64_v5_kernel<<<ceil_div(n, kWarpsPerBlock),
                                    kWarpsPerBlock * 32, 0, s>>>(
      k, kk, a, ka, key, neg, kka, inv, n, h);
}
void key_prepare_backward(cudaStream_t s, int rows, int c, BF *k, BF *kk, BF *a,
                          BF *ka, BF *dkey, BF *dneg, BF *dkka, float *inv,
                          BF *dk, float *dkk, BF *da, float *dka) {
  const int h = c / kHeadSize;
  const int64_t n = int64_t(rows) * h;
  tmix_kk_pre_backward64_v5_kernel<<<ceil_div(n, kWarpsPerBlock),
                                     kWarpsPerBlock * 32, 0, s>>>(
      dkey, dneg, dkka, k, kk, a, ka, inv, dk, dkk, da, dka, n, h);
}
} // namespace rwkv
