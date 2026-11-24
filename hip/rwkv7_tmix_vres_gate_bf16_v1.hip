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
    __hip_bfloat16 out = __float2bfloat16(value);
    *reinterpret_cast<__hip_bfloat16*>(ptr) = out;
}

__device__ inline float sigmoidf_stable(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void vres_gate_forward_kernel(
    const __hip_bfloat16* __restrict__ v,
    const __hip_bfloat16* __restrict__ v_first,
    const __hip_bfloat16* __restrict__ v0,
    const __hip_bfloat16* __restrict__ v12,
    __hip_bfloat16* __restrict__ out,
    int64_t total,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    store_bf16(out + idx, vv + (vf - vv) * gate);
}

__global__ void vres_gate_forward_pow2c_kernel(
    const __hip_bfloat16* __restrict__ v,
    const __hip_bfloat16* __restrict__ v_first,
    const __hip_bfloat16* __restrict__ v0,
    const __hip_bfloat16* __restrict__ v12,
    __hip_bfloat16* __restrict__ out,
    int64_t total,
    int64_t c_mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx & c_mask;
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    store_bf16(out + idx, vv + (vf - vv) * gate);
}

__global__ void vres_gate_backward_kernel(
    const __hip_bfloat16* __restrict__ grad_out,
    const __hip_bfloat16* __restrict__ v,
    const __hip_bfloat16* __restrict__ v_first,
    const __hip_bfloat16* __restrict__ v0,
    const __hip_bfloat16* __restrict__ v12,
    __hip_bfloat16* __restrict__ grad_v,
    __hip_bfloat16* __restrict__ grad_v_first,
    float* __restrict__ grad_pre,
    int64_t total,
    int64_t c_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx % c_size;
    float go = bf16_to_float(grad_out + idx);
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    float grad_vf = go * gate;
    float grad_vv = go - grad_vf;
    float gp = go * (vf - vv) * gate * (1.0f - gate);

    store_bf16(grad_v + idx, grad_vv);
    store_bf16(grad_v_first + idx, grad_vf);
    grad_pre[idx] = gp;
}

__global__ void vres_gate_backward_pow2c_kernel(
    const __hip_bfloat16* __restrict__ grad_out,
    const __hip_bfloat16* __restrict__ v,
    const __hip_bfloat16* __restrict__ v_first,
    const __hip_bfloat16* __restrict__ v0,
    const __hip_bfloat16* __restrict__ v12,
    __hip_bfloat16* __restrict__ grad_v,
    __hip_bfloat16* __restrict__ grad_v_first,
    float* __restrict__ grad_pre,
    int64_t total,
    int64_t c_mask) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t c = idx & c_mask;
    float go = bf16_to_float(grad_out + idx);
    float vv = bf16_to_float(v + idx);
    float vf = bf16_to_float(v_first + idx);
    float pre = bf16_to_float(v0 + c) + bf16_to_float(v12 + idx);
    float gate = sigmoidf_stable(pre);
    float grad_vf = go * gate;
    float grad_vv = go - grad_vf;
    float gp = go * (vf - vv) * gate * (1.0f - gate);

    store_bf16(grad_v + idx, grad_vv);
    store_bf16(grad_v_first + idx, grad_vf);
    grad_pre[idx] = gp;
}

}


namespace rwkv::hip_ops {
void value_gate(hipStream_t s, int rows, int c, BF *v, BF *first, BF *v0,
                BF *v12, BF *out) {
  const int64_t n = int64_t(rows) * c;
  const int blocks = ceil_div(n, 128);
  if (is_power_of_two(c))
    vres_gate_forward_pow2c_kernel<<<blocks, 128, 0, s>>>(v, first, v0, v12,
                                                          out, n, c - 1);
  else
    vres_gate_forward_kernel<<<blocks, 128, 0, s>>>(v, first, v0, v12, out, n,
                                                    c);
}
void value_gate_backward(hipStream_t s, int rows, int c, BF *dy, BF *v,
                         BF *first, BF *v0, BF *v12, BF *dv, BF *dfirst,
                         float *dpre) {
  const int64_t n = int64_t(rows) * c;
  const int blocks = ceil_div(n, 128);
  if (is_power_of_two(c))
    vres_gate_backward_pow2c_kernel<<<blocks, 128, 0, s>>>(
        dy, v, first, v0, v12, dv, dfirst, dpre, n, c - 1);
  else
    vres_gate_backward_kernel<<<blocks, 128, 0, s>>>(dy, v, first, v0, v12, dv,
                                                     dfirst, dpre, n, c);
}
} // namespace rwkv::hip_ops
