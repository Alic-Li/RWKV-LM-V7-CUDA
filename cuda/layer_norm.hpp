#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rwkv {
using BF = __nv_bfloat16;
void layer_norm(cudaStream_t stream, int rows, int width, const BF *input,
                const BF *residual, BF *sum, const BF *weight, const BF *bias,
                BF *output, float *mean, float *inv_std);
void layer_norm_backward(cudaStream_t stream, int rows, int width,
                         const BF *input, const BF *dy, const BF *weight,
                         const float *mean, const float *inv_std, BF *dx,
                         float *dw, float *db);
} // namespace rwkv
