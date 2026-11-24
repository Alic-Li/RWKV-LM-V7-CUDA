#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rwkv {
// Writes touched vocabulary rows into a zero-initialized FP32 gradient.
void embedding_gradient(cudaStream_t stream, int rows, int channels,
                        const int *tokens, const __nv_bfloat16 *upstream,
                        float *gradient);
} // namespace rwkv
