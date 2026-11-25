#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rwkv {
class Arena;
std::size_t embedding_sort_bytes(cudaStream_t stream, int rows);
// Writes touched vocabulary rows into a zero-initialized FP32 gradient.
void embedding_gradient(Arena &arena, std::size_t scratch_bytes,
                        cudaStream_t stream, int rows, int channels,
                        const int *tokens, const __nv_bfloat16 *upstream,
                        float *gradient);
} // namespace rwkv
