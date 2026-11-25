#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rwkv {
class Arena;
// Reduce contiguous [rows, channels] bias gradients for the value and decay
// gates.
template <typename T>
void value_bias_gradient(Arena &arena, cudaStream_t stream, int rows,
                         int channels, int resident_threads, const T *terms,
                         float *gradient);
} // namespace rwkv
