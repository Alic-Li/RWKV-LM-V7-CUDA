#pragma once
#include <cuda_runtime.h>

namespace rwkv {
__device__ inline void atomic_add_float2(float *ptr, float x, float y) {
#if __CUDA_ARCH__ >= 900
    atomicAdd(reinterpret_cast<float2 *>(ptr), make_float2(x, y));
#else
    atomicAdd(ptr, x);
    atomicAdd(ptr + 1, y);
#endif
}
}
