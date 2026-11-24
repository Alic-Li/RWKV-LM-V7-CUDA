#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace rwkv {
struct Adam {
  double lr, beta1, beta2, eps, decay;
  float clip;
};

struct AdamStep {
  float beta2, alpha1, alpha2, eps, step_size, inv_sqrt_correction,
      decay_factor;
};

AdamStep adam_step(const Adam &opt, std::uint64_t step, double lr,
                   double decay);
void adam_update(cudaStream_t stream, __nv_bfloat16 *weight, float *master,
                 const float *grad, float *m, float *v, std::size_t count,
                 AdamStep step, float scale);
} // namespace rwkv
