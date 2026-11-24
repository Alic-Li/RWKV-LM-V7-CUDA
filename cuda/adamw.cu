#include "adamw.hpp"
#include <cmath>

namespace rwkv {
namespace {
__global__ void update_weights(__nv_bfloat16 *weight, float *master,
                               const float *grad, float *m, float *v,
                               std::size_t count, AdamStep opt, float scale) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float g = grad[i] * scale;
  const float diff = g - m[i];
  // torch.optim.AdamW uses lerp_, then mul_ followed by addcmul_.
  const float first = opt.alpha1 < 0.5f ? fmaf(opt.alpha1, diff, m[i])
                                        : fmaf(-(1.f - opt.alpha1), diff, g);
  const float previous = __fmul_rn(opt.beta2, v[i]);
  const float second = opt.alpha2 == 1.f
                           ? fmaf(g, g, previous)
                           : fmaf(opt.alpha2, __fmul_rn(g, g), previous);
  m[i] = first;
  v[i] = second;
  const float denom =
      __fmul_rn(sqrtf(second), opt.inv_sqrt_correction) + opt.eps;
  const float updated = fmaf(-opt.step_size, first / denom,
                             __fmul_rn(master[i], opt.decay_factor));
  master[i] = updated;
  weight[i] = __float2bfloat16_rn(updated);
}
} // namespace

AdamStep adam_step(const Adam &opt, std::uint64_t step, double lr,
                   double decay) {
  const double bc1 = 1. - std::pow(opt.beta1, step);
  const double sqrt_bc2 = std::sqrt(1. - std::pow(opt.beta2, step));
  // Scalar division by a CPU scalar in ATen multiplies by its reciprocal.
  return {float(opt.beta2),      float(1. - opt.beta1), float(1. - opt.beta2),
          float(opt.eps),        float(lr / bc1),       1.f / float(sqrt_bc2),
          float(1. - lr * decay)};
}

void adam_update(cudaStream_t stream, __nv_bfloat16 *weight, float *master,
                 const float *grad, float *m, float *v, std::size_t count,
                 AdamStep step, float scale) {
  constexpr unsigned threads = 256;
  update_weights<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      weight, master, grad, m, v, count, step, scale);
}
} // namespace rwkv
