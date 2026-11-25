#pragma once
#include "gpu.hpp"
#include <vector>

namespace rwkv {
// Gradient slices follow the parameter order used by clip_grad_norm_.
struct GradientSlice {
  std::size_t offset, count;
};

struct GradientNormGroup {
  GradientSlice slice;
  unsigned threads, start, blocks;
};

class GradientNorm {
public:
  GradientNorm(cudaStream_t stream, int resident_threads,
               const std::vector<GradientSlice> &slices);
  float compute(cudaStream_t stream, const float *gradient) const;

private:
  using Group = GradientNormGroup;
  static Group group(GradientSlice slice, unsigned start, int resident_threads);
  unsigned groups_, partials_;
  Group final_;
  mutable std::vector<float, PinnedAllocator<float>> host_result_{0.f};
  Buffer<Group> plan_;
  Buffer<unsigned> owners_;
  Buffer<float> partial_, norms_, final_partial_, result_;
};
} // namespace rwkv
