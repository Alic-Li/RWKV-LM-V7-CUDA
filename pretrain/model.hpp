#pragma once
#include "gpu.hpp"
#include "weights.hpp"
#include <cuda_runtime_api.h>
#include <memory>

namespace rwkv {
struct MicroBatch {
  int lanes = 0, time = 0;
  std::vector<int, PinnedAllocator<int>> input, active_rows;
  std::vector<std::int64_t, PinnedAllocator<std::int64_t>> labels;
  std::vector<float, PinnedAllocator<float>> losses;
};

struct Adam {
  float lr, beta1, beta2, eps, decay, clip;
};

class Model {
public:
  Model(int device, const Checkpoint &input, const Shape &shape,
        const Weights &layout, int head_chunk, bool optimized_wkv,
        bool checkpoint_activations);
  ~Model();

  void activate() const;
  cudaStream_t stream() const;
  void synchronize() const;
  float *gradients();
  std::size_t parameter_count() const;
  void clear_gradients();
  void accumulate(MicroBatch &batch, float inv_tokens);
  float gradient_scale(float clip) const;
  void update(const Adam &opt, std::uint64_t step, float scale);
  void save(const std::string &path) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace rwkv
