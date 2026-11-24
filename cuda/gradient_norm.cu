#include "gradient_norm.hpp"
#include <algorithm>
#include <limits>

namespace rwkv {
namespace {
constexpr unsigned max_threads = 512;
std::size_t ceil_div(std::size_t n, std::size_t d) { return 1 + (n - 1) / d; }

__device__ float block_sum(float value) {
  __shared__ float scratch[max_threads];
  const unsigned lane = threadIdx.x;
  scratch[lane] = value;
  for (unsigned stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    __syncthreads();
    if (lane < stride) {
      value += scratch[lane + stride];
      scratch[lane] = value;
    }
  }
  if (lane < warpSize) {
    const unsigned mask = __activemask();
    for (unsigned stride = min(blockDim.x, 32u) / 2; stride; stride /= 2)
      value += __shfl_down_sync(mask, value, stride);
  }
  return value;
}
} // namespace

GradientNorm::Group GradientNorm::group(GradientSlice slice, unsigned start,
                                        int resident_threads) {
  const auto n = slice.count;
  const auto width = n >= 128 ? n / 4 : n;
  unsigned threads = 1;
  while (threads < max_threads && threads * 2 <= width)
    threads *= 2;
  const auto per_thread = ceil_div(n, threads);
  const auto blocks =
      per_thread < 256
          ? 1
          : std::max(ceil_div(per_thread, 256),
                     std::min<std::size_t>(resident_threads / threads,
                                           ceil_div(per_thread, 16)));
  if (blocks > std::numeric_limits<int>::max() - start)
    throw std::runtime_error("gradient norm grid exceeds CUDA limits");
  return {slice, threads, start, static_cast<unsigned>(blocks)};
}

GradientNorm::GradientNorm(cudaStream_t stream, int resident_threads,
                           const std::vector<GradientSlice> &slices)
    : groups_(slices.size()), partials_(0) {
  std::vector<Group> plan;
  std::vector<unsigned> owners;
  for (const auto &slice : slices) {
    auto g = group(slice, partials_, resident_threads);
    owners.insert(owners.end(), g.blocks, plan.size());
    plan.push_back(g);
    partials_ += g.blocks;
  }
  // The outer norm uses the same reduction on the vector of parameter norms.
  final_ = group({0, groups_}, 0, resident_threads);
  plan.push_back(final_);
  plan_ = Buffer<Group>(stream, plan.size());
  owners_ = Buffer<unsigned>(stream, owners.size());
  partial_ = Buffer<float>(stream, partials_);
  norms_ = Buffer<float>(stream, groups_);
  final_partial_ = Buffer<float>(stream, final_.blocks);
  result_ = Buffer<float>(stream, 1);
  CUDA(cudaMemcpyAsync(plan_.get(), plan.data(), plan.size() * sizeof(Group),
                       cudaMemcpyHostToDevice, stream));
  CUDA(cudaMemcpyAsync(owners_.get(), owners.data(),
                       owners.size() * sizeof(unsigned), cudaMemcpyHostToDevice,
                       stream));
  CUDA(cudaStreamSynchronize(stream));
}

namespace {
__global__ void reduce(const float *grad, const GradientNormGroup *plan,
                       const unsigned *owners, float *partial) {
  const auto g = plan[owners ? owners[blockIdx.x] : 0];
  const auto part = blockIdx.x - g.start;
  const float *values = grad + g.slice.offset;
  const auto n = g.slice.count;
  const auto stride = std::size_t(g.threads) * g.blocks;
  float sums[4] = {};
  if (threadIdx.x < g.threads) {
    std::size_t i = std::size_t(part) * g.threads + threadIdx.x;
    if (n >= 128) {
      for (; i < n / 4; i += stride) {
        const auto v = reinterpret_cast<const float4 *>(values)[i];
        sums[0] = fmaf(v.x, v.x, sums[0]);
        sums[1] = fmaf(v.y, v.y, sums[1]);
        sums[2] = fmaf(v.z, v.z, sums[2]);
        sums[3] = fmaf(v.w, v.w, sums[3]);
      }
      const auto tail = n / 4 * 4 + threadIdx.x;
      if (!part && tail < n)
        sums[0] = fmaf(values[tail], values[tail], sums[0]);
    } else {
      for (; i < n; i += stride)
        sums[0] = fmaf(values[i], values[i], sums[0]);
    }
  }
  const float sum = block_sum(((sums[0] + sums[1]) + sums[2]) + sums[3]);
  if (!threadIdx.x)
    partial[blockIdx.x] = sum;
}

__global__ void finish(const GradientNormGroup *plan, const float *partial,
                       float *norms) {
  const auto g = plan[blockIdx.x];
  float value = 0;
  if (threadIdx.x < g.threads)
    for (unsigned i = threadIdx.x; i < g.blocks; i += g.threads)
      value += partial[g.start + i];
  const float sum = block_sum(value);
  if (!threadIdx.x)
    norms[blockIdx.x] = __fsqrt_rn(sum);
}

} // namespace

float GradientNorm::compute(cudaStream_t stream, const float *gradient) const {
  reduce<<<partials_, max_threads, 0, stream>>>(gradient, plan_.get(),
                                                owners_.get(), partial_.get());
  finish<<<groups_, max_threads, 0, stream>>>(plan_.get(), partial_.get(),
                                              norms_.get());
  const Group *outer = plan_.get() + groups_;
  reduce<<<final_.blocks, final_.threads, 0, stream>>>(
      norms_.get(), outer, nullptr, final_partial_.get());
  finish<<<1, final_.threads, 0, stream>>>(outer, final_partial_.get(),
                                           result_.get());
  float norm;
  CUDA(cudaMemcpyAsync(&norm, result_.get(), sizeof(norm),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  return norm;
}
} // namespace rwkv
