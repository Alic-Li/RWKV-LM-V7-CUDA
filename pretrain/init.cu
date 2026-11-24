#include "cli.hpp"
#include "gpu.hpp"
#include "weights.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <cusolverDn.h>
#include <iostream>
#include <random>

namespace rwkv {
namespace {
void solver_ok(cusolverStatus_t result) {
  if (result != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("cuSOLVER QR failed with status " +
                             std::to_string(result));
}

__global__ void save_signs(const float *q, float *signs, int m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    signs[i] = q[std::size_t(i) * (m + 1)] < 0 ? -1.f : 1.f;
}

__global__ void pack_q(const float *q, const float *signs, __nv_bfloat16 *out,
                       int rows, int cols, float gain) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(rows) * cols)
    return;
  const auto r = i / cols, c = i % cols;
  const auto src = rows < cols ? r * cols + c : c * rows + r;
  out[i] = __float2bfloat16_rn(q[src] * signs[rows < cols ? r : c] * gain);
}

class Initializer {
public:
  explicit Initializer(std::uint64_t seed) : rng_(seed) {
    solver_ok(cusolverDnCreate(&solver_));
  }
  ~Initializer() { cusolverDnDestroy(solver_); }

  std::vector<std::uint16_t> make(const Weight &w, const Shape &s) {
    const auto &name = w.name;
    const auto n = w.size();
    if (name == "emb.weight") {
      std::uniform_real_distribution<float> uniform(-1e-4f, 1e-4f);
      std::vector<std::uint16_t> data(n);
      for (auto &v : data)
        v = llm_infer::float_to_bf16_bits(uniform(rng_));
      return data;
    }
    if (name == "head.weight")
      return orthogonal(
          w, 0.5 * std::sqrt(std::max(1.0, double(s.vocab) / s.channels)));
    if (name == "ln_out.weight")
      return fill(n, 1.f);
    if (name == "ln_out.bias")
      return fill(n, 0.f);

    const auto split = name.find('.', 7);
    const int layer = integer<int>(name.substr(7, split - 7));
    const auto field = name.substr(split + 1);
    const double remaining = 1.0 - double(layer) / s.layers;
    const double depth = s.layers > 1 ? double(layer) / (s.layers - 1) : 0.0;

    if (field.find(".bias") != std::string::npos ||
        field == "att.output.weight" || field == "ffn.value.weight")
      return fill(n, 0.f);
    if (field == "att.ln_x.weight")
      return fill(n, std::pow(double(layer + 1) / s.layers, 0.7));
    if (field.compare(0, 2, "ln") == 0)
      return fill(n, 1.f);
    if (field == "att.r_k")
      return fill(n, -0.04f);
    if (field == "att.k_a")
      return fill(n, 1.02f);
    if (field == "att.receptance.weight" || field == "att.value.weight" ||
        field == "ffn.key.weight")
      return orthogonal(w, 1.f);
    if (field == "att.key.weight")
      return orthogonal(w, 0.1f);
    if (field == "att.w2" || field == "att.a2" || field == "att.v2" ||
        field == "att.g2")
      return orthogonal(
          w, 0.1 * std::sqrt(std::max(1.0, double(w.dims[0]) / w.dims[1])));
    if (field == "att.w1" || field == "att.a1" || field == "att.v1" ||
        field == "att.g1")
      return fill(n, 0.f);

    std::vector<std::uint16_t> data(n);
    for (int i = 0; i < s.channels; ++i) {
      const double position = double(i) / s.channels;
      const double linear = double(i) / (s.channels - 1) - 0.5;
      const double z = ((i % 64) - 31.5) / 31.5;
      const double zigzag = z * std::abs(z);
      double value;
      if (field == "att.x_r" || field == "att.x_g")
        value = 1 - std::pow(position, 0.2 * remaining);
      else if (field == "att.x_w" || field == "att.x_a")
        value = 1 - std::pow(position, 0.9 * remaining);
      else if (field == "att.x_k" || field == "att.x_v")
        value = 1 - std::pow(position, 0.7 * remaining);
      else if (field == "ffn.x_k")
        value = 1 - std::pow(position, std::pow(remaining, 4));
      else if (field == "att.w0")
        value = -5.5 + 6 * std::pow(linear + 0.5, 1 + std::pow(depth, 0.3)) +
                zigzag * 2.5;
      else if (field == "att.a0")
        value = -0.19 + zigzag * 0.3 + linear * 0.4;
      else if (field == "att.v0")
        value = 0.73 - linear * 0.4;
      else if (field == "att.k_k")
        value = 0.71 - linear * 0.1;
      else
        throw std::logic_error("missing initializer for " + name);
      data[i] = llm_infer::float_to_bf16_bits(value);
    }
    return data;
  }

private:
  static std::vector<std::uint16_t> fill(std::size_t n, float v) {
    return std::vector<std::uint16_t>(n, llm_infer::float_to_bf16_bits(v));
  }

  std::vector<std::uint16_t> orthogonal(const Weight &w, float gain) {
    const int rows = w.dims[0], cols = w.dims[1];
    const int m = std::max(rows, cols), n = std::min(rows, cols);
    const std::size_t size = w.size();
    Buffer<float> q(nullptr, size), tau(nullptr, n), sign(nullptr, n);
    Buffer<int> info(nullptr, 1);
    std::vector<float> host(size);
    std::normal_distribution<float> gaussian;
    for (auto &v : host)
      v = gaussian(rng_);
    CUDA(cudaMemcpyAsync(q.get(), host.data(), size * sizeof(float),
                         cudaMemcpyHostToDevice));
    int qr_size, q_size;
    solver_ok(cusolverDnSgeqrf_bufferSize(solver_, m, n, q.get(), m, &qr_size));
    solver_ok(cusolverDnSorgqr_bufferSize(solver_, m, n, n, q.get(), m,
                                          tau.get(), &q_size));
    Buffer<float> work(nullptr, std::max(qr_size, q_size));
    solver_ok(cusolverDnSgeqrf(solver_, m, n, q.get(), m, tau.get(), work.get(),
                               qr_size, info.get()));
    int qr_info;
    CUDA(cudaMemcpyAsync(&qr_info, info.get(), sizeof(int),
                         cudaMemcpyDeviceToHost));
    save_signs<<<(n + 255) / 256, 256>>>(q.get(), sign.get(), m, n);
    solver_ok(cusolverDnSorgqr(solver_, m, n, n, q.get(), m, tau.get(),
                               work.get(), q_size, info.get()));
    int q_info;
    CUDA(cudaMemcpyAsync(&q_info, info.get(), sizeof(int),
                         cudaMemcpyDeviceToHost));
    Buffer<__nv_bfloat16> packed(nullptr, size);
    pack_q<<<(size + 255) / 256, 256>>>(q.get(), sign.get(), packed.get(), rows,
                                        cols, gain);
    CUDA(cudaGetLastError());
    std::vector<std::uint16_t> result(size);
    CUDA(cudaMemcpyAsync(result.data(), packed.get(), size * sizeof(result[0]),
                         cudaMemcpyDeviceToHost));
    CUDA(cudaStreamSynchronize(nullptr));
    if (qr_info || q_info)
      throw std::runtime_error("QR failed for " + w.name);
    return result;
  }
  std::mt19937_64 rng_;
  cusolverDnHandle_t solver_ = nullptr;
};
} // namespace
} // namespace rwkv

int main(int argc, char **argv) try {
  using namespace rwkv;
  if (argc == 1 || std::string(argv[1]) == "--help") {
    std::cout << "Usage: rwkv7_init OUTPUT.pth --layers N --channels C [--ffn "
                 "F] [--vocab V] [--seed N] [--device N]\n";
    return argc == 1 ? 1 : 0;
  }
  Options options(argc, argv, 2);
  const int layers = options.number("--layers", 0);
  const int c = options.number("--channels", 0);
  if (c < 64 || c % 64 || c > INT32_MAX / 4)
    throw std::invalid_argument("--channels must be a positive multiple of 64 "
                                "within the supported range");
  const int ffn = options.number("--ffn", c * 4);
  const int vocab = options.number("--vocab", 65536);
  const auto seed = options.number<std::uint64_t>("--seed", 1234);
  const int device = options.number("--device", 0);
  options.finish();
  const auto rank = [c](double factor) {
    return std::max(32, int(std::nearbyint(factor * std::sqrt(c) / 32)) * 32);
  };
  Shape shape{layers, c, ffn, vocab, rank(2.5), rank(2.5), rank(1.7), rank(5)};
  const auto layout = weight_layout(shape);
  CUDA(cudaSetDevice(device));
  Initializer init(seed);
  save_weights(argv[1], layout,
               [&](const auto &w) { return init.make(w, shape); });
  std::cout << "Wrote " << argv[1] << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "rwkv7_init: " << error.what() << '\n';
  return 1;
}
