#include "model.hpp"
#include "layer_norm.hpp"
#include "value_bias.hpp"
#include "embedding.hpp"
#include "gradient_norm.hpp"

#ifdef RWKV_ROCM
#include "../hip/rwkv7_training_ops.hpp"
#else
#include "rwkv7_training_ops.hpp"
#endif
#include "gpu.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cublas_v2.h>
#include <unordered_map>

namespace rwkv {
#ifdef RWKV_ROCM
using hip_ops::alpha;
using hip_ops::alpha_backward;
using hip_ops::alpha_tiles;
using hip_ops::cmix;
using hip_ops::cmix_backward;
using hip_ops::head_loss;
using hip_ops::key_prepare;
using hip_ops::key_prepare_backward;
using hip_ops::mix6;
using hip_ops::mix6_backward;
using hip_ops::norm_gate;
using hip_ops::norm_gate_backward;
using hip_ops::relu_square;
using hip_ops::relu_square_backward;
using hip_ops::value_gate;
using hip_ops::value_gate_backward;
using hip_ops::wkv_backward;
using hip_ops::wkv_backward_v3;
using hip_ops::wkv_forward;
using hip_ops::wkv_forward_v3;
#endif
namespace {
constexpr int threads = 256;
using Count = std::size_t;

unsigned grid(Count n) { return (n + threads - 1) / threads; }

struct Device {
  explicit Device(int id) : id(id) {
    CUDA(cudaSetDevice(id));
    cudaDeviceProp properties;
    CUDA(cudaGetDeviceProperties(&properties, id));
    resident_threads = properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      blas_ok(cublasCreate(&blas));
      blas_ok(cublasSetStream(blas, stream));
      workspace = Buffer<unsigned char>(workspace_bytes);
      blas_ok(cublasSetWorkspace(blas, workspace.get(), workspace_bytes));
    } catch (...) {
      if (blas)
        cublasDestroy(blas);
      (void)cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Device() {
    (void)cudaStreamSynchronize(stream);
    cublasDestroy(blas);
    (void)cudaStreamDestroy(stream);
  }
  static void blas_ok(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(std::string("GPU BLAS: ") +
                               cublasGetStatusString(status));
  }
  static constexpr Count workspace_bytes = 32 * 1024 * 1024;
  Buffer<unsigned char> workspace;
  int id, resident_threads;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
};

__device__ float fl(BF x) { return __bfloat162float(x); }
__device__ BF bf(float x) { return __float2bfloat16_rn(x); }
__device__ Count tid() { return Count(blockIdx.x) * blockDim.x + threadIdx.x; }

__global__ void to_bf(const float *input, BF *out, Count n) {
  const auto i = tid();
  if (i < n)
    out[i] = bf(input[i]);
}
__global__ void add_bf_to_float(const BF *x, float *y, Count n, float scale = 1.f) {
  const auto i = tid();
  if (i < n)
    y[i] += fl(bf(fl(x[i]) * scale));
}
__global__ void add(BF *x, const BF *y, BF *out, Count n) {
  const auto i = tid();
  if (i < n)
    out[i] = bf(fl(x[i]) + fl(y[i]));
}
__global__ void add_three(BF *x, const BF *y, const BF *z, Count n) {
  const auto i = tid();
  if (i < n)
    x[i] = bf(fl(bf(fl(z[i]) + fl(x[i]))) + fl(y[i]));
}
__global__ void add_bias(BF *x, const BF *bias, Count n, int c) {
  const auto i = tid();
  if (i < n)
    x[i] = bf(fl(x[i]) + fl(bias[i % c]));
}

enum class Activation { Tanh, Sigmoid };
__global__ void activate_element(BF *x, Count n, Activation fn) {
  const auto i = tid();
  if (i < n) {
    const float u = fl(x[i]);
    x[i] = bf(fn == Activation::Tanh ? tanhf(u) : 1.f / (1.f + expf(-u)));
  }
}
__global__ void activate_backward(BF *dx, const BF *activation, Count n,
                                  Activation fn, bool recompute) {
  const auto i = tid();
  if (i < n) {
    const float a = fl(activation[i]);
    const float g = fl(dx[i]);
    if (!recompute) {
      // The optimized TorchScript backward fuses the elementwise derivative
      // in FP32 and casts its result back to BF16.
      const float derivative = fn == Activation::Tanh ? 1.f - a * a : (1.f - a) * a;
      dx[i] = bf(g * derivative);
    } else if (fn == Activation::Tanh) {
      const BF derivative = bf(1.f - fl(bf(a * a)));
      dx[i] = bf(g * fl(derivative));
    } else {
      const BF scaled = bf(g * fl(bf(1.f - a)));
      dx[i] = bf(fl(scaled) * a);
    }
  }
}

__global__ void gather(const BF *table, const int *indices, BF *out, Count n,
                       int c) {
  const auto i = tid();
  if (i < n)
    out[i] = table[Count(indices[i / c]) * c + i % c];
}
__global__ void scatter(const BF *input, const int *indices, BF *out, Count n,
                        int c, float scale) {
  const auto i = tid();
  if (i < n)
    out[Count(indices[i / c]) * c + i % c] = bf(fl(input[i]) * scale);
}
__global__ void check_finite(const float *gradient, Count n, int *bad) {
  bool invalid = false;
  for (auto i = tid(); i < n; i += Count(gridDim.x) * threads)
    invalid |= !isfinite(gradient[i]);
  if (__syncthreads_or(invalid) && !threadIdx.x)
    atomicExch(bad, 1);
}

// Each BF16 parameter receives a BF16 gradient from one backward pass.
__global__ void finish_microbatch(float *grad, float *accumulated, Count n) {
  const auto i = tid();
  if (i < n) {
    const float value = fl(bf(grad[i]));
    if (accumulated)
      accumulated[i] += value;
    else
      grad[i] = value;
  }
}

struct Param {
  BF *w = nullptr;
  float *g = nullptr;
  Count offset = 0, size = 0;
};
struct NormWeights {
  Param w, b;
};
struct LowRank {
  Param first, second;
  int rank;
};
struct BlockWeights {
  NormWeights ln1, ln2, group_norm;
  std::array<Param, 6> mix;
  Param r, k, v, out, w0, a0, v0, kk, ka, rk;
  LowRank w, a, v_res, g;
  Param ffn_mix, ffn_key, ffn_value;
};

struct NormCache {
  View<BF> out;
  View<float> mean, inv;
};
struct BlockCache {
  NormCache ln1, ln2;
  std::array<View<BF>, 6> mixed;
  View<BF> r, raw_k, raw_v, v, w, w_hidden, a_pre, a_hidden, a, v_pre, v_hidden,
      g_hidden, g;
  View<BF> k, neg, kka, recurrent, gated, middle, ffn_mix, ffn_act, output;
  View<float> key_inv, state, sa, group_mean, group_inv;
};

} // namespace

struct Model::Impl {
  Impl(int id, const Checkpoint &input, const Shape &s, const Weights &layout,
       int max_rows, bool accumulate, bool clip, int chunk, bool optimized_wkv,
       bool checkpoint)
      : gpu(id), shape(s), layout(layout), c(s.channels), h(c / 64), ffn(s.ffn),
        head_chunk(chunk), optimized_wkv(optimized_wkv), checkpoint(checkpoint),
        sort_bytes(embedding_sort_bytes(stream(), max_rows)),
        arena(arena_bytes(s, max_rows, chunk, checkpoint, sort_bytes)) {
    for (const auto &weight : layout) {
      const auto size = weight.size();
      if (size > SIZE_MAX / sizeof(float) - total)
        throw std::runtime_error("model size overflow");
      total += size;
    }
    Count largest = 0;
    for (const auto &weight : layout)
      largest = std::max(largest, weight.size());
    checkpoint_staging.resize(std::min(largest, Count(1) << 20));
    weights = Buffer<BF>(total);
    master = Buffer<float>(total);
    gradient = Buffer<float>(total);
    first_moment = Buffer<float>(total);
    second_moment = Buffer<float>(total);
    if (accumulate)
      accumulated = Buffer<float>(total);
    zero(first_moment.get(), total);
    zero(second_moment.get(), total);
    Count off = 0;
    for (const auto &weight : layout) {
      const auto values = input.read(weight);
      const Param p{weights.get() + off, gradient.get() + off, off,
                    weight.size()};
      if (values.size() != p.size)
        throw std::runtime_error("wrong element count: " + weight.name);
      CUDA(cudaMemcpyAsync(master.get() + off, values.data(),
                           p.size * sizeof(float), cudaMemcpyHostToDevice,
                           stream()));
      to_bf<<<grid(p.size), threads, 0, stream()>>>(master.get() + off, p.w,
                                                    p.size);
      CUDA(cudaStreamSynchronize(stream()));
      params.emplace(weight.name, p);
      off += p.size;
    }
    for (const auto &weight : layout)
      read_weight(params.at(weight.name), [](Count, const std::uint16_t *, Count) {});
    embed = params.at("emb.weight");
    head = params.at("head.weight");
    ln0 = norm_weights("blocks.0.ln0");
    ln_out = norm_weights("ln_out");
    blocks.reserve(s.layers);
    for (int i = 0; i < s.layers; ++i)
      blocks.push_back(block_weights(i));
    if (clip)
      gradient_norm = std::make_unique<GradientNorm>(
          stream(), gpu.resident_threads, norm_slices());
    residuals.reserve(s.layers + 1);
    if (!checkpoint)
      saved.reserve(s.layers);
  }

  // Prime the same D2H transfers used during saves, including the driver's
  // first use of its large-copy path, while all ranks are still initializing.
  template<class Consume>
  void read_weight(Param p, Consume consume) {
    for (Count off = 0; off < p.size;) {
      const auto count = std::min(checkpoint_staging.size(), p.size - off);
      CUDA(cudaMemcpyAsync(checkpoint_staging.data(), p.w + off,
                           count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                           stream()));
      CUDA(cudaStreamSynchronize(stream()));
      consume(off, checkpoint_staging.data(), count);
      off += count;
    }
  }

  cudaStream_t stream() const { return gpu.stream; }
  // Upper bounds in bytes per row. Each forward block retains the mixed,
  // projected and FFN activations, FP32 WKV state and normalization statistics.
  // Reverse traverses one block at a time. Recompute retains only residuals.
  static Count arena_bytes(const Shape &s, Count rows, Count chunk,
                           bool checkpoint, Count sort_bytes) {
    const Count c = s.channels;
    const Count ranks = Count(s.decay) + s.alpha + s.value + s.gate;
    const Count forward =
        checked_sum(16384, checked_product(rows, 96 * c + 2 * ranks +
                                                     2 * Count(s.ffn) + 32));
    const Count backward =
        checked_sum(32768, checked_product(rows, 112 * c + 4 * ranks +
                                                     2 * Count(s.ffn) + 32));
    const Count tape = checked_product(
        s.layers,
        checkpoint ? checked_sum(256, checked_product(rows, 2 * c)) : forward);
    const Count base = checked_sum(
        tape, checked_sum(16384, checked_product(rows, 24 * c + 64)));
    const Count head = checked_sum(
        checked_product(2 * c, s.vocab),
        checked_product(std::min(rows, chunk), 2 * Count(s.vocab) + 4 * c));
    const Count block = checked_sum(backward, forward);
    return checked_sum(base, std::max({block, head, sort_bytes}));
  }
  View<BF> bf_buffer(Count size) const { return arena.take<BF>(size); }
  View<float> fp_buffer(Count size) const { return arena.take<float>(size); }
  template <class T> void zero(T *p, Count size) const {
    CUDA(cudaMemsetAsync(p, 0, size * sizeof(T), stream()));
  }

  std::vector<GradientSlice> norm_slices() const {
    // clip_grad_norm_ follows named_parameters() and skips the unused
    // value-residual parameters in the first block.
    std::vector<GradientSlice> slices;
    const auto append = [&](std::initializer_list<Param> parameters) {
      for (const auto &p : parameters)
        slices.push_back({p.offset, p.size});
    };
    append({embed});
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      const auto &b = blocks[i];
      append({b.ln1.w, b.ln1.b, b.ln2.w, b.ln2.b});
      if (i == 0)
        append({ln0.w, ln0.b});
      for (const auto &p : b.mix)
        append({p});
      append({b.w.first, b.w.second, b.w0, b.a.first, b.a.second, b.a0});
      if (i > 0)
        append({b.v_res.first, b.v_res.second, b.v0});
      append({b.g.first, b.g.second, b.kk, b.ka, b.rk, b.r, b.k, b.v, b.out,
              b.group_norm.w, b.group_norm.b, b.ffn_mix, b.ffn_key, b.ffn_value});
    }
    append({ln_out.w, ln_out.b, head});
    return slices;
  }
  NormWeights norm_weights(const std::string &name) const {
    return {params.at(name + ".weight"), params.at(name + ".bias")};
  }
  BlockWeights block_weights(int layer) const {
    const auto b = "blocks." + std::to_string(layer) + ".";
    const auto a = b + "att.";
    const auto att = [&](const std::string &name) {
      return params.at(a + name);
    };
    return {norm_weights(b + "ln1"),
            norm_weights(b + "ln2"),
            norm_weights(a + "ln_x"),
            {att("x_r"), att("x_w"), att("x_k"), att("x_v"), att("x_a"),
             att("x_g")},
            att("receptance.weight"),
            att("key.weight"),
            att("value.weight"),
            att("output.weight"),
            att("w0"),
            att("a0"),
            att("v0"),
            att("k_k"),
            att("k_a"),
            att("r_k"),
            {att("w1"), att("w2"), shape.decay},
            {att("a1"), att("a2"), shape.alpha},
            {att("v1"), att("v2"), shape.value},
            {att("g1"), att("g2"), shape.gate},
            params.at(b + "ffn.x_k"),
            params.at(b + "ffn.key.weight"),
            params.at(b + "ffn.value.weight")};
  }
  // Map row-major A * B to column-major B^T * A^T for cuBLAS.
  void gemm(int m, int cols, int k, const BF *a, bool ta, const BF *b, bool tb,
            void *out, cudaDataType_t dtype, float beta) const {
    const float alpha = 1.f;
#ifdef RWKV_ROCM
    Device::blas_ok(rocblas_gemm_ex(
        gpu.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
        ta ? CUBLAS_OP_T : CUBLAS_OP_N, cols, m, k, &alpha, b, CUDA_R_16BF,
        tb ? k : cols, a, CUDA_R_16BF, ta ? m : k, &beta, out, dtype, cols, out,
        dtype, cols, CUDA_R_32F, rocblas_gemm_algo_standard, 0, 0));
#else
    Device::blas_ok(cublasGemmEx(
        gpu.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
        ta ? CUBLAS_OP_T : CUBLAS_OP_N, cols, m, k, &alpha, b, CUDA_R_16BF,
        tb ? k : cols, a, CUDA_R_16BF, ta ? m : k, &beta, out, dtype, cols,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
#endif
  }
  View<BF> linear(const BF *x, Param p, int in, int out,
                  bool weight_transposed = true) const {
    auto y = bf_buffer(Count(rows) * out);
    gemm(rows, out, in, x, false, p.w, weight_transposed, y.get(), CUDA_R_16BF,
         0.f);
    return y;
  }
  void linear_backward_rows(const BF *x, Param p, const BF *dy, BF *dx, int m,
                            int in, int out, bool weight_transposed) const {
    gemm(m, in, out, dy, false, p.w, !weight_transposed, dx, CUDA_R_16BF, 0.f);
    if (weight_transposed)
      gemm(out, in, m, dy, true, x, false, p.g, CUDA_R_32F, 1.f);
    else
      gemm(in, out, m, x, true, dy, false, p.g, CUDA_R_32F, 1.f);
  }
  View<BF> linear_backward(const BF *x, Param p, const BF *dy, int in, int out,
                           bool weight_transposed = true) const {
    auto dx = bf_buffer(Count(rows) * in);
    linear_backward_rows(x, p, dy, dx.get(), rows, in, out, weight_transposed);
    return dx;
  }
  NormCache normalize(const BF *x, NormWeights p, const BF *residual = nullptr,
                      BF *sum = nullptr) const {
    NormCache norm{bf_buffer(n), fp_buffer(rows), fp_buffer(rows)};
    layer_norm(stream(), rows, c, x, residual, sum, p.w.w, p.b.w,
               norm.out.get(), norm.mean.get(), norm.inv.get());
    return norm;
  }
  View<BF> normalize_backward(const BF *x, const BF *dy, const NormCache &f,
                              NormWeights p) const {
    auto dx = bf_buffer(n);
    layer_norm_backward(arena, stream(), rows, c, x, dy, p.w.w, f.mean.get(),
                        f.inv.get(), dx.get(), p.w.g, p.b.g);
    return dx;
  }
  View<BF> low_rank(const BF *x, LowRank p, View<BF> &hidden) const {
    hidden = linear(x, p.first, c, p.rank, false);
    return linear(hidden.get(), p.second, p.rank, c, false);
  }
  View<BF> low_rank_activated(const BF *x, LowRank p, View<BF> &hidden,
                              Activation activation) const {
    hidden = linear(x, p.first, c, p.rank, false);
    const Count elems = Count(rows) * p.rank;
    activate_element<<<grid(elems), threads, 0, stream()>>>(hidden.get(), elems,
                                                            activation);
    return linear(hidden.get(), p.second, p.rank, c, false);
  }
  View<BF> low_rank_backward(const BF *x, LowRank p, const BF *hidden,
                             const BF *dy) const {
    const auto dh = linear_backward(hidden, p.second, dy, p.rank, c, false);
    return linear_backward(x, p.first, dh.get(), c, p.rank, false);
  }
  View<BF> low_rank_activated_backward(const BF *x, LowRank p, const BF *hidden,
                                       const BF *dy,
                                       Activation activation) const {
    auto dh = linear_backward(hidden, p.second, dy, p.rank, c, false);
    const Count elems = Count(rows) * p.rank;
    activate_backward<<<grid(elems), threads, 0, stream()>>>(dh.get(), hidden,
                                                             elems, activation, checkpoint);
    return linear_backward(x, p.first, dh.get(), c, p.rank, false);
  }
  BlockCache forward_block(const BF *x, BF *first,
                           const BlockWeights &p, bool need_output) const {
    BlockCache f;
    f.ln1 = normalize(x, p.ln1);
    BF *mixed[6], *mix_w[6];
    for (int i = 0; i < 6; ++i) {
      f.mixed[i] = bf_buffer(n);
      mixed[i] = f.mixed[i].get();
      mix_w[i] = p.mix[i].w;
    }
    mix6(stream(), rows, time, c, f.ln1.out.get(), mix_w, mixed);

    f.r = linear(mixed[0], p.r, c, c);
    f.w = low_rank_activated(mixed[1], p.w, f.w_hidden, Activation::Tanh);
    add_bias<<<grid(n), threads, 0, stream()>>>(f.w.get(), p.w0.w, n, c);
    f.raw_k = linear(mixed[2], p.k, c, c);
    f.raw_v = linear(mixed[3], p.v, c, c);
    if (first) {
      f.v_pre = low_rank(mixed[3], p.v_res, f.v_hidden);
      f.v = bf_buffer(n);
      value_gate(stream(), rows, c, f.raw_v.get(), first, p.v0.w, f.v_pre.get(),
                 f.v.get());
    }
    BF *v = first ? f.v.get() : f.raw_v.get();
    f.a_pre = low_rank(mixed[4], p.a, f.a_hidden);
    f.a = bf_buffer(n);
    alpha(stream(), rows, c, p.a0.w, f.a_pre.get(), f.a.get());
    f.g = low_rank_activated(mixed[5], p.g, f.g_hidden, Activation::Sigmoid);
    f.k = bf_buffer(n);
    f.neg = bf_buffer(n);
    f.kka = bf_buffer(n);
    f.key_inv = fp_buffer(Count(rows) * h);
    key_prepare(stream(), rows, c, f.raw_k.get(), p.kk.w, f.a.get(), p.ka.w,
                f.k.get(), f.neg.get(), f.kka.get(), f.key_inv.get());

    f.recurrent = bf_buffer(n);
    f.state = fp_buffer(Count(rows) * c * 4);
    f.sa = fp_buffer(n);
    const auto forward = optimized_wkv ? wkv_forward_v3 : wkv_forward;
    forward(stream(), batch, time, h, f.r.get(), f.w.get(), f.k.get(), v,
            f.neg.get(), f.kka.get(), f.recurrent.get(), f.state.get(),
            f.sa.get());

    f.gated = bf_buffer(n);
    f.group_mean = fp_buffer(Count(rows) * h);
    f.group_inv = fp_buffer(Count(rows) * h);
    norm_gate(stream(), rows, c, f.recurrent.get(), f.r.get(), f.k.get(), v,
              p.rk.w, p.group_norm.w.w, p.group_norm.b.w, f.g.get(),
              f.gated.get(), f.group_mean.get(), f.group_inv.get());
    const auto att = linear(f.gated.get(), p.out, c, c);
    f.middle = bf_buffer(n);
    f.ln2 = normalize(x, p.ln2, att.get(), f.middle.get());
    f.ffn_mix = bf_buffer(n);
    cmix(stream(), rows, time, c, f.ln2.out.get(), p.ffn_mix.w,
         f.ffn_mix.get());
    f.ffn_act = linear(f.ffn_mix.get(), p.ffn_key, c, ffn);
    relu_square(stream(), Count(rows) * ffn, f.ffn_act.get());
    if (!need_output)
      return f;
    const auto ffn_y = linear(f.ffn_act.get(), p.ffn_value, ffn, c);
    f.output = bf_buffer(n);
    add<<<grid(n), threads, 0, stream()>>>(f.middle.get(), ffn_y.get(),
                                           f.output.get(), n);
    return f;
  }

  View<BF> backward_block(const BF *x, BF *dy, BF *first, BF *d_first,
                          const BlockCache &f, const BlockWeights &p) const {
    auto da_ffn = linear_backward(f.ffn_act.get(), p.ffn_value, dy, ffn, c);
    relu_square_backward(stream(), Count(rows) * ffn, da_ffn.get(),
                         f.ffn_act.get());
    const auto d_mixed_ffn =
        linear_backward(f.ffn_mix.get(), p.ffn_key, da_ffn.get(), c, ffn);
    const auto d_ln2 = bf_buffer(n);
    cmix_backward(stream(), rows, time, c, d_mixed_ffn.get(), f.ln2.out.get(),
                  p.ffn_mix.w, d_ln2.get(), p.ffn_mix.g);
    const auto d_middle =
        normalize_backward(f.middle.get(), d_ln2.get(), f.ln2, p.ln2);
    add<<<grid(n), threads, 0, stream()>>>(dy, d_middle.get(), d_middle.get(),
                                           n);

    const auto d_gate =
        linear_backward(f.gated.get(), p.out, d_middle.get(), c, c);
    BF *v = first ? f.v.get() : f.raw_v.get();
    const auto d_recurrent = bf_buffer(n), dr = bf_buffer(n), dk = bf_buffer(n);
    const auto dv = bf_buffer(n), dg = bf_buffer(n);
    norm_gate_backward(stream(), rows, c, d_gate.get(), f.recurrent.get(),
                       f.r.get(), f.k.get(), v, p.rk.w, p.group_norm.w.w,
                       p.group_norm.b.w, f.g.get(), f.group_mean.get(),
                       f.group_inv.get(), d_recurrent.get(), dr.get(), dk.get(),
                       dv.get(), p.rk.g, p.group_norm.w.g, p.group_norm.b.g,
                       dg.get());
    const auto wr = bf_buffer(n), ww = bf_buffer(n), wk = bf_buffer(n),
               wv = bf_buffer(n);
    const auto wa = bf_buffer(n), wz = bf_buffer(n);
    const auto backward = optimized_wkv ? wkv_backward_v3 : wkv_backward;
    backward(stream(), batch, time, h, f.r.get(), f.w.get(), f.k.get(), v,
             f.neg.get(), f.kka.get(), d_recurrent.get(), f.state.get(),
             f.sa.get(), wr.get(), ww.get(), wk.get(), wv.get(), wa.get(),
             wz.get());
    add<<<grid(n), threads, 0, stream()>>>(dr.get(), wr.get(), dr.get(), n);
    add<<<grid(n), threads, 0, stream()>>>(dk.get(), wk.get(), dk.get(), n);
    if (first)
      add<<<grid(n), threads, 0, stream()>>>(dv.get(), wv.get(), dv.get(), n);
    else
      add_three<<<grid(n), threads, 0, stream()>>>(dv.get(), wv.get(), d_first,
                                                   n);

    const auto d_raw_k = bf_buffer(n), d_alpha = bf_buffer(n);
    key_prepare_backward(stream(), rows, c, f.raw_k.get(), p.kk.w, f.a.get(),
                         p.ka.w, dk.get(), wa.get(), wz.get(), f.key_inv.get(),
                         d_raw_k.get(), p.kk.g, d_alpha.get(), p.ka.g);
    const auto d_a_pre = bf_buffer(n), d_a0 = bf_buffer(c);
    const auto partial = fp_buffer(Count(alpha_tiles(rows)) * c);
    alpha_backward(stream(), rows, c, d_alpha.get(), p.a0.w, f.a_pre.get(),
                   d_a_pre.get(), partial.get(), d_a0.get());
    add_bf_to_float<<<grid(c), threads, 0, stream()>>>(d_a0.get(), p.a0.g, c);
    value_bias_gradient(arena, stream(), rows, c, gpu.resident_threads,
                        ww.get(), p.w0.g);

    std::array<View<BF>, 6> gradients;
    gradients[0] = linear_backward(f.mixed[0].get(), p.r, dr.get(), c, c);
    gradients[1] = low_rank_activated_backward(
        f.mixed[1].get(), p.w, f.w_hidden.get(), ww.get(), Activation::Tanh);
    gradients[2] = linear_backward(f.mixed[2].get(), p.k, d_raw_k.get(), c, c);
    gradients[4] = low_rank_backward(f.mixed[4].get(), p.a, f.a_hidden.get(),
                                     d_a_pre.get());
    gradients[5] = low_rank_activated_backward(
        f.mixed[5].get(), p.g, f.g_hidden.get(), dg.get(), Activation::Sigmoid);
    if (first) {
      const auto d_raw_v = bf_buffer(n), d_base = bf_buffer(n),
                 d_v_pre = bf_buffer(n);
      const auto d_pre_fp = fp_buffer(n);
      value_gate_backward(stream(), rows, c, dv.get(), f.raw_v.get(), first,
                          p.v0.w, f.v_pre.get(), d_raw_v.get(), d_base.get(),
                          d_pre_fp.get());
      add<<<grid(n), threads, 0, stream()>>>(d_first, d_base.get(), d_first, n);
      value_bias_gradient(arena, stream(), rows, c, gpu.resident_threads,
                          d_pre_fp.get(), p.v0.g);
      to_bf<<<grid(n), threads, 0, stream()>>>(d_pre_fp.get(), d_v_pre.get(),
                                               n);
      gradients[3] =
          linear_backward(f.mixed[3].get(), p.v, d_raw_v.get(), c, c);
      const auto d_mix = low_rank_backward(f.mixed[3].get(), p.v_res,
                                           f.v_hidden.get(), d_v_pre.get());
      add<<<grid(n), threads, 0, stream()>>>(gradients[3].get(), d_mix.get(),
                                             gradients[3].get(), n);
    } else {
      gradients[3] = linear_backward(f.mixed[3].get(), p.v, dv.get(), c, c);
    }

    BF *d_mixed[6], *mix_w[6];
    float *d_mix_w[6];
    for (int i = 0; i < 6; ++i) {
      d_mixed[i] = gradients[i].get();
      mix_w[i] = p.mix[i].w;
      d_mix_w[i] = p.mix[i].g;
    }
    const auto d_ln1 = bf_buffer(n);
    mix6_backward(stream(), rows, time, c, f.ln1.out.get(), mix_w, d_mixed,
                  d_ln1.get(), d_mix_w);
    auto dx = normalize_backward(x, d_ln1.get(), f.ln1, p.ln1);
    add<<<grid(n), threads, 0, stream()>>>(dx.get(), d_middle.get(), dx.get(),
                                           n);
    return dx;
  }

  Device gpu;
  std::unique_ptr<GradientNorm> gradient_norm;
  Shape shape;
  Weights layout;
  int batch = 0, time = 0, rows = 0;
  int c, h, ffn, head_chunk;
  Count n = 0, total = 0;
  bool optimized_wkv, checkpoint;
  Count sort_bytes;
  mutable Arena arena;
  Buffer<int> bad{1};
  std::vector<int, PinnedAllocator<int>> finite_status{0};
  std::vector<std::uint16_t, PinnedAllocator<std::uint16_t>> checkpoint_staging;
  std::vector<View<BF>> residuals;
  std::vector<BlockCache> saved;
  Buffer<BF> weights;
  Buffer<float> master, gradient, accumulated, first_moment, second_moment;
  std::unordered_map<std::string, Param> params;
  Param embed, head;
  NormWeights ln0, ln_out;
  std::vector<BlockWeights> blocks;
};

Model::Model(int id, const Checkpoint &input, const Shape &s,
             const Weights &layout, int head_chunk, bool optimized_wkv,
             bool checkpoint_activations, int max_rows, bool accumulate,
             bool clip)
    : impl_(std::make_unique<Impl>(id, input, s, layout, max_rows, accumulate,
                                   clip, head_chunk, optimized_wkv,
                                   checkpoint_activations)) {}
Model::~Model() {
  (void)cudaSetDevice(impl_->gpu.id);
  impl_.reset();
}
cudaStream_t Model::stream() const { return impl_->stream(); }
void Model::synchronize() const {
  CUDA(cudaStreamSynchronize(stream()));
}
std::size_t Model::parameter_count() const { return impl_->total; }
float *Model::gradients() const {
  return impl_->accumulated.get() ? impl_->accumulated.get() : impl_->gradient.get();
}
void Model::clear_gradients() { impl_->zero(gradients(), parameter_count()); }

void Model::accumulate(MicroBatch &batch, std::size_t total_targets) {
  const int active = batch.active_rows.size();
  if (!active)
    return;
  auto &m = *impl_;
  const auto s = stream();
  if (m.accumulated.get())
    m.zero(m.gradient.get(), m.total);
  m.batch = batch.lanes;
  m.time = batch.time;
  m.rows = m.batch * m.time;
  m.n = Count(m.rows) * m.c;
  m.arena.rewind(0);
  auto tokens = m.arena.take<int>(m.rows),
       active_rows = m.arena.take<int>(active);
  auto labels = m.arena.take<std::int64_t>(active);
  CUDA(cudaMemcpyAsync(tokens.get(), batch.input.data(), m.rows * sizeof(int),
                       cudaMemcpyHostToDevice, s));
  CUDA(cudaMemcpyAsync(active_rows.get(), batch.active_rows.data(),
                       active * sizeof(int), cudaMemcpyHostToDevice, s));
  CUDA(cudaMemcpyAsync(labels.get(), batch.labels.data(),
                       active * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                       s));

  const auto emb = m.bf_buffer(m.n);
  gather<<<grid(m.n), threads, 0, s>>>(m.embed.w, tokens.get(), emb.get(), m.n,
                                       m.c);
  auto ln0 = m.normalize(emb.get(), m.ln0);
  auto &residuals = m.residuals;
  auto &saved = m.saved;
  residuals.clear();
  saved.clear();
  residuals.push_back(ln0.out);
  auto first = m.bf_buffer(m.n);
  const auto copy = [&](View<BF> dest, View<BF> src) {
    CUDA(cudaMemcpyAsync(dest.get(), src.get(), m.n * sizeof(BF),
                         cudaMemcpyDeviceToDevice, s));
  };
  for (const auto &block : m.blocks) {
    const bool initial = residuals.size() == 1;
    auto output = m.checkpoint ? m.bf_buffer(m.n) : View<BF>{};
    const auto mark = m.arena.mark();
    auto f = m.forward_block(residuals.back().get(),
                             initial ? nullptr : first.get(), block, true);
    if (initial)
      copy(first, f.raw_v);
    if (m.checkpoint) {
      copy(output, f.output);
      residuals.push_back(output);
      m.arena.rewind(mark);
    } else {
      residuals.push_back(f.output);
      saved.push_back(f);
    }
  }
  const auto ln_out = m.normalize(residuals.back().get(), m.ln_out);
  auto d_hidden = m.bf_buffer(m.n);
  m.zero(d_hidden.get(), m.n);
  batch.losses.resize(active);
  const auto head_mark = m.arena.mark();
  {
    // The fused loss returns a local mean. Its scalar upstream is applied
    // after the chunked head backward has written its BF16 gradients.
    const float head_scale = __bfloat162float(__float2bfloat16_rn(
        static_cast<float>(static_cast<double>(active) / total_targets)));
    const auto head_grad = m.bf_buffer(m.head.size);
    const int chunk = std::min(m.head_chunk, m.rows);
    const auto logits = m.bf_buffer(Count(chunk) * m.shape.vocab);
    const auto x = m.bf_buffer(Count(chunk) * m.c);
    const auto dx = m.bf_buffer(Count(chunk) * m.c);
    const auto losses = m.fp_buffer(active);
    for (int start = 0; start < active;) {
      const int count = std::min(chunk, active - start);
      const auto elems = Count(count) * m.c;
      const auto *idx = active_rows.get() + start;
      gather<<<grid(elems), threads, 0, s>>>(ln_out.out.get(), idx, x.get(),
                                             elems, m.c);
      // Keep GEMM shapes fixed, including the final partial chunk. Padded
      // rows contribute zero to the weight gradient and are never scattered.
      if (count < chunk)
        m.zero(x.get() + elems, Count(chunk - count) * m.c);
      m.gemm(chunk, m.shape.vocab, m.c, x.get(), false, m.head.w, true,
             logits.get(), CUDA_R_16BF, 0.f);
      head_loss(s, count, m.shape.vocab, logits.get(), labels.get() + start,
                losses.get() + start, 1.f / active);
      if (count < chunk)
        m.zero(logits.get() + Count(count) * m.shape.vocab,
               Count(chunk - count) * m.shape.vocab);
      m.gemm(chunk, m.c, m.shape.vocab, logits.get(), false, m.head.w, false,
             dx.get(), CUDA_R_16BF, 0.f);
      m.gemm(m.shape.vocab, m.c, chunk, logits.get(), true, x.get(), false,
             head_grad.get(), CUDA_R_16BF, start == 0 ? 0.f : 1.f);
      scatter<<<grid(elems), threads, 0, s>>>(dx.get(), idx, d_hidden.get(),
                                              elems, m.c, head_scale);
      start += count;
    }
    add_bf_to_float<<<grid(m.head.size), threads, 0, s>>>(
        head_grad.get(), m.head.g, m.head.size, head_scale);
    CUDA(cudaMemcpyAsync(batch.losses.data(), losses.get(),
                         active * sizeof(float), cudaMemcpyDeviceToHost, s));
  }
  m.arena.rewind(head_mark);
  auto dy = m.normalize_backward(residuals.back().get(), d_hidden.get(), ln_out,
                                 m.ln_out);
  const auto d_first = m.bf_buffer(m.n);
  m.zero(d_first.get(), m.n);
  for (int layer = int(m.blocks.size()) - 1; layer >= 0; --layer) {
    const auto &p = m.blocks[layer];
    BF *base = layer ? first.get() : nullptr;
    auto *x = residuals[layer].get();
    residuals.pop_back();
    const auto mark = m.arena.mark();
    auto f =
        m.checkpoint ? m.forward_block(x, base, p, false) : std::move(saved[layer]);
    copy(dy, m.backward_block(x, dy.get(), base, d_first.get(), f, p));
    m.arena.rewind(mark);
  }
  const auto d_emb = m.normalize_backward(emb.get(), dy.get(), ln0, m.ln0);
  embedding_gradient(m.arena, m.sort_bytes, s, m.rows, m.c, tokens.get(),
                     d_emb.get(), m.embed.g);
  finish_microbatch<<<grid(m.total), threads, 0, s>>>(
      m.gradient.get(), m.accumulated.get(), m.total);
  CUDA(cudaGetLastError());
}

float Model::gradient_scale(float clip) const {
  const auto s = stream();
  auto &m = *impl_;
  if (clip > 0.f) {
    const float norm = m.gradient_norm->compute(s, gradients());
    if (!std::isfinite(norm))
      throw std::runtime_error("non-finite gradient norm");
    // PyTorch scalar / tensor uses reciprocal followed by multiplication.
    return std::min(1.f, (1.f / (norm + 1e-6f)) * clip);
  }
  const auto &bad = m.bad;
  m.zero(bad.get(), 1);
  check_finite<<<std::min(grid(m.total), 4096u), threads, 0, s>>>(
      gradients(), m.total, bad.get());
  CUDA(cudaMemcpyAsync(m.finite_status.data(), bad.get(), sizeof(int),
                       cudaMemcpyDeviceToHost, s));
  synchronize();
  if (m.finite_status.front())
    throw std::runtime_error("non-finite gradient");
  return 1.f;
}

// Prime the optimizer kernel using temporary local state. No checkpoint weight
// or moment participates in this launch.
void Model::warm_optimizer() {
  auto &m = *impl_;
  m.arena.rewind(0);
  auto *state = m.arena.take<float>(4).get();
  auto weight = m.arena.take<BF>(1);
  m.zero(state, 4);
  AdamStep options{};
  options.eps = 1.f;
  adam_update(stream(), weight.get(), state, state + 1, state + 2, state + 3, 1,
              options, 1.f);
  CUDA(cudaGetLastError());
}

void Model::update(const Adam &opt, std::uint64_t step, float scale) {
  auto &m = *impl_;
  const auto standard = adam_step(opt, step, opt.lr, 0.);
  const auto decayed = adam_step(opt, step, opt.lr, opt.decay);
  const auto faster = adam_step(opt, step, 2. * opt.lr, 0.);
  for (const auto &w : m.layout) {
    const auto p = m.params.at(w.name);
    const float *grad = gradients() + p.offset;
    // The first layer supplies v_first and has no value-residual branch.
    if (w.name == "blocks.0.att.v0" || w.name == "blocks.0.att.v1" ||
        w.name == "blocks.0.att.v2")
      continue;
    const bool use_decay = w.dims.size() == 2 && w.name.find(".weight") != std::string::npos;
    const auto &options = w.name.find(".att.w0") != std::string::npos
                              ? faster : (use_decay ? decayed : standard);
    adam_update(stream(), p.w, m.master.get() + p.offset, grad,
                m.first_moment.get() + p.offset, m.second_moment.get() + p.offset,
                p.size, options, scale);
  }
  CUDA(cudaGetLastError());
}

void Model::save(const std::string &path) const {
  save_weights(path, impl_->layout, [&](const auto &w) {
    const auto p = impl_->params.at(w.name);
    std::vector<std::uint16_t> values(p.size);
    impl_->read_weight(p, [&](Count off, const std::uint16_t *data, Count count) {
      std::copy_n(data, count, values.data() + off);
    });
    return values;
  });
}
} // namespace rwkv
