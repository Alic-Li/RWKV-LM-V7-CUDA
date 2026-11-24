#include "model.hpp"

#include "rwkv7_training_ops.hpp"
#include "gpu.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cublas_v2.h>
#include <unordered_map>

namespace rwkv {
namespace {
constexpr int threads = 256;
using Count = std::size_t;

unsigned grid(Count n) { return (n + threads - 1) / threads; }

struct Device {
  explicit Device(int id) : id(id) {
    CUDA(cudaSetDevice(id));
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      blas_ok(cublasCreate(&blas));
      blas_ok(cublasSetStream(blas, stream));
      cudaMemPool_t pool;
      CUDA(cudaDeviceGetDefaultMemPool(&pool, id));
      std::uint64_t retain = UINT64_MAX;
      CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                   &retain));
    } catch (...) {
      if (blas)
        cublasDestroy(blas);
      cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Device() {
    cudaStreamSynchronize(stream);
    cublasDestroy(blas);
    cudaStreamDestroy(stream);
  }
  static void blas_ok(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(std::string("cuBLAS: ") +
                               cublasGetStatusString(status));
  }
  int id;
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
__global__ void add_bf_to_float(const BF *x, float *y, Count n) {
  const auto i = tid();
  if (i < n)
    y[i] += fl(x[i]);
}
__global__ void add(BF *x, const BF *y, BF *out, Count n) {
  const auto i = tid();
  if (i < n)
    out[i] = bf(fl(x[i]) + fl(y[i]));
}
__global__ void add_three(BF *x, const BF *y, const BF *z, Count n) {
  const auto i = tid();
  if (i < n)
    x[i] = bf(fl(bf(fl(x[i]) + fl(y[i]))) + fl(z[i]));
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
                                  Activation fn) {
  const auto i = tid();
  if (i < n) {
    const float a = fl(activation[i]);
    dx[i] =
        bf(fl(dx[i]) * (fn == Activation::Tanh ? 1.f - a * a : a * (1.f - a)));
  }
}

template <typename T> __device__ float float_value(T x) { return x; }
template <> __device__ float float_value(BF x) { return fl(x); }

// One warp-sized channel tile per block, with 8-way reduction over rows.
template <typename T>
__global__ void sum_columns(const T *x, float *out, int rows, int c) {
  __shared__ float partial[8][32];
  const int col = int(blockIdx.x) * 32 + threadIdx.x;
  const int lane = threadIdx.y;
  float s = 0;
  if (col < c) {
    const int begin = int(blockIdx.y) * 128;
    for (int r = begin + lane; r < min(rows, begin + 128); r += blockDim.y)
      s += float_value(x[Count(r) * c + col]);
  }
  partial[lane][threadIdx.x] = s;
  __syncthreads();
  if (lane == 0 && col < c) {
    for (int y = 1; y < 8; ++y)
      s += partial[y][threadIdx.x];
    atomicAdd(out + col, s);
  }
}

// LayerNorm follows the row-wise FP32 reductions in rwkv_lightning_cuda.
__device__ float warp_sum(float x) {
  for (int off = 16; off > 0; off /= 2)
    x += __shfl_down_sync(0xffffffff, x, off);
  return x;
}
__device__ float block_sum(float x, float *scratch) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  x = warp_sum(x);
  if (!lane)
    scratch[warp] = x;
  __syncthreads();
  if (!warp) {
    x = warp_sum(lane < threads / 32 ? scratch[lane] : 0.f);
    if (!lane)
      scratch[0] = x;
  }
  __syncthreads();
  const float sum = scratch[0];
  __syncthreads();
  return sum;
}

__global__ void layer_norm(const BF *x, const BF *residual, BF *sum,
                           const BF *w, const BF *b, BF *out, float *mean_out,
                           float *inv_out, int c) {
  __shared__ float scratch[threads / 32];
  const auto off = Count(blockIdx.x) * c;
  if (residual) {
    for (int i = threadIdx.x; i < c; i += threads)
      sum[off + i] = bf(fl(x[off + i]) + fl(residual[off + i]));
    x = sum;
  }
  float s = 0;
  for (int i = threadIdx.x; i < c; i += threads)
    s += fl(x[off + i]);
  const float mean = block_sum(s, scratch) / c;
  s = 0;
  for (int i = threadIdx.x; i < c; i += threads) {
    const float d = fl(x[off + i]) - mean;
    s += d * d;
  }
  const float inv = rsqrtf(block_sum(s, scratch) / c + 1e-5f);
  if (!threadIdx.x) {
    mean_out[blockIdx.x] = mean;
    inv_out[blockIdx.x] = inv;
  }
  for (int i = threadIdx.x; i < c; i += threads)
    out[off + i] = bf((fl(x[off + i]) - mean) * inv * fl(w[i]) + fl(b[i]));
}

__global__ void layer_norm_input_backward(const BF *x, const BF *dy,
                                          const BF *w, const float *mean,
                                          const float *inv, BF *dx, int c) {
  __shared__ float scratch[threads / 32];
  const auto off = Count(blockIdx.x) * c;
  const float m = mean[blockIdx.x], r = inv[blockIdx.x];
  float s = 0, p = 0;
  for (int i = threadIdx.x; i < c; i += threads) {
    const float g = fl(dy[off + i]) * fl(w[i]);
    s += g;
    p += g * ((fl(x[off + i]) - m) * r);
  }
  s = block_sum(s, scratch) / c;
  p = block_sum(p, scratch) / c;
  for (int i = threadIdx.x; i < c; i += threads)
    dx[off + i] =
        bf(r * (fl(dy[off + i]) * fl(w[i]) - s - (fl(x[off + i]) - m) * r * p));
}

__global__ void layer_norm_param_backward(const BF *x, const BF *dy,
                                          const float *mean, const float *inv,
                                          float *dw, float *db, int rows,
                                          int c) {
  __shared__ float partial_w[8][32], partial_b[8][32];
  const int col = int(blockIdx.x) * 32 + threadIdx.x;
  const int lane = threadIdx.y;
  float w = 0, b = 0;
  if (col < c) {
    const int begin = int(blockIdx.y) * 128;
    for (int row = begin + lane; row < min(rows, begin + 128);
         row += blockDim.y) {
      const auto i = Count(row) * c + col;
      const float g = fl(dy[i]);
      b += g;
      w += g * (fl(x[i]) - mean[row]) * inv[row];
    }
  }
  partial_w[lane][threadIdx.x] = w;
  partial_b[lane][threadIdx.x] = b;
  __syncthreads();
  if (lane == 0 && col < c) {
    for (int y = 1; y < 8; ++y) {
      w += partial_w[y][threadIdx.x];
      b += partial_b[y][threadIdx.x];
    }
    atomicAdd(dw + col, w);
    atomicAdd(db + col, b);
  }
}

__global__ void gather(const BF *table, const int *indices, BF *out, Count n,
                       int c) {
  const auto i = tid();
  if (i < n)
    out[i] = table[Count(indices[i / c]) * c + i % c];
}
__global__ void scatter(const BF *input, const int *indices, BF *out, Count n,
                        int c) {
  const auto i = tid();
  if (i < n)
    out[Count(indices[i / c]) * c + i % c] = input[i];
}
__global__ void embedding_backward(const BF *input, const int *indices,
                                   float *out, Count n, int c) {
  const auto i = tid();
  if (i < n)
    atomicAdd(out + Count(indices[i / c]) * c + i % c, fl(input[i]));
}

__global__ void gradient_stats(const float *g, Count n, float *sum, int *bad,
                               bool clip) {
  __shared__ float scratch[threads / 32];
  float acc = 0.f;
  bool invalid = false;
  for (auto i = tid(); i < n; i += Count(gridDim.x) * threads) {
    const auto v = g[i];
    invalid |= !isfinite(v);
    if (clip)
      acc += v * v;
  }
  if (__syncthreads_or(invalid) && !threadIdx.x)
    atomicExch(bad, 1);
  if (clip) {
    const auto local = block_sum(acc, scratch);
    if (!threadIdx.x)
      atomicAdd(sum, local);
  }
}

__global__ void adam_update(BF *weight, float *master, const float *grad,
                            float *m, float *v, Count n, Adam opt,
                            float inv_bc1, float inv_bc2, float lr, float decay,
                            float scale) {
  const auto i = tid();
  if (i >= n)
    return;
  const float g = grad[i] * scale;
  const float a = m[i] = opt.beta1 * m[i] + (1.f - opt.beta1) * g;
  const float b = v[i] = opt.beta2 * v[i] + (1.f - opt.beta2) * g * g;
  const float p = master[i] * (1.f - lr * decay) -
                  lr * (a * inv_bc1) / (sqrtf(b * inv_bc2) + opt.eps);
  master[i] = p;
  weight[i] = bf(p);
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
  Buffer<BF> out;
  Buffer<float> mean, inv;
};
struct BlockCache {
  NormCache ln1, ln2;
  std::array<Buffer<BF>, 6> mixed;
  Buffer<BF> r, raw_k, raw_v, v, w, w_hidden, a_pre, a_hidden, a, v_pre,
      v_hidden, g_hidden, g;
  Buffer<BF> k, neg, kka, recurrent, gated, middle, ffn_mix, ffn_act, output;
  Buffer<float> key_inv, state, sa, group_mean, group_inv;
};

} // namespace

struct Model::Impl {
  Impl(int id, const Checkpoint &input, const Shape &s, const Weights &layout,
       int chunk, bool optimized_wkv, bool checkpoint)
      : gpu(id), shape(s), layout(layout), c(s.channels), h(c / 64), ffn(s.ffn),
        head_chunk(chunk), optimized_wkv(optimized_wkv),
        checkpoint(checkpoint) {
    for (const auto &weight : layout) {
      const auto size = weight.size();
      if (size > SIZE_MAX / sizeof(float) - total)
        throw std::runtime_error("model size overflow");
      total += size;
    }
    weights = bf_buffer(total);
    master = fp_buffer(total);
    gradient = fp_buffer(total);
    first_moment = fp_buffer(total);
    second_moment = fp_buffer(total);
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
    embed = params.at("emb.weight");
    head = params.at("head.weight");
    ln0 = norm_weights("blocks.0.ln0");
    ln_out = norm_weights("ln_out");
    blocks.reserve(s.layers);
    for (int i = 0; i < s.layers; ++i)
      blocks.push_back(block_weights(i));
  }

  cudaStream_t stream() const { return gpu.stream; }
  Buffer<BF> bf_buffer(Count size) const { return {stream(), size}; }
  Buffer<float> fp_buffer(Count size) const { return {stream(), size}; }
  template <class T> void zero(T *p, Count size) const {
    CUDA(cudaMemsetAsync(p, 0, size * sizeof(T), stream()));
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
    Device::blas_ok(cublasGemmEx(
        gpu.blas, tb ? CUBLAS_OP_T : CUBLAS_OP_N,
        ta ? CUBLAS_OP_T : CUBLAS_OP_N, cols, m, k, &alpha, b, CUDA_R_16BF,
        tb ? k : cols, a, CUDA_R_16BF, ta ? m : k, &beta, out, dtype, cols,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  Buffer<BF> linear(const BF *x, Param p, int in, int out,
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
  Buffer<BF> linear_backward(const BF *x, Param p, const BF *dy, int in,
                             int out, bool weight_transposed = true) const {
    auto dx = bf_buffer(Count(rows) * in);
    linear_backward_rows(x, p, dy, dx.get(), rows, in, out, weight_transposed);
    return dx;
  }
  NormCache normalize(const BF *x, NormWeights p, const BF *residual = nullptr,
                      BF *sum = nullptr) const {
    NormCache norm{bf_buffer(n), fp_buffer(rows), fp_buffer(rows)};
    layer_norm<<<rows, threads, 0, stream()>>>(x, residual, sum, p.w.w, p.b.w,
                                               norm.out.get(), norm.mean.get(),
                                               norm.inv.get(), c);
    return norm;
  }
  Buffer<BF> normalize_backward(const BF *x, const BF *dy, const NormCache &f,
                                NormWeights p) const {
    auto dx = bf_buffer(n);
    layer_norm_input_backward<<<rows, threads, 0, stream()>>>(
        x, dy, p.w.w, f.mean.get(), f.inv.get(), dx.get(), c);
    layer_norm_param_backward<<<dim3((c + 31) / 32, (rows + 127) / 128),
                                dim3(32, 8), 0, stream()>>>(
        x, dy, f.mean.get(), f.inv.get(), p.w.g, p.b.g, rows, c);
    return dx;
  }
  Buffer<BF> low_rank(const BF *x, LowRank p, Buffer<BF> &hidden) const {
    hidden = linear(x, p.first, c, p.rank, false);
    return linear(hidden.get(), p.second, p.rank, c, false);
  }
  Buffer<BF> low_rank_activated(const BF *x, LowRank p, Buffer<BF> &hidden,
                                Activation activation) const {
    hidden = linear(x, p.first, c, p.rank, false);
    const Count elems = Count(rows) * p.rank;
    activate_element<<<grid(elems), threads, 0, stream()>>>(hidden.get(), elems,
                                                            activation);
    return linear(hidden.get(), p.second, p.rank, c, false);
  }
  Buffer<BF> low_rank_backward(const BF *x, LowRank p, const BF *hidden,
                               const BF *dy) const {
    const auto dh = linear_backward(hidden, p.second, dy, p.rank, c, false);
    return linear_backward(x, p.first, dh.get(), c, p.rank, false);
  }
  Buffer<BF> low_rank_activated_backward(const BF *x, LowRank p,
                                         const BF *hidden, const BF *dy,
                                         Activation activation) const {
    auto dh = linear_backward(hidden, p.second, dy, p.rank, c, false);
    const Count elems = Count(rows) * p.rank;
    activate_backward<<<grid(elems), threads, 0, stream()>>>(dh.get(), hidden,
                                                             elems, activation);
    return linear_backward(x, p.first, dh.get(), c, p.rank, false);
  }
  template <class T> void sum(const T *x, Param p) const {
    sum_columns<<<dim3((c + 31) / 32, (rows + 127) / 128), dim3(32, 8), 0,
                  stream()>>>(x, p.g, rows, c);
  }
  BlockCache forward_block(const BF *x, BF *first,
                           const BlockWeights &p) const {
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
    const auto ffn_y = linear(f.ffn_act.get(), p.ffn_value, ffn, c);
    f.output = bf_buffer(n);
    add<<<grid(n), threads, 0, stream()>>>(f.middle.get(), ffn_y.get(),
                                           f.output.get(), n);
    return f;
  }

  Buffer<BF> backward_block(const BF *x, BF *dy, BF *first, BF *d_first,
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
    sum(ww.get(), p.w0);

    std::array<Buffer<BF>, 6> gradients;
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
      sum(d_pre_fp.get(), p.v0);
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
  Shape shape;
  Weights layout;
  int batch = 0, time = 0, rows = 0;
  int c, h, ffn, head_chunk;
  Count n = 0, total = 0;
  bool optimized_wkv, checkpoint;
  Buffer<BF> weights;
  Buffer<float> master, gradient, first_moment, second_moment;
  std::unordered_map<std::string, Param> params;
  Param embed, head;
  NormWeights ln0, ln_out;
  std::vector<BlockWeights> blocks;
};

Model::Model(int id, const Checkpoint &input, const Shape &s,
             const Weights &layout, int head_chunk, bool optimized_wkv,
             bool checkpoint_activations)
    : impl_(std::make_unique<Impl>(id, input, s, layout, head_chunk,
                                   optimized_wkv, checkpoint_activations)) {}
Model::~Model() {
  cudaSetDevice(impl_->gpu.id);
  impl_.reset();
}
void Model::activate() const { CUDA(cudaSetDevice(impl_->gpu.id)); }
cudaStream_t Model::stream() const { return impl_->stream(); }
void Model::synchronize() const {
  activate();
  CUDA(cudaStreamSynchronize(stream()));
}
std::size_t Model::parameter_count() const { return impl_->total; }
float *Model::gradients() { return impl_->gradient.get(); }
void Model::clear_gradients() {
  activate();
  impl_->zero(gradients(), parameter_count());
}

void Model::accumulate(MicroBatch &batch, float inv_tokens) {
  const int active = batch.active_rows.size();
  if (!active)
    return;
  activate();
  auto &m = *impl_;
  const auto s = stream();
  m.batch = batch.lanes;
  m.time = batch.time;
  m.rows = m.batch * m.time;
  m.n = Count(m.rows) * m.c;
  const Buffer<int> tokens(s, m.rows), active_rows(s, active);
  const Buffer<std::int64_t> labels(s, active);
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
  std::vector<Buffer<BF>> residuals;
  residuals.reserve(m.blocks.size() + 1);
  residuals.push_back(std::move(ln0.out));
  std::vector<BlockCache> saved;
  if (!m.checkpoint)
    saved.reserve(m.blocks.size());
  Buffer<BF> first;
  for (const auto &block : m.blocks) {
    auto f = m.forward_block(residuals.back().get(), first.get(), block);
    if (!first.get())
      first = std::move(f.raw_v);
    residuals.push_back(std::move(f.output));
    if (!m.checkpoint)
      saved.push_back(std::move(f));
  }
  const auto ln_out = m.normalize(residuals.back().get(), m.ln_out);
  auto d_hidden = m.bf_buffer(m.n);
  m.zero(d_hidden.get(), m.n);
  batch.losses.resize(active);
  {
    const int chunk = std::min(m.head_chunk, active);
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
      m.gemm(count, m.shape.vocab, m.c, x.get(), false, m.head.w, true,
             logits.get(), CUDA_R_16BF, 0.f);
      head_loss(s, count, m.shape.vocab, logits.get(), labels.get() + start,
                losses.get() + start, inv_tokens);
      m.linear_backward_rows(x.get(), m.head, logits.get(), dx.get(), count,
                             m.c, m.shape.vocab, true);
      scatter<<<grid(elems), threads, 0, s>>>(dx.get(), idx, d_hidden.get(),
                                              elems, m.c);
      start += count;
    }
    CUDA(cudaMemcpyAsync(batch.losses.data(), losses.get(),
                         active * sizeof(float), cudaMemcpyDeviceToHost, s));
  }
  auto dy = m.normalize_backward(residuals.back().get(), d_hidden.get(), ln_out,
                                 m.ln_out);
  const auto d_first = m.bf_buffer(m.n);
  m.zero(d_first.get(), m.n);
  for (int layer = int(m.blocks.size()) - 1; layer >= 0; --layer) {
    const auto &p = m.blocks[layer];
    BF *base = layer ? first.get() : nullptr;
    auto *x = residuals[layer].get();
    residuals.pop_back();
    auto f =
        m.checkpoint ? m.forward_block(x, base, p) : std::move(saved[layer]);
    if (!m.checkpoint && !layer)
      f.raw_v = std::move(first);
    dy = m.backward_block(x, dy.get(), base, d_first.get(), f, p);
  }
  const auto d_emb = m.normalize_backward(emb.get(), dy.get(), ln0, m.ln0);
  embedding_backward<<<grid(m.n), threads, 0, s>>>(d_emb.get(), tokens.get(),
                                                   m.embed.g, m.n, m.c);
  CUDA(cudaGetLastError());
}

float Model::gradient_scale(float clip) const {
  activate();
  const auto s = stream();
  const auto &m = *impl_;
  const auto sum = m.fp_buffer(1);
  const Buffer<int> bad(s, 1);
  m.zero(sum.get(), 1);
  m.zero(bad.get(), 1);
  gradient_stats<<<std::min(grid(m.total), 4096u), threads, 0, s>>>(
      m.gradient.get(), m.total, sum.get(), bad.get(), clip > 0.f);
  CUDA(cudaGetLastError());
  float norm2;
  int nonfinite;
  CUDA(cudaMemcpyAsync(&norm2, sum.get(), sizeof(float), cudaMemcpyDeviceToHost,
                       s));
  CUDA(cudaMemcpyAsync(&nonfinite, bad.get(), sizeof(int),
                       cudaMemcpyDeviceToHost, s));
  synchronize();
  if (nonfinite || !std::isfinite(norm2))
    throw std::runtime_error("non-finite gradient or gradient norm");
  return clip > 0 ? std::min(1.f, clip / (std::sqrt(norm2) + 1e-6f)) : 1.f;
}

void Model::update(const Adam &opt, std::uint64_t step, float scale) {
  activate();
  auto &m = *impl_;
  const float inv_bc1 = 1.0 / (1.0 - std::pow(double(opt.beta1), double(step)));
  const float inv_bc2 = 1.0 / (1.0 - std::pow(double(opt.beta2), double(step)));
  for (const auto &w : m.layout) {
    const auto p = m.params.at(w.name);
    const float lr =
        w.name.find(".att.w0") != std::string::npos ? 2 * opt.lr : opt.lr;
    const float decay =
        w.dims.size() == 2 && w.name.find(".weight") != std::string::npos
            ? opt.decay
            : 0.f;
    adam_update<<<grid(p.size), threads, 0, stream()>>>(
        p.w, m.master.get() + p.offset, p.g, m.first_moment.get() + p.offset,
        m.second_moment.get() + p.offset, p.size, opt, inv_bc1, inv_bc2, lr,
        decay, scale);
  }
  CUDA(cudaGetLastError());
}

void Model::save(const std::string &path) const {
  activate();
  save_weights(path, impl_->layout, [&](const auto &w) {
    const auto p = impl_->params.at(w.name);
    std::vector<std::uint16_t> values(p.size);
    CUDA(cudaMemcpyAsync(values.data(), p.w, p.size * sizeof(values[0]),
                         cudaMemcpyDeviceToHost, stream()));
    synchronize();
    return values;
  });
}
} // namespace rwkv
