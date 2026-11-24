#pragma once
#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace rwkv::hip_ops {
using BF = __hip_bfloat16;
void wkv_forward(hipStream_t, int, int, int, BF *, BF *, BF *, BF *, BF *,
                 BF *, BF *, float *, float *);
void wkv_forward_v3(hipStream_t, int, int, int, BF *, BF *, BF *, BF *, BF *,
                    BF *, BF *, float *, float *);
void wkv_backward(hipStream_t, int, int, int, BF *, BF *, BF *, BF *, BF *,
                  BF *, BF *, float *, float *, BF *, BF *, BF *, BF *, BF *,
                  BF *);
void wkv_backward_v3(hipStream_t, int, int, int, BF *, BF *, BF *, BF *, BF *,
                     BF *, BF *, float *, float *, BF *, BF *, BF *, BF *, BF *,
                     BF *);

void mix6(hipStream_t, int rows, int time, int channels, BF *x, BF *const *mix,
          BF *const *out);
void mix6_backward(hipStream_t, int rows, int time, int channels, BF *x,
                   BF *const *mix, BF *const *dy, BF *dx, float *const *dmix);
void key_prepare(hipStream_t, int rows, int c, BF *k, BF *kk, BF *a, BF *ka,
                 BF *key, BF *neg, BF *kka, float *inv);
void key_prepare_backward(hipStream_t, int rows, int c, BF *k, BF *kk, BF *a,
                          BF *ka, BF *dkey, BF *dneg, BF *dkka, float *inv,
                          BF *dk, float *dkk, BF *da, float *dka);
void norm_gate(hipStream_t, int rows, int c, BF *x, BF *r, BF *k, BF *v,
               BF *rk, BF *weight, BF *bias, BF *g, BF *y, float *mean,
               float *rstd);
void norm_gate_backward(hipStream_t, int rows, int c, BF *dy, BF *x, BF *r,
                        BF *k, BF *v, BF *rk, BF *weight, BF *bias, BF *g,
                        float *mean, float *rstd, BF *dx, BF *dr, BF *dk,
                        BF *dv, float *drk, float *dw, float *db, BF *dg);
void alpha(hipStream_t, int rows, int c, BF *a0, BF *a12, BF *out);
int alpha_tiles(int rows);
void alpha_backward(hipStream_t, int rows, int c, BF *dy, BF *a0, BF *a12,
                    BF *da12, float *partial, BF *da0);
void value_gate(hipStream_t, int rows, int c, BF *v, BF *first, BF *v0,
                BF *v12, BF *out);
void value_gate_backward(hipStream_t, int rows, int c, BF *dy, BF *v,
                         BF *first, BF *v0, BF *v12, BF *dv, BF *dfirst,
                         float *dpre);
void cmix(hipStream_t, int rows, int time, int c, BF *x, BF *mix, BF *out);
void cmix_backward(hipStream_t, int rows, int time, int c, BF *dy, BF *x,
                   BF *mix, BF *dx, float *dmix);
void relu_square(hipStream_t, std::int64_t n, BF *x);
void relu_square_backward(hipStream_t, std::int64_t n, BF *dy, BF *out);
void head_loss(hipStream_t, int rows, int vocab, BF *logits,
               const std::int64_t *labels, float *loss, float inv_tokens);
} // namespace rwkv::hip_ops
