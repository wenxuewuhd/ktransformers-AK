// kernel_compute_probe.c — cache-resident throughput of the MXFP4×Q8_0 NEON dot.
// All data fits in L1/L2 -> measures the pure COMPUTE roofline of the kernel
// (GB of weight bytes consumed per second per core), no DRAM in the picture.
//   gcc -O3 -march=armv8.2-a+fp16+dotprod -o /tmp/kcp tools/kernel_compute_probe.c
// Usage: kcp <mode: mxfp4|q8_0> <iters_millions>
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
typedef uint16_t ggml_half;
typedef struct { uint8_t e; uint8_t qs[QK/2]; } block_mxfp4;   // 17B
typedef struct { ggml_half d; int8_t qs[QK]; } block_q8_0;     // 34B

static const int8_t kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

static inline float e8m0_half(uint8_t x) {
  uint32_t bits = (x < 2) ? (uint32_t)0x00200000 << x : (uint32_t)(x - 1) << 23;
  float f; memcpy(&f, &bits, 4); return f;
}
static inline float fp16_to_fp32(ggml_half h) {
  __fp16 t; memcpy(&t, &h, 2); return (float)t;
}

// verbatim port of the NEON path in ggml_vec_dot_mxfp4_q8_0
static float dot_mxfp4(int nb, const block_mxfp4* x, const block_q8_0* y) {
  const int8x16_t values = vld1q_s8(kvalues_mxfp4);
  const uint8x16_t m4b = vdupq_n_u8(0x0f);
  float sumf = 0;
  for (int ib = 0; ib + 1 < nb; ib += 2) {
    uint8x16_t q4_0 = vld1q_u8(x[ib+0].qs);
    uint8x16_t q4_1 = vld1q_u8(x[ib+1].qs);
    int8x16_t q8_0 = vld1q_s8(y[ib+0].qs);
    int8x16_t q8_1 = vld1q_s8(y[ib+0].qs + 16);
    int8x16_t q8_2 = vld1q_s8(y[ib+1].qs);
    int8x16_t q8_3 = vld1q_s8(y[ib+1].qs + 16);
    int8x16_t b0 = vqtbl1q_s8(values, vandq_u8(q4_0, m4b));
    int8x16_t b1 = vqtbl1q_s8(values, vshrq_n_u8(q4_0, 4));
    int8x16_t b2 = vqtbl1q_s8(values, vandq_u8(q4_1, m4b));
    int8x16_t b3 = vqtbl1q_s8(values, vshrq_n_u8(q4_1, 4));
    int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), b0, q8_0), b1, q8_1);
    int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), b2, q8_2), b3, q8_3);
    sumf += e8m0_half(x[ib+0].e) * fp16_to_fp32(y[ib+0].d) * vaddvq_s32(p1)
          + e8m0_half(x[ib+1].e) * fp16_to_fp32(y[ib+1].d) * vaddvq_s32(p2);
  }
  return sumf;
}

// Q8_0×Q8_0 reference (same structure as ggml_vec_dot_q8_0_q8_0 NEON)
static float dot_q8(int nb, const block_q8_0* x, const block_q8_0* y) {
  float sumf = 0;
  for (int ib = 0; ib + 1 < nb; ib += 2) {
    int8x16_t a0 = vld1q_s8(x[ib+0].qs), a1 = vld1q_s8(x[ib+0].qs + 16);
    int8x16_t a2 = vld1q_s8(x[ib+1].qs), a3 = vld1q_s8(x[ib+1].qs + 16);
    int8x16_t b0 = vld1q_s8(y[ib+0].qs), b1 = vld1q_s8(y[ib+0].qs + 16);
    int8x16_t b2 = vld1q_s8(y[ib+1].qs), b3 = vld1q_s8(y[ib+1].qs + 16);
    int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), a0, b0), a1, b1);
    int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), a2, b2), a3, b3);
    sumf += fp16_to_fp32(x[ib+0].d) * fp16_to_fp32(y[ib+0].d) * vaddvq_s32(p1)
          + fp16_to_fp32(x[ib+1].d) * fp16_to_fp32(y[ib+1].d) * vaddvq_s32(p2);
  }
  return sumf;
}

static double now_s(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "mxfp4";
  long iters = (argc > 2 ? atol(argv[2]) : 1) * 1000000L;
  const int K = (argc > 3 ? atoi(argv[3]) : 4096), nb = K / QK;             // one gate/up row: K=4096
  block_mxfp4* wx = aligned_alloc(64, nb * sizeof(block_mxfp4));   // 2176B  (L1)
  block_q8_0* wq = aligned_alloc(64, nb * sizeof(block_q8_0));     // 4352B  (L1)
  block_q8_0* act = aligned_alloc(64, nb * sizeof(block_q8_0));    // 4352B  (L1)
  srand(7);
  for (int i = 0; i < nb; i++) {
    wx[i].e = 120 + rand() % 8;
    for (int j = 0; j < 16; j++) wx[i].qs[j] = rand() & 0xff;
    wq[i].d = 0x3400; act[i].d = 0x3400;       // ~0.25 in fp16
    for (int j = 0; j < 32; j++) { wq[i].qs[j] = rand() % 64 - 32; act[i].qs[j] = rand() % 64 - 32; }
  }
  volatile float sink = 0;
  double t0 = now_s();
  if (!strcmp(mode, "mxfp4")) {
    for (long i = 0; i < iters; i++) sink += dot_mxfp4(nb, wx, act);
    double dt = now_s() - t0;
    double wbytes = (double)iters * nb * sizeof(block_mxfp4);
    printf("MXFP4 dot: %.2f GB(w)/s/core  %.1f ns/row  (%.2f cyc/B @2.6GHz) sink=%f\n",
           wbytes / dt / 1e9, dt / iters * 1e9, 2.6e9 * dt / wbytes, sink);
  } else {
    for (long i = 0; i < iters; i++) sink += dot_q8(nb, wq, act);
    double dt = now_s() - t0;
    double wbytes = (double)iters * nb * sizeof(block_q8_0);
    printf("Q8_0  dot: %.2f GB(w)/s/core  %.1f ns/row  (%.2f cyc/B @2.6GHz) sink=%f\n",
           wbytes / dt / 1e9, dt / iters * 1e9, 2.6e9 * dt / wbytes, sink);
  }
  return 0;
}
