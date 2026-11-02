# MXFP4 small-batch weight reuse

`test_mxfp4_weight_reuse.cpp` compares the actual `vec_mul_kgroup` dispatcher
with the original per-token GEMV. Run on an x86 Linux CPU supporting AVX-512
BF16, with the normal kt-kernel build dependencies and initialized submodules.
From `kt-kernel/`:

```bash
g++ -std=gnu++20 -O3 -ffast-math -DNDEBUG -march=native -fopenmp \
  -DFMT_HEADER_ONLY -DKTRANSFORMERS_CPU_ONLY=1 -DUSE_AMX_AVX_KERNEL=1 \
  -I. -Icpu_backend -I../third_party -I../third_party/llama.cpp \
  test/cpp/test_mxfp4_weight_reuse.cpp cpu_backend/shared_mem_buffer.cpp \
  -lnuma -lhwloc -pthread -o /tmp/test_mxfp4_weight_reuse
/tmp/test_mxfp4_weight_reuse
```

The expected result is 483 passing cases, 2,086,560 finite float outputs,
bitwise equality, and zero maximum absolute error. Coverage includes M=1..8,
real gate/up/down dimensions, short and non-multiple N, empty workers,
serial and threaded execution, FP32 and E8M0 scales, strided zero-copy weights,
and the existing M=1 natural-order path. M is the number of tokens routed to
one expert, which can differ from the speculative-token count.

For the optional warm-cache, single-thread microbenchmark:

```bash
taskset -c <allowed-cpu> /tmp/test_mxfp4_weight_reuse --benchmark
```

The optimized path shares weight loading and dequantization across M=2..6
tokens while preserving each token's accumulation order. M=1 and M>6 keep
their existing paths. Local microkernel timing does not establish end-to-end
model acceleration or measured DRAM traffic reduction.
