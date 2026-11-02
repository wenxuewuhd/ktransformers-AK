// Compare actual optimized vec dispatch to the unchanged original GEMV.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "operators/amx/fp4-moe.hpp"

using Kernel = amx::GemmKernel224MXFP4SmallKGroup;
struct Aligned {
  void* ptr;
  explicit Aligned(size_t bytes) {
    ptr = std::aligned_alloc(64, ((std::max(bytes, size_t(1)) + 63) / 64) * 64);
    if (!ptr) throw std::bad_alloc();
  }
  ~Aligned() { std::free(ptr); }
};

struct Case {
  int m, n, k, padded_n, nth;
  bool compact, strided, natural;
  Aligned a_storage, b_storage, weights, old_c_storage, new_c_storage, input_storage;
  std::shared_ptr<Kernel::BufferA> a;
  std::shared_ptr<Kernel::BufferB> b;
  std::shared_ptr<Kernel::BufferC> old_c, new_c;

  Case(int m_, int n_, int k_, int format, bool natural_ = false)
      : m(m_),
        n(n_),
        k(k_),
        padded_n((n + 31) / 32 * 32),
        nth((n + 255) / 256 + 1),
        compact(format != 0),
        strided(format == 2),
        natural(natural_),
        a_storage(Kernel::BufferA::required_size(m, k)),
        b_storage(Kernel::BufferB::required_size(padded_n, k, 32, compact)),
        weights(size_t(padded_n) * (k / 2 + 64) + 64),
        old_c_storage(Kernel::BufferC::required_size(m, padded_n)),
        new_c_storage(Kernel::BufferC::required_size(m, padded_n)),
        input_storage(size_t(m) * k * sizeof(ggml_bf16_t)) {
    a = std::make_shared<Kernel::BufferA>(m, k, a_storage.ptr);
    if (strided) {
      b = std::make_shared<Kernel::BufferB>(padded_n, k, 32, b_storage.ptr, compact,
                                            typename Kernel::BufferB::Base::ScaleOnly{});
      b->set_external_weights(static_cast<uint8_t*>(weights.ptr) + 56, k / 2 + 64);
    } else {
      b = std::make_shared<Kernel::BufferB>(padded_n, k, 32, b_storage.ptr, compact);
    }
    old_c = std::make_shared<Kernel::BufferC>(m, padded_n, old_c_storage.ptr);
    new_c = std::make_shared<Kernel::BufferC>(m, padded_n, new_c_storage.ptr);
    std::mt19937 gen(731 + m * 17 + n * 3 + k + format);
    auto* input = static_cast<ggml_bf16_t*>(input_storage.ptr);
    for (int i = 0; i < m * k; ++i) {
      // Finite inputs, signs and magnitudes varied; no cancellation shortcut.
      float value = static_cast<int>(gen() % 2001) / 1000.0f - 1.0f;
      input[i] = GGML_FP32_TO_BF16(value);
    }
    if (natural)
      a->from_mat_natural(m, input, 0, 1);
    else
      a->from_mat(m, input, 0, 1);
    for (int row = 0; row < padded_n; ++row) {
      auto* w = b->get_submat(padded_n, k, row, 0);
      for (int byte = 0; byte < k / 2; ++byte) w[byte] = gen() & 0xff;
      for (int g = 0; g < k / 32; ++g) {
        if (compact)
          b->scale_e8[row * (k / 32) + g] = 123 + gen() % 8;
        else
          b->d[row * (k / 32) + g] = .02f + (gen() % 191) / 100.0f;
      }
    }
  }

  void old_worker(int ith) {
#if defined(__AVX512BF16__)
    if (m == 1 && a->natural_order) {
      Kernel::fp4_mat_vec_kgroup_natural(n, k, a.get(), b.get(), old_c.get(), ith, nth);
      return;
    }
#endif
    Kernel::fp4_mat_vec_kgroup(m, n, k, 32, a.get(), b.get(), old_c.get(), ith, nth);
  }
  void new_worker(int ith) { amx::vec_mul_kgroup(m, n, k, 32, a, b, new_c, ith, nth); }

  void run(bool optimized, bool parallel) {
    auto worker = [&](int ith) { optimized ? new_worker(ith) : old_worker(ith); };
    if (parallel) {
      std::vector<std::thread> threads;
      for (int ith = 0; ith < nth; ++ith) threads.emplace_back(worker, ith);
      for (auto& t : threads) t.join();
    } else
      for (int ith = 0; ith < nth; ++ith) worker(ith);
  }

  bool compare(bool parallel, size_t& checked, double& max_abs) {
    std::fill(old_c->c, old_c->c + size_t(m) * padded_n, std::numeric_limits<float>::quiet_NaN());
    std::fill(new_c->c, new_c->c + size_t(m) * padded_n, std::numeric_limits<float>::quiet_NaN());
    run(false, parallel);
    run(true, parallel);
    for (int mi = 0; mi < m; ++mi) {
      for (int ni = 0; ni < n; ++ni) {
        float old = *old_c->get_submat(m, n, mi, ni);
        float now = *new_c->get_submat(m, n, mi, ni);
        ++checked;
        // Bit tests stay meaningful under the production -ffast-math flags.
        const uint32_t old_bits = std::bit_cast<uint32_t>(old);
        const uint32_t new_bits = std::bit_cast<uint32_t>(now);
        if ((old_bits & 0x7f800000u) == 0x7f800000u || (new_bits & 0x7f800000u) == 0x7f800000u) return false;
        max_abs = std::max(max_abs, double(std::abs(old - now)));
        if (old_bits != new_bits) {
          std::cerr << "MISMATCH m=" << m << " n=" << n << " k=" << k << " mi=" << mi << " ni=" << ni << " old=" << old
                    << " now=" << now << std::endl;
          return false;
        }
      }
    }
    return true;
  }
};

int main(int argc, char** argv) {
  size_t checked = 0, cases = 0;
  double max_abs = 0;
  for (auto [n, k] : std::vector<std::pair<int, int>>{{1152, 5120},
                                                      {5120, 1152},
                                                      {2304, 5120},
                                                      {1, 32},
                                                      {3, 64},
                                                      {5, 96},
                                                      {255, 64},
                                                      {257, 160},
                                                      {259, 32},
                                                      {288, 224}}) {
    for (int m = 1; m <= 8; ++m) {
      for (int format = 0; format < 3; ++format) {
        Case c(m, n, k, format);
        for (bool parallel : {false, true}) {
          if (!c.compare(parallel, checked, max_abs)) return 1;
          ++cases;
        }
      }
    }
  }
  // Preserve and exercise the existing single-token natural-order fast path.
  for (int format = 0; format < 3; ++format) {
    Case c(1, 1152, 5120, format, true);
    if (!c.compare(true, checked, max_abs)) return 2;
    ++cases;
  }
  std::cout << "VALIDATION {\"passed\":true,\"bitwise_equal\":true,\"cases\":" << cases
            << ",\"float_elements_checked\":" << checked << ",\"max_absolute_error\":" << max_abs
            << ",\"m_range\":[1,8],\"actual_vec_dispatch_tested\":true,\"empty_worker_tested\":true}" << std::endl;
  if (argc < 2 || std::string(argv[1]) != "--benchmark") return 0;
  // Warm-cache, one-thread microbenchmark. This is not an LLM throughput claim.
  for (auto [n, k] : std::vector<std::pair<int, int>>{{1152, 5120}, {5120, 1152}}) {
    for (int m = 1; m <= 8; ++m) {
      Case c(m, n, k, 2);
      for (int i = 0; i < 2; ++i) {
        c.run(false, false);
        c.run(true, false);
      }
      std::vector<double> old, now;
      for (int trial = 0; trial < 5; ++trial) {
        for (bool opt : {false, true}) {
          auto start = std::chrono::steady_clock::now();
          for (int iter = 0; iter < 5; ++iter) c.run(opt, false);
          double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 5;
          (opt ? now : old).push_back(us);
        }
      }
      std::sort(old.begin(), old.end());
      std::sort(now.begin(), now.end());
      std::cout << "MICRO {\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k << ",\"baseline_us\":" << old[2]
                << ",\"optimized_us\":" << now[2] << ",\"speedup\":" << old[2] / now[2]
                << ",\"warm_cache\":true,\"single_thread\":true}" << std::endl;
    }
  }
}
