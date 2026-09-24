// Evidence that a failed BufferB allocation in AMX_MOE_BASE::init() now says
// what happened instead of making the process disappear.
//
// Why this check exists: kt's worker pool hard-binds every worker thread to one
// NUMA node with hwloc_set_membind(BIND | STRICT | THREAD)
// (cpu_backend/worker_pool.h:49) and never resets the bind. STRICT forbids
// spilling, so a full node makes std::aligned_alloc return nullptr. Before this
// change that nullptr went straight into make_buffer_b(); the BufferB
// constructor's only guard is assert(ptr % 64 == 0)
// (operators/amx/la/amx_raw_buffers.hpp), which 0 passes -- and which NDEBUG
// deletes from a Release build in any case. The crash therefore landed on the
// first write, and with kernel.print-fatal-signals=0 plus a core_pattern piped
// to apport it left no dmesg line and no core: the process just vanished
// (observed 2026-09-24 10:41:21, EngineCore gone with no output).
//
// What is checked here:
//   1. the REAL call site in operators/amx/moe_base.hpp is exercised -- this
//      file instantiates AMX_MOE_BASE with a probe Derived whose
//      buffer_b_required_size_impl() returns a size that cannot be satisfied,
//      so the allocation that fails is the one that ships. Failure is injected
//      through the size, not through the allocator: a hard bind to a
//      deliberately filled node cannot be arranged on this machine (it would
//      need the node's memory taken from the other tenants), and an
//      LD_PRELOAD'd aligned_alloc would prove the wrapper works but not that
//      the shipped call site uses it;
//   2. the thrown message carries the requested byte count AND the calling
//      thread's NUMA binding. Without the second half, "allocation failed" on a
//      machine with free memory elsewhere is still guesswork;
//   3. the mutation arm (the negative control): the same nullptr, put through
//      the code as it was before this change, kills a forked child with SIGSEGV
//      and no output at all. A check that could not tell those two apart would
//      be no evidence;
//   4. the normal path is untouched: a satisfiable size still allocates, still
//      returns 64-byte-aligned memory, and still constructs.
//
// Build and run (same flags as the Release extension, incl. -DNDEBUG, because
// "does it survive NDEBUG" is half the point):
//   KT=$PWD  # kt-kernel/
//   g++ -std=gnu++20 -O2 -DNDEBUG -DUSE_AMX_AVX_KERNEL=1 -D__x86_64__ \
//       -I$KT -I$KT/../third_party -I$KT/../third_party/llama.cpp/. \
//       -mf16c -mfma -mavx -msse3 -mavx2 -mavx512f -mavx512bw -mavx512dq \
//       -mavx512vl -mavx512vbmi -mavx512vnni -mavx512vpopcntdq -mavx512bf16 -fopenmp \
//       $KT/test/cpp/test_checked_aligned_alloc.cpp $KT/cpu_backend/shared_mem_buffer.cpp \
//       -lnuma -lhwloc -o /tmp/test_checked_aligned_alloc && /tmp/test_checked_aligned_alloc

#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "operators/amx/moe_base.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) g_failures++;
}

// A size no allocator can satisfy, and a multiple of 64 so aligned_alloc's
// contract holds. glibc rejects it without touching a single page, so this
// injection allocates nothing -- it must not perturb anything else on the box.
constexpr size_t kImpossible = (size_t)1 << 62;

// ---------------------------------------------------------------------------
// The smallest thing AMX_MOE_BASE can be instantiated on. The kernel type is
// only used by init() for M_STEP and the three buffer typedefs, so a stub is
// enough; the point is to reach the real allocation in moe_base.hpp, not to
// compute anything.
struct ProbeBuffer {
  void* data;
  ProbeBuffer(size_t, size_t, void* p) : data(p) {}
};

struct ProbeKernel {
  using BufferA = ProbeBuffer;
  using BufferB = ProbeBuffer;
  using BufferC = ProbeBuffer;
  static constexpr int M_STEP = 16;
  static constexpr double ELEMENT_SIZE = 2;
};

class ProbeMOE : public AMX_MOE_BASE<ProbeKernel, ProbeMOE> {
 public:
  using Base = AMX_MOE_BASE<ProbeKernel, ProbeMOE>;

  // What the injected BufferB size is. Everything else stays tiny.
  static size_t bb_size;

  explicit ProbeMOE(GeneralMOEConfig config) : Base(config, 0) {}

  size_t buffer_a_required_size_impl(size_t, size_t) const { return 64; }
  size_t buffer_b_required_size_impl(size_t, size_t) const { return bb_size; }
  size_t buffer_c_required_size_impl(size_t, size_t) const { return 64; }

  std::shared_ptr<ProbeBuffer> make_buffer_a_impl(size_t m, size_t k, void* d) const {
    return std::make_shared<ProbeBuffer>(m, k, d);
  }
  std::shared_ptr<ProbeBuffer> make_buffer_b_impl(size_t n, size_t k, void* d) const {
    return std::make_shared<ProbeBuffer>(n, k, d);
  }
  std::shared_ptr<ProbeBuffer> make_buffer_c_impl(size_t m, size_t n, void* d) const {
    return std::make_shared<ProbeBuffer>(m, n, d);
  }

  void derived_init() {}
};

size_t ProbeMOE::bb_size = 4096;

GeneralMOEConfig probe_config() {
  GeneralMOEConfig config;
  config.expert_num = 2;
  config.num_experts_per_tok = 1;
  config.hidden_size = 64;
  config.intermediate_size = 64;
  config.max_len = 1;
  config.layer_idx = 0;
  config.pool = nullptr;  // init() never touches the pool
  return config;
}

// ---------------------------------------------------------------------------
// 1 + 2: the real call site throws, and the message is usable.
void test_real_call_site_throws() {
  printf("[1] AMX_MOE_BASE::init() with an unsatisfiable BufferB size\n");
  ProbeMOE::bb_size = kImpossible;

  std::string message;
  bool threw = false;
  try {
    ProbeMOE moe(probe_config());
    (void)moe;
  } catch (const std::runtime_error& e) {
    threw = true;
    message = e.what();
  }
  ProbeMOE::bb_size = 4096;

  check(threw, "init() throws std::runtime_error instead of storing nullptr");
  if (!threw) return;

  printf("  --- thrown message ---\n  %s\n  ----------------------\n", message.c_str());
  check(message.find(std::to_string(kImpossible)) != std::string::npos,
        "message names the requested size in bytes");
  check(message.find("BufferB") != std::string::npos, "message names which buffer failed");
  check(message.find("expert 0") != std::string::npos, "message names the expert index");
  check(message.find("mempolicy") != std::string::npos && message.find("node") != std::string::npos,
        "message names the calling thread's NUMA binding");
  check(message.find("worker_pool.h") != std::string::npos,
        "message points at the hard bind as the reason a full node cannot spill");
}

// ---------------------------------------------------------------------------
// 3: the negative control. Exactly what the three call sites did before this
// change, in a child process, so the difference between "checked" and
// "unchecked" is observable rather than asserted.
void test_unchecked_control_dies_silently() {
  printf("[2] negative control: the pre-change code path, in a forked child\n");

#ifdef NDEBUG
  printf("  (built with NDEBUG, i.e. exactly the Release configuration)\n");
#else
  printf("  (WARNING: built without NDEBUG -- Release deletes the assert entirely)\n");
#endif

  // The old guard, evaluated here: nullptr satisfies it even when asserts are
  // live, so it never stood between a failed allocation and the first write.
  void* null_ptr = nullptr;
  check((reinterpret_cast<intptr_t>(null_ptr) % 64) == 0,
        "assert(ptr % 64 == 0) -- the only old guard -- accepts nullptr");

  int fds[2];
  if (pipe(fds) != 0) {
    check(false, "pipe()");
    return;
  }

  pid_t pid = fork();
  if (pid == 0) {
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[0]);
    close(fds[1]);
    // --- pre-change code, verbatim in shape ---
    void* bb_ptr = std::aligned_alloc(64, kImpossible);
    assert(reinterpret_cast<intptr_t>(bb_ptr) % 64 == 0);  // passes for nullptr; gone under NDEBUG
    *reinterpret_cast<volatile char*>(bb_ptr) = 1;         // SIGSEGV here
    // -----------------------------------------
    printf("child reached code after the write -- unexpected\n");
    fflush(stdout);
    _exit(0);
  }

  close(fds[1]);
  char buf[4096];
  ssize_t n = read(fds[0], buf, sizeof(buf) - 1);
  if (n < 0) n = 0;
  buf[n] = '\0';
  close(fds[0]);

  int status = 0;
  waitpid(pid, &status, 0);

  const bool signalled = WIFSIGNALED(status);
  const int sig = signalled ? WTERMSIG(status) : 0;
  printf("  child: %s%d, output on stdout+stderr: %zd bytes %s\n", signalled ? "killed by signal " : "exited ",
         signalled ? sig : WEXITSTATUS(status), n, n ? buf : "(nothing)");

  check(signalled && sig == SIGSEGV, "unchecked path dies of SIGSEGV");
  check(n == 0, "unchecked path prints nothing at all before dying");
}

// ---------------------------------------------------------------------------
// 4: the normal path still works.
void test_normal_path_unaffected() {
  printf("[3] normal path: a satisfiable size still allocates\n");
  ProbeMOE::bb_size = 4096;

  bool ok = true;
  try {
    ProbeMOE moe(probe_config());
    for (size_t i = 0; i < 2; i++) {
      ok = ok && moe.gate_bb_[i] && moe.up_bb_[i] && moe.down_bb_[i];
      ok = ok && moe.gate_bb_[i]->data != nullptr;
      ok = ok && (reinterpret_cast<uintptr_t>(moe.gate_bb_[i]->data) % 64) == 0;
      ok = ok && (reinterpret_cast<uintptr_t>(moe.up_bb_[i]->data) % 64) == 0;
      ok = ok && (reinterpret_cast<uintptr_t>(moe.down_bb_[i]->data) % 64) == 0;
    }
  } catch (const std::exception& e) {
    printf("  unexpected exception: %s\n", e.what());
    ok = false;
  }
  check(ok, "init() allocates three 64-byte-aligned BufferB per expert and does not throw");
}

// ---------------------------------------------------------------------------
// 5: the same failure on a thread that IS hard-bound, which is the case the
// message exists for. The bind is real (set_mempolicy(MPOL_BIND) on this
// thread, the same policy hwloc_set_membind installs on every kt worker); only
// the size is injected, so no node is filled and nothing else on the machine
// is touched.
void test_hard_bound_thread_is_named() {
  printf("[5] same failure on a thread hard-bound with set_mempolicy(MPOL_BIND)\n");

  const int cpu = sched_getcpu();
  const int node = cpu >= 0 ? numa_node_of_cpu(cpu) : -1;
  if (node < 0) {
    printf("  skipped: cannot determine this thread's node\n");
    return;
  }

  unsigned long mask[16] = {0};
  mask[node / 64] |= 1ULL << (node % 64);
  if (set_mempolicy(MPOL_BIND, mask, sizeof(mask) * 8) != 0) {
    perror("  set_mempolicy");
    check(false, "set_mempolicy(MPOL_BIND) on this thread");
    return;
  }

  ProbeMOE::bb_size = kImpossible;
  std::string message;
  try {
    ProbeMOE moe(probe_config());
    (void)moe;
  } catch (const std::runtime_error& e) {
    message = e.what();
  }
  ProbeMOE::bb_size = 4096;
  set_mempolicy(MPOL_DEFAULT, nullptr, 0);  // undo the bind for the rest of the run

  printf("  --- thrown message ---\n  %s\n  ----------------------\n", message.c_str());
  check(message.find("MPOL_BIND (hard-bound)") != std::string::npos, "message reports the policy as a hard bind");
  check(message.find("nodes {" + std::to_string(node) + ":free ") != std::string::npos,
        "message names the bound node and its free memory");
  check(message.find("could not fall back to another node") != std::string::npos,
        "message explains why free memory elsewhere does not help");
}

// ---------------------------------------------------------------------------
void test_binding_string_is_informative() {
  printf("[4] kt_thread_numa_binding() on this thread\n");
  const std::string s = kt_thread_numa_binding();
  printf("  %s\n", s.c_str());
  check(!s.empty() && s.find("cpu ") == 0, "binding string reports the running cpu");
}

}  // namespace

int main() {
  test_real_call_site_throws();
  test_unchecked_control_dies_silently();
  test_normal_path_unaffected();
  test_hard_bound_thread_is_named();
  test_binding_string_is_informative();

  printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", g_failures,
         g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
