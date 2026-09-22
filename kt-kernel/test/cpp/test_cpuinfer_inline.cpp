// CPU-only check of the inline execution path (task_queue.cpp, cpuinfer.h).
//
// DSV4.1 offload §4.22 P2: one MoE layer on the NPU graph path is one
// submit+sync from a host-func callback, and the submit hands the task to the
// TaskQueue worker thread -- a full thread round trip measured at 10.44 us
// median against 0.04 us for running it inline. run_inline() removes the trip
// by running the task on the calling thread.
//
// What has to hold for that to be allowed to ship, and is checked here:
//   1. the switch parses like every other kt switch: unset/empty/"0" = off,
//      so a build carrying this change behaves exactly as before;
//   2. the two paths produce bit-identical output for the same task -- the
//      task here has the shape kt's forward has (fan out to per-NUMA threads,
//      wait, merge), because that is where an inline run could go wrong;
//   3. the mutation arm: the same inline run with the wait for ONE pool
//      removed must be caught by that comparison. A check that cannot see a
//      missing wait would not be evidence that inlining is safe;
//   4. an exception from an inline task reaches the caller at the next
//      sync(), exactly where a queued task's exception reaches it -- kt's
//      contract is "you learn at sync", and losing it means a failed layer
//      turns into silently stale output;
//   5. run_inline never jumps a queued task: pending_tasks() is what the
//      caller (CPUInfer::run_inline) reads to decide, and it must be exact.
//
// Build and run (no NUMA/hwloc, no ggml: only the task queue is needed):
//   g++ -std=c++20 -O2 -I. -Icpu_backend test/cpp/test_cpuinfer_inline.cpp \
//       cpu_backend/task_queue.cpp -o /tmp/inline_test && /tmp/inline_test

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "task_queue.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) failures++;
}

// ---------------------------------------------------------------- the task
//
// The shape of TP_MOE::forward: fan the work out to one thread per "NUMA
// node", wait for each, then merge into the output. Deterministic, so the two
// paths must agree bit for bit.

static constexpr int kNuma = 2;
static constexpr int kLen = 4096;

struct FakeMoE {
  std::vector<float> partial[kNuma];
  std::vector<float> out;

  FakeMoE() : out(kLen, 0.0f) {
    for (int n = 0; n < kNuma; ++n) partial[n].assign(kLen, 0.0f);
  }

  // skip_wait_for: pretend the caller forgot to wait for that pool (mutation)
  void forward(int token, int skip_wait_for = -1) {
    std::vector<std::thread> pools;
    for (int n = 0; n < kNuma; ++n) {
      pools.emplace_back([this, n, token] {
        // slow enough that a missing wait really reads unwritten memory
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        for (int i = 0; i < kLen; ++i) {
          partial[n][i] = static_cast<float>((i % 97) * (n + 1) + token);
        }
      });
    }
    for (int n = 0; n < kNuma; ++n) {
      if (n == skip_wait_for) {
        pools[n].detach();  // the mutation: merge without waiting for this one
        continue;
      }
      pools[n].join();
    }
    for (int i = 0; i < kLen; ++i) {  // merge_results
      out[i] = partial[0][i] + partial[1][i];
    }
    if (skip_wait_for >= 0) {
      // the detached thread must not outlive the buffers it writes
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
};

static bool identical(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main() {
  // 1 -- the switch
  check(kt_parse_inline_flag(nullptr) == false, "KT_CPUINFER_INLINE unset -> off (the old path)");
  check(kt_parse_inline_flag("") == false, "KT_CPUINFER_INLINE empty -> off");
  check(kt_parse_inline_flag("0") == false, "KT_CPUINFER_INLINE=0 -> off");
  check(kt_parse_inline_flag("1") == true, "KT_CPUINFER_INLINE=1 -> on");
  check(kt_parse_inline_flag("yes") == false, "a typo falls back to off, not silently on");

  // 2 -- the two paths agree, bit for bit
  TaskQueue q;
  FakeMoE queued, inlined;
  std::thread::id ran_on_queue, ran_on_inline;
  const std::thread::id caller = std::this_thread::get_id();
  for (int token = 0; token < 8; ++token) {
    q.enqueue([&] {
      ran_on_queue = std::this_thread::get_id();
      queued.forward(token);
    });
    q.sync(0);
    q.run_inline([&] {
      ran_on_inline = std::this_thread::get_id();
      inlined.forward(token);
    });
    q.sync(0);
    if (!identical(queued.out, inlined.out)) {
      check(false, "inline and queued output differ");
      break;
    }
  }
  check(identical(queued.out, inlined.out), "inline output is bit-identical to the queued output");
  check(ran_on_queue != caller, "the queued task really ran on the worker thread");
  check(ran_on_inline == caller, "the inline task really ran on the calling thread");

  // 3 -- the mutation arm: inline, but one pool is not waited for
  FakeMoE mutated;
  bool caught = false;
  for (int token = 0; token < 8 && !caught; ++token) {
    FakeMoE reference;
    reference.forward(token);
    mutated = FakeMoE();
    q.run_inline([&] { mutated.forward(token, /*skip_wait_for=*/1); });
    q.sync(0);
    caught = !identical(reference.out, mutated.out);
  }
  check(caught, "MUTATION: skipping the wait for one pool is caught by the same comparison");

  // 4 -- an inline task's exception surfaces at the next sync, not earlier
  bool threw_inside_run_inline = false;
  try {
    q.run_inline([] { throw std::runtime_error("boom"); });
  } catch (...) {
    threw_inside_run_inline = true;
  }
  check(!threw_inside_run_inline, "run_inline does not raise where kt never raised (the task is not the call)");
  bool threw_at_sync = false;
  try {
    q.sync(0);
  } catch (const std::runtime_error& e) {
    threw_at_sync = std::strcmp(e.what(), "boom") == 0;
  }
  check(threw_at_sync, "the inline task's exception is rethrown by the next sync(), like a queued one");
  bool clean = true;
  try {
    q.sync(0);
  } catch (...) {
    clean = false;
  }
  check(clean, "and it is consumed once, not left to poison every later sync");

  // 5 -- pending_tasks is what decides; it must be exact
  check(q.pending_tasks() == 0, "an idle queue reports 0 pending");
  std::atomic<bool> release{false};
  q.enqueue([&] {
    while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
  });
  // the worker may not have picked it up yet, but it is pending either way
  check(q.pending_tasks() == 1, "a queued, unfinished task is counted -- this is what refuses an inline run");
  release.store(true, std::memory_order_release);
  q.sync(0);
  check(q.pending_tasks() == 0, "and it is back to 0 once drained");

  printf(failures ? "\nFAILED (%d)\n" : "\nall checks passed\n", failures);
  return failures == 0 ? 0 : 1;
}
