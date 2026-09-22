// CPU-only check of KT_WORKER_SPIN_MS parsing (worker_pool.cpp).
//
// The default is the point of the test: an unset, empty or invalid value must
// come back as 50, which is the value the loop was written with, so a build
// carrying this change behaves byte for byte like the old one unless somebody
// sets the variable on purpose.
//
// Build and run (no pool is started, so no NUMA/hwloc hardware is needed):
//   g++ -std=c++20 -I. -Icpu_backend test/cpp/test_worker_spin_ms.cpp \
//       cpu_backend/worker_pool.cpp -lnuma -lhwloc -o /tmp/spin_ms && /tmp/spin_ms

#include <cstdio>
#include <cstdlib>

#include "worker_pool.h"

static int failures = 0;

static void check(const char* value, long expected) {
  long got = kt_parse_spin_ms(value);
  const char* shown = value == nullptr ? "<unset>" : value;
  if (got != expected) {
    printf("FAIL  KT_WORKER_SPIN_MS=%-10s expected %ld, got %ld\n", shown, expected, got);
    failures++;
  } else {
    printf("ok    KT_WORKER_SPIN_MS=%-10s -> %ld ms\n", shown, got);
  }
}

int main() {
  check(nullptr, 50);  // unset: the historical behaviour
  check("", 50);       // set but empty: same
  check("50", 50);
  check("1", 1);   // the D2 arm
  check("0", 0);   // sleep immediately
  check("1000", 1000);
  // Rejected values fall back to the default rather than to 0: a typo must not
  // silently turn the spin off, because that changes performance and nothing
  // else would say so.
  check("abc", 50);
  check("-1", 50);
  check("5ms", 50);
  check(" 5", 5);  // strtol skips leading blanks; only trailing junk is rejected
  printf("%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
  return failures != 0;
}
