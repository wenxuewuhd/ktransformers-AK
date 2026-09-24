// KT_ZEROCOPY_MLOCK parses like every other switch in zerocopy_weights.hpp --
// except that unset still means ON.
//
// Why this check exists: mlock_enabled() used to return true for everything
// that was not the exact string "0". So KT_ZEROCOPY_MLOCK=false / =off / =no,
// or one typo, silently turned pinning ON -- the opposite of what was asked,
// and the expensive direction: pinning means 84.4 GiB mlocked onto node0, the
// node with a few GiB free, when decode majflt = 0 says the pin is not needed.
// vllm_ascend/kt_offload/layer.py exports the exact string "0", so no shipped
// run was affected; what is removed here is the shape.
//
// The default is the part that must NOT change. Every other switch in that file
// defaults off; this one defaults on, and a later "let's make it consistent"
// would flip the meaning of every measured arm. So the negative control here is
// not only "0 still means off" -- it is also "unset still means on".
//
// Build and run, from kt-kernel/ (header-only, nothing else needed):
//   g++ -std=gnu++20 -O2 -DNDEBUG -I. test/cpp/test_zerocopy_mlock_flag.cpp -o /tmp/zc_mlock
//   /tmp/zc_mlock

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "operators/zerocopy_weights.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) g_failures++;
}

void set_flag(const char* value) {
  if (value == nullptr) {
    unsetenv("KT_ZEROCOPY_MLOCK");
  } else {
    setenv("KT_ZEROCOPY_MLOCK", value, 1);
  }
}

// Returns "on" / "off" / "throw: <message>".
std::string evaluate(const char* value) {
  set_flag(value);
  try {
    return kt_zerocopy::mlock_enabled() ? "on" : "off";
  } catch (const std::runtime_error& e) {
    return std::string("throw: ") + e.what();
  }
}

void expect(const char* value, const std::string& want, const char* what) {
  const std::string got = evaluate(value);
  if (got != want) printf("  wanted \"%s\", got \"%s\"\n", want.c_str(), got.c_str());
  check(got == want, what);
}

}  // namespace

int main() {
  printf("[1] the values that must keep working (negative control: the default is untouched)\n");
  expect(nullptr, "on", "unset -> ON (this switch's default, unlike the others in the file)");
  expect("", "on", "empty -> ON (an exported-but-empty variable is not 'off')");
  expect("0", "off", "\"0\" -> OFF -- what layer.py exports today");
  expect("1", "on", "\"1\" -> ON");

  printf("[2] the values that used to silently mean ON and now throw\n");
  for (const char* bad : {"false", "off", "no", "00", "1 ", "TRUE", "O"}) {
    const std::string got = evaluate(bad);
    const bool threw = got.rfind("throw: ", 0) == 0;
    const bool names_var = got.find("KT_ZEROCOPY_MLOCK") != std::string::npos;
    const bool quotes_value = got.find(std::string("got \"") + bad + "\"") != std::string::npos;
    printf("  KT_ZEROCOPY_MLOCK=\"%s\" -> %s\n", bad, got.c_str());
    check(threw && names_var && quotes_value, "throws, names the variable, and quotes the bad value");
  }

  printf("[3] before this change, every one of those returned ON\n");
  // The old body, verbatim -- the mutation arm. If it disagreed with the new
  // one only on "0", the change would be cosmetic.
  auto old_mlock_enabled = []() {
    const char* v = std::getenv("KT_ZEROCOPY_MLOCK");
    if (v != nullptr && std::string(v) == "0") return false;
    return true;
  };
  int diverged = 0;
  for (const char* bad : {"false", "off", "no", "00", "1 ", "TRUE", "O"}) {
    set_flag(bad);
    const bool old_says_on = old_mlock_enabled();
    bool new_threw = false;
    try {
      (void)kt_zerocopy::mlock_enabled();
    } catch (const std::runtime_error&) {
      new_threw = true;
    }
    if (old_says_on && new_threw) diverged++;
  }
  printf("  %d/7 values that the old parse read as ON are now refused\n", diverged);
  check(diverged == 7, "the two parses disagree on exactly the dangerous values");

  // And they agree everywhere it matters.
  for (const char* good : {"0", "1"}) {
    set_flag(good);
    check(old_mlock_enabled() == kt_zerocopy::mlock_enabled(),
          std::string(std::string("old and new agree on \"") + good + "\"").c_str());
  }
  set_flag(nullptr);
  check(old_mlock_enabled() == kt_zerocopy::mlock_enabled() && kt_zerocopy::mlock_enabled(),
        "old and new agree on unset, and both say ON");

  printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", g_failures,
         g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 1;
}
