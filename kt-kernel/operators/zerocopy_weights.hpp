/**
 * Zero-copy CPU-MoE weights: switch, residency and the audit that proves it.
 *
 * WHY
 *   Streaming prefill and the CPU MoE consume the *same* expert bytes. Today
 *   each keeps its own copy: the checkpoint pages sit in the page cache while
 *   kt memcpy's 268.9 GiB of them into anonymous BufferB memory. Pointing
 *   BufferB straight at the mapping removes the duplicate and lets both
 *   consumers share one locked page-cache copy.
 *
 * WHAT THIS FILE DOES NOT DO
 *   It does not create the mapping. It can only pin what it is handed, and the
 *   kind of mapping decides whether pinning helps at all -- see
 *   check_mapping_is_shared_readonly() below, which is the load-bearing guard.
 *
 * Origin: dsv41 stream-prefill, CPU-MoE weight zero-copy.
 */
#ifndef KT_ZEROCOPY_WEIGHTS_H
#define KT_ZEROCOPY_WEIGHTS_H

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kt_zerocopy {

#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#define MPOL_PREFERRED 1
#define MPOL_BIND 2
#define MPOL_INTERLEAVE 3
#endif

// Strict three-state parse, same convention as KT_FULLSET_LOAD in
// python/utils/amx.py: unset/"0" off, "1" on, anything else is an error rather
// than a silent off. A switch that fails quietly is worse than no switch: the
// run looks like it took the new path and the readings say otherwise.
inline bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0' || std::string(v) == "0") return false;
  if (std::string(v) == "1") return true;
  throw std::runtime_error(std::string(name) + " must be \"\", \"0\" or \"1\", got \"" + v + "\"");
}

// KT_ZEROCOPY_WEIGHTS=1 makes the AMX MXFP4 BufferB point at the checkpoint
// mapping instead of owning a memcpy'd copy. Default OFF.
inline bool enabled() {
  static const bool v = env_flag("KT_ZEROCOPY_WEIGHTS");
  return v;
}

// KT_ZEROCOPY_SCOPE picks WHICH of the three expert tensors are served from the
// mapping. Default "all" -- unset behaves exactly as before this switch existed.
//
// 🔴 WHY A PARTIAL SCOPE EXISTS AT ALL. The two tensor families do not behave
// the same under a per-page placement, and it is a property of their layout,
// not of the code:
//   w13 (gate/up)  source [full_inter, hidden/2] row-major, TP splits `inter`
//                  => this part owns a CONTIGUOUS ROW BLOCK. Its bytes are one
//                  unbroken extent, every 4 KiB page belongs to exactly one
//                  part, so "this part's pages -> this part's node" reaches
//                  100% local. Zero-copy costs nothing here.
//   w2  (down)     source [hidden, full_inter/2] row-major, TP still splits
//                  `inter` => this part owns a COLUMN SLICE OF EVERY ROW. With
//                  full_inter/2 = 1024 B the row is a quarter of a page, so a
//                  single page carries rows from BOTH parts and both sockets
//                  must read it. No per-page placement can do better than
//                  1/tp_count local for w2, ever.
// So "w13" means: gate/up zero-copy (+ placed), down goes back to the ordinary
// memcpy into this part's own node-local BufferB. It keeps the 2/3 of the
// weight bytes that zero-copy can hold without a locality penalty and pays the
// remaining 1/3 in memory to keep w2 node-local.
enum class Scope { kAll, kW13 };

inline Scope scope() {
  static const Scope s = [] {
    const char* v = std::getenv("KT_ZEROCOPY_SCOPE");
    const std::string p = v == nullptr ? "" : v;
    if (p.empty() || p == "all") return Scope::kAll;
    if (p == "w13") return Scope::kW13;
    // Same rule as env_flag(): an unrecognised value is an error, never a
    // silent fallback to the default. A run that silently took "all" while the
    // operator asked for "w13" produces a reading nobody can tell apart.
    throw std::runtime_error(std::string("KT_ZEROCOPY_SCOPE must be \"\", \"all\" or \"w13\", got \"") + p + "\"");
  }();
  return s;
}

// True when w2/down must NOT be zero-copied (it keeps the copy path).
inline bool scope_w13_only() { return scope() == Scope::kW13; }

// KT_ZEROCOPY_MLOCK=0 turns off the pinning (placement + mlock) while keeping
// the zero-copy pointers, so the two effects can be measured apart.
inline bool mlock_enabled() {
  const char* v = std::getenv("KT_ZEROCOPY_MLOCK");
  if (v != nullptr && std::string(v) == "0") return false;
  return true;
}

// KT_ZEROCOPY_AUDIT=1 prints per-node page counts and residency at the end of
// the load. Costs one move_pages()/mincore() sweep over the pinned bytes.
inline bool audit_enabled() {
  static const bool v = env_flag("KT_ZEROCOPY_AUDIT");
  return v;
}

// The measured hazard, not a hypothetical one: mlock() on a PRIVATE WRITABLE
// file mapping (`rw-p`, which is what safetensors' own safe_open() produces)
// breaks COW and materialises an anonymous copy of every page. Measured on
// this machine over 1 GiB of DeepSeek-V4.1-Flash:
//     rw-p mapping : dRssAnon +1024 MiB, dRssFile -0.1 MiB, dCached +1022 MiB
//     r--s mapping : dRssAnon     0 MiB, dRssFile +1024 MiB, dCached +1024 MiB
// So pinning an `rw-p` mapping gives you TWO copies instead of zero. Refuse.
inline bool allow_private_writable() {
  const char* v = std::getenv("KT_ZEROCOPY_ALLOW_PRIVATE_WRITABLE");
  return v != nullptr && std::string(v) == "1";
}

struct Stats {
  std::atomic<uint64_t> ranges{0};
  std::atomic<uint64_t> bytes_requested{0};
  std::atomic<uint64_t> bytes_locked{0};
  std::atomic<uint64_t> mlock_failures{0};
  std::atomic<uint64_t> last_mlock_errno{0};
  // Placement (move_pages migration). `policy_pages` is every page handed to
  // move_pages; `policy_failures` counts pages whose RETURNED per-page status
  // is not the node we asked for -- see migrate_range() for why the syscall
  // return value alone is worthless here.
  std::atomic<uint64_t> policy_calls{0};
  std::atomic<uint64_t> policy_pages{0};
  std::atomic<uint64_t> policy_moved{0};
  std::atomic<uint64_t> policy_retried{0};
  std::atomic<uint64_t> policy_failures{0};
  std::atomic<uint64_t> policy_ns{0};
  std::atomic<int64_t> last_page_status{0};
  std::atomic<uint64_t> last_policy_errno{0};
};

inline Stats& stats() {
  static Stats s;
  return s;
}

// Ranges kept for the audit sweep (address, bytes). Only filled when the audit
// is on, so the normal path allocates nothing here.
inline std::vector<std::pair<uintptr_t, size_t>>& audit_ranges() {
  static std::vector<std::pair<uintptr_t, size_t>> v;
  return v;
}
inline std::mutex& audit_mutex() {
  static std::mutex m;
  return m;
}

inline long page_size() {
  static const long ps = sysconf(_SC_PAGESIZE);
  return ps;
}

// One line of /proc/self/maps for `addr`, or "" if none.
inline std::string maps_line_for(uintptr_t addr) {
  FILE* f = fopen("/proc/self/maps", "r");
  if (f == nullptr) return "";
  char line[1024];
  std::string hit;
  while (fgets(line, sizeof(line), f) != nullptr) {
    unsigned long lo = 0, hi = 0;
    if (sscanf(line, "%lx-%lx", &lo, &hi) != 2) continue;
    if (addr >= lo && addr < hi) {
      hit = line;
      break;
    }
  }
  fclose(f);
  return hit;
}

// perms field of the maps line, e.g. "r--s" / "rw-p".
inline std::string maps_perms_for(uintptr_t addr) {
  std::string line = maps_line_for(addr);
  if (line.empty()) return "";
  size_t sp = line.find(' ');
  if (sp == std::string::npos || line.size() < sp + 5) return "";
  return line.substr(sp + 1, 4);
}

// Checked once per distinct mapping. Throws on `rw-p` (see above) unless the
// escape hatch is set; also refuses a mapping that is not file backed.
inline void check_mapping_is_shared_readonly(const void* addr) {
  static std::mutex m;
  static std::map<uintptr_t, bool> seen;
  const uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  std::string line = maps_line_for(a);
  if (line.empty()) throw std::runtime_error("zero-copy weights: pointer is in no mapping of /proc/self/maps");
  unsigned long lo = 0, hi = 0;
  sscanf(line.c_str(), "%lx-%lx", &lo, &hi);
  {
    std::lock_guard<std::mutex> g(m);
    if (seen.count(lo)) return;
    seen[lo] = true;
  }
  const std::string perms = maps_perms_for(a);
  const bool writable = perms.size() > 1 && perms[1] == 'w';
  const bool shared = perms.size() > 3 && perms[3] == 's';
  printf("[kt zero-copy] source mapping %s", line.c_str());
  if (writable && !shared) {
    if (!allow_private_writable()) {
      throw std::runtime_error(
          "zero-copy weights: the source mapping is PRIVATE+WRITABLE (" + perms +
          "). mlock() would break COW and allocate an anonymous copy of every page -- measured +1024 MiB RssAnon per "
          "1 GiB on this machine -- so this gives two copies, not zero. Map the checkpoint PROT_READ|MAP_SHARED "
          "(`r--s`) instead. Set KT_ZEROCOPY_ALLOW_PRIVATE_WRITABLE=1 only to reproduce that measurement.");
    }
    printf("[kt zero-copy] 🔴 PRIVATE+WRITABLE mapping accepted by KT_ZEROCOPY_ALLOW_PRIVATE_WRITABLE=1; "
           "mlock will COW into anonymous memory\n");
  }
}

// ---------------------------------------------------------------------------
// Placement: move_pages(2) MIGRATION, not an allocation policy.
//
// 🔴🔴 WHY THE PREVIOUS MECHANISM WAS WRONG, AND HOW IT LIED.
// This used to call set_mempolicy(MPOL_BIND) on each loader thread. An
// allocation policy only decides where the kernel puts a page it is allocating
// *right now*. A checkpoint page that is ALREADY in the page cache is reused in
// place: no allocation happens, so no policy is consulted, and the page stays
// wherever it landed the first time anybody read the file. Measured 2026-09-24:
//   model-00004-of-00048.safetensors, 66.5% resident, 91.8% of those on node1.
//   KT_ZEROCOPY_POLICY=bind:0, --layers 1 --tp 2 --nodes 0,1 --n-cpu 256
//     -> audit node0=2.96% / node1=97.04%   with policy_failures=0
// The earlier "bind:0 -> 100.00% node0" positive control passed only because
// the file was COLD at the time, i.e. it measured the one case production never
// sees: the whole point of zero-copy is to share the page cache with streaming
// prefill, so in production those pages are always already cached.
// Two lessons are burned into the code below:
//   (a) the mechanism must MIGRATE existing pages, not steer new allocations;
//   (b) policy_failures must come from the PER-PAGE status[] array. The old
//       code only looked at the syscall return value, which is why it could
//       report policy_failures=0 while placing exactly nothing.
//
// move_pages(2) (x86_64 nr 279) with MPOL_MF_MOVE_ALL migrates page-cache pages
// that are already resident and already mapped here. Measured on this machine:
//   1 GiB sample 21.65%/78.35% -> 100.00% node0
//   4 GiB sample 28.92%/71.08% -> 100.00% node1 in 0.55 s single thread
//                                 (~1.16 GiB actually moved => ~2.1 GiB/s)
//   MPOL_MF_MOVE (without ALL) returned -EBUSY for 15229 pages of that same
//   4 GiB sample; MPOL_MF_MOVE_ALL moved all of them. Hence ALL, always.
// Precondition: the page must be mapped in THIS process's page table, or
// move_pages reports -ENOENT and leaves it alone. Being in the page cache is
// not enough -- hence the touch loop in migrate_range().
//
// KT_ZEROCOPY_POLICY (switch names and meanings unchanged):
//   "segment"    (default) every page of this TP part's slice -> `own_node`,
//                the node of the sub-pool that reads those bytes. This is what
//                recreates the per-socket locality the copy path gets for free
//                by memcpy'ing each part's slice into its own node-local buffer.
//   "interleave" round-robin over `nodes` BY PAGE. With migration this is now
//                genuinely deterministic 50/50 -- it no longer goes through
//                MPOL_INTERLEAVE, which was only a hint and was measured being
//                dropped wholesale (100% node1) once node0 got tight.
//   "bind:<n>"   everything on node <n>. The control: on a hot file it must
//                still reach ~100% on node n, and if it reads the same as the
//                segment arm the measurement is broken, not the placement.
// Two values were ADDED, both only so the arms can be told apart; neither
// changes what the three above mean:
//   "none"        pin, do not place (KT_ZEROCOPY_MLOCK=0 turns off both).
//   "antisegment" segment with the node flipped -- the negative control.
// ---------------------------------------------------------------------------
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif
#ifndef MPOL_MF_MOVE_ALL
#define MPOL_MF_MOVE_ALL (1 << 2)
#endif

// Per-page target nodes, round-robined over the pages of a range. Empty means
// "do not place" (no valid target).
inline std::vector<int> policy_targets(const std::vector<int>& nodes, int own_node) {
  const char* pol = std::getenv("KT_ZEROCOPY_POLICY");
  const std::string p = pol == nullptr ? "" : pol;
  if (p.rfind("bind:", 0) == 0) {
    int n = atoi(p.c_str() + 5);
    if (n < 0 || n >= 1024) throw std::runtime_error("KT_ZEROCOPY_POLICY=bind:<node>: node out of range");
    return std::vector<int>{n};
  }
  // "none": pin but do not place. Added so the compute arms can separate the
  // two effects -- KT_ZEROCOPY_MLOCK=0 turns off BOTH, which would leave any
  // "placement helps" reading confounded with "pinning costs/helps".
  if (p == "none") return {};
  if (p == "interleave") {
    std::vector<int> t;
    for (int n : nodes)
      if (n >= 0) t.push_back(n);
    if (t.empty()) t.push_back(0);
    return t;
  }
  if (p.empty() || p == "segment") {
    if (own_node < 0) return {};
    return std::vector<int>{own_node};
  }
  // "antisegment": segment with the target node flipped. The negative control
  // for the compute arms, and the only one that is airtight: it migrates the
  // SAME ranges, the SAME number of pages, through the SAME syscalls as
  // "segment", and differs only in which node they land on. If segment is fast
  // and antisegment is slow, the win is placement and cannot be a side effect
  // of the migration pass itself (warmed TLB, populated PTEs, ...).
  // "bind:<n>" cannot do that job here: measured 2026-09-24, torch had already
  // put every synthetic weight page on node0, so bind:0 reported
  // needed_move=0 -- a control that moves nothing proves nothing.
  if (p == "antisegment") {
    if (own_node < 0) return {};
    for (int n : nodes)
      if (n >= 0 && n != own_node) return std::vector<int>{n};
    return std::vector<int>{own_node};
  }
  throw std::runtime_error("KT_ZEROCOPY_POLICY must be \"segment\", \"antisegment\", \"interleave\", \"bind:<node>\" or \"none\"");
}

// Migrate [start, start+span) page by page onto `targets` (round-robin).
// Called from the sub-pool's worker threads, one range at a time, so the
// parallelism is whatever the sub-pool has (32 threads in the check runs).
inline void migrate_range(uintptr_t start, size_t span, const std::vector<int>& targets) {
  if (targets.empty() || span == 0) return;
  const size_t ps = static_cast<size_t>(page_size());
  const size_t npages = span / ps;
  if (npages == 0) return;
  // 32768 pages = 128 MiB of payload per syscall; the three arrays cost 320 KiB
  // of thread-local scratch, which stays in L2 and is reused across ranges.
  constexpr size_t kBatch = 32768;
  static thread_local std::vector<void*> pages;
  static thread_local std::vector<int> want;
  static thread_local std::vector<int> status;

  for (size_t base = 0; base < npages; base += kBatch) {
    const size_t n = std::min(kBatch, npages - base);
    pages.resize(n);
    want.resize(n);
    status.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
      void* a = reinterpret_cast<void*>(start + (base + i) * ps);
      // Map it into our page table. A page that is only in the page cache is
      // reported -ENOENT by move_pages and is silently left where it is.
      (void)*reinterpret_cast<volatile const unsigned char*>(a);
      pages[i] = a;
      want[i] = targets[(base + i) % targets.size()];
    }
    // Query first: this is what lets a reading say "the mutation really was
    // applied" -- without it, a 100%-on-target result after migration is
    // indistinguishable from pages that were already there.
    long qrc = syscall(SYS_move_pages, 0, (unsigned long)n, pages.data(), (int*)nullptr, status.data(), 0);
    uint64_t needed = 0;
    if (qrc == 0)
      for (size_t i = 0; i < n; ++i)
        if (status[i] != want[i]) ++needed;

    const auto t0 = std::chrono::steady_clock::now();
    long rc = syscall(SYS_move_pages, 0, (unsigned long)n, pages.data(), want.data(), status.data(),
                      MPOL_MF_MOVE_ALL);
    const auto t1 = std::chrono::steady_clock::now();

    stats().policy_calls.fetch_add(1);
    stats().policy_pages.fetch_add(n);
    stats().policy_moved.fetch_add(needed);
    stats().policy_ns.fetch_add(
        (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    if (rc < 0) {
      // Whole-call failure. Every page of this batch is a failure.
      stats().policy_failures.fetch_add(n);
      stats().last_policy_errno.store((uint64_t)errno);
      continue;
    }
    // 🔴 rc >= 0 says nothing about placement: move_pages returns 0 while
    // reporting -ENOENT/-EBUSY/-EACCES per page in status[]. Count the pages
    // that did NOT end up on the node we asked for -- errors and
    // silently-unmoved alike.
    //
    // One retry for those, because a few per-page -ENOENT are expected and
    // transient: pin_range rounds its range out to page boundaries, so adjacent
    // experts share a first/last page, and while another worker thread is
    // migrating that page its PTE is a migration entry -- follow_page finds
    // nothing and move_pages reports -ENOENT. Measured 344 of 1.1M pages
    // (0.031%) on the bind:0 control run, all -ENOENT. Retried pages are
    // counted separately; only what still misses after the retry is a failure.
    std::vector<size_t> redo;
    for (size_t i = 0; i < n; ++i)
      if (status[i] != want[i]) redo.push_back(i);
    if (!redo.empty()) {
      stats().policy_retried.fetch_add(redo.size());
      std::vector<void*> rp(redo.size());
      std::vector<int> rw(redo.size());
      std::vector<int> rs(redo.size(), 0);
      for (size_t j = 0; j < redo.size(); ++j) {
        (void)*reinterpret_cast<volatile const unsigned char*>(pages[redo[j]]);
        rp[j] = pages[redo[j]];
        rw[j] = want[redo[j]];
      }
      long rrc = syscall(SYS_move_pages, 0, (unsigned long)rp.size(), rp.data(), rw.data(), rs.data(),
                         MPOL_MF_MOVE_ALL);
      uint64_t bad = 0;
      if (rrc < 0) {
        bad = rp.size();
        stats().last_policy_errno.store((uint64_t)errno);
      } else {
        for (size_t j = 0; j < rp.size(); ++j)
          if (rs[j] != rw[j]) {
            ++bad;
            stats().last_page_status.store(rs[j]);
          }
      }
      if (bad) stats().policy_failures.fetch_add(bad);
    }
  }
}

// Fault the range in, place it, and pin it there.
//
// readahead(2) would be the direct expression of "get these bytes cached", but
// it needs an fd + file offset and kt only ever sees a virtual address, so the
// address-based equivalent is used: MADV_WILLNEED, which for a file mapping is
// the same force_page_cache_readahead() path. Chunked at 128 KiB to keep the
// queue full instead of asking for one huge readahead the kernel will clamp.
//
// Order is load bearing: readahead -> migrate -> mlock.
//   * migrate before mlock, because mlock's own populate pass is not what makes
//     the pages movable (the touch loop inside migrate_range is), and doing the
//     migration on pages that are not yet locked keeps this off the
//     "mlocked page" path in the kernel entirely.
//   * mlock last, so what gets pinned is the page as finally placed.
inline void pin_range(const void* addr, size_t len, const std::vector<int>& targets) {
  if (addr == nullptr || len == 0) return;
  const uintptr_t ps = static_cast<uintptr_t>(page_size());
  const uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(ps - 1);
  const uintptr_t end = (reinterpret_cast<uintptr_t>(addr) + len + ps - 1) & ~(ps - 1);
  const size_t span = static_cast<size_t>(end - start);

  stats().ranges.fetch_add(1);
  stats().bytes_requested.fetch_add(span);

  constexpr size_t kChunk = 128 * 1024;
  for (uintptr_t off = start; off < end; off += kChunk) {
    const size_t n = static_cast<size_t>(std::min<uintptr_t>(kChunk, end - off));
    madvise(reinterpret_cast<void*>(off), n, MADV_WILLNEED);
  }
  migrate_range(start, span, targets);
  if (::mlock(reinterpret_cast<void*>(start), span) == 0) {
    stats().bytes_locked.fetch_add(span);
  } else {
    stats().mlock_failures.fetch_add(1);
    stats().last_mlock_errno.store(static_cast<uint64_t>(errno));
  }
  if (audit_enabled()) {
    std::lock_guard<std::mutex> g(audit_mutex());
    audit_ranges().emplace_back(start, span);
  }
}

inline void print_stats(const char* tag) {
  const Stats& s = stats();
  const uint64_t pages = s.policy_pages.load();
  const uint64_t moved = s.policy_moved.load();
  const uint64_t fail = s.policy_failures.load();
  const double ps_gib = page_size() / 1073741824.0;
  const double sec = s.policy_ns.load() / 1e9;  // summed over threads, not wall clock
  printf("[kt zero-copy] %s: ranges=%llu requested=%.3f GiB locked=%.3f GiB mlock_failures=%llu (errno=%llu)\n",
         tag, (unsigned long long)s.ranges.load(), s.bytes_requested.load() / 1073741824.0,
         s.bytes_locked.load() / 1073741824.0, (unsigned long long)s.mlock_failures.load(),
         (unsigned long long)s.last_mlock_errno.load());
  // needed_move = pages that were NOT already on their target when we looked.
  // It is the "the mutation was actually applied" assertion: a placement
  // reading with needed_move=0 proves nothing about the mechanism.
  printf("[kt zero-copy] %s placement: move_pages calls=%llu pages=%llu (%.3f GiB) needed_move=%llu (%.2f%%) "
         "retried=%llu failed_pages=%llu (%.4f%%) last_status=%lld cpu_time=%.2fs\n",
         tag, (unsigned long long)s.policy_calls.load(), (unsigned long long)pages, pages * ps_gib,
         (unsigned long long)moved, pages ? 100.0 * moved / pages : 0.0,
         (unsigned long long)s.policy_retried.load(), (unsigned long long)fail,
         pages ? 100.0 * fail / pages : 0.0, (long long)s.last_page_status.load(), sec);
  if (fail != 0)
    printf("[kt zero-copy] 🔴 %llu of %llu pages did not land on the requested node (last per-page status %lld, "
           "syscall errno %llu). Placement readings from this run are not trustworthy.\n",
           (unsigned long long)fail, (unsigned long long)pages, (long long)s.last_page_status.load(),
           (unsigned long long)s.last_policy_errno.load());
}

// move_pages(2) in query mode + mincore(2) over everything that was pinned.
// Sampled by `stride` pages so the sweep stays cheap on 268 GiB.
//
// 🔴 The stride is forced ODD. Measured 2026-09-24: with stride=16 the
// KT_ZEROCOPY_POLICY=interleave arm audited as "node0=100.00%", because
// interleave assigns node by page index parity and a stride of 16 only ever
// samples even indices. The sampler was resonating with the thing it measured
// and reported a perfect result for a 50/50 placement. Any even stride aliases
// the same way; an odd one visits both parities.
inline void audit_report(int stride = 1) {
  if (stride > 1 && (stride % 2) == 0) stride += 1;
  std::vector<std::pair<uintptr_t, size_t>> ranges;
  {
    std::lock_guard<std::mutex> g(audit_mutex());
    ranges = audit_ranges();
  }
  if (ranges.empty()) {
    printf("[kt zero-copy audit] no ranges recorded\n");
    return;
  }
  const uintptr_t ps = static_cast<uintptr_t>(page_size());
  std::map<int, uint64_t> per_node;
  uint64_t sampled = 0, resident = 0, mincore_pages = 0;
  constexpr size_t kBatch = 4096;
  std::vector<void*> pages;
  std::vector<int> status;
  pages.reserve(kBatch);
  status.resize(kBatch);
  auto flush = [&]() {
    if (pages.empty()) return;
    long rc = syscall(SYS_move_pages, 0, (unsigned long)pages.size(), pages.data(), nullptr, status.data(), 0);
    if (rc != 0) {
      per_node[-1000] += pages.size();
    } else {
      for (size_t i = 0; i < pages.size(); ++i) per_node[status[i]]++;
    }
    sampled += pages.size();
    pages.clear();
  };
  std::vector<unsigned char> vec;
  for (const auto& r : ranges) {
    const uint64_t npages = r.second / ps;
    for (uint64_t i = 0; i < npages; i += static_cast<uint64_t>(stride)) {
      pages.push_back(reinterpret_cast<void*>(r.first + i * ps));
      if (pages.size() == kBatch) flush();
    }
    vec.assign(npages, 0);
    if (mincore(reinterpret_cast<void*>(r.first), r.second, vec.data()) == 0) {
      for (uint64_t i = 0; i < npages; ++i) {
        mincore_pages++;
        if (vec[i] & 1) resident++;
      }
    }
  }
  flush();
  printf("[kt zero-copy audit] move_pages sampled=%llu pages (stride=%d):", (unsigned long long)sampled, stride);
  for (const auto& kv : per_node) {
    printf("  node%d=%llu (%.2f%%)", kv.first, (unsigned long long)kv.second,
           sampled ? 100.0 * kv.second / sampled : 0.0);
  }
  printf("\n[kt zero-copy audit] mincore resident=%llu/%llu pages (%.4f%%)\n", (unsigned long long)resident,
         (unsigned long long)mincore_pages, mincore_pages ? 100.0 * resident / mincore_pages : 0.0);
}

}  // namespace kt_zerocopy

#endif  // KT_ZEROCOPY_WEIGHTS_H
