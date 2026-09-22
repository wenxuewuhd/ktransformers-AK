/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>

// ---------------------------------------------------------------------------
// KT_CPUINFER_INLINE: run a submitted task on the calling thread instead of
// handing it to the TaskQueue worker.
//
// Why it exists: on the DSV4.1 NPU-offload path one MoE layer is one
// submit+sync from inside a graph host-func callback, and the hand-off is a
// full thread round trip -- enqueue (new Node + mutex + notify), the worker
// waking up, and the worker's notify waking the sync. Measured empty-task
// round trip on this box: 10.44 us median (p90 11.0), against 0.04 us for
// running the same task inline; 40 layers per token makes that 0.42 ms.
//
// Unset, empty or "0" = off = byte-for-byte the old path. Read once, and the
// one line it prints when on is how a run's log says which arm it was.
// ---------------------------------------------------------------------------
bool kt_parse_inline_flag(const char* value) {
  if (value == nullptr || *value == '\0') return false;
  if (std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  fprintf(stderr, "[kt] KT_CPUINFER_INLINE=%s is not 0 or 1; keeping the default (off)\n", value);
  return false;
}

bool kt_cpuinfer_inline_enabled() {
  static const bool enabled = [] {
    bool on = kt_parse_inline_flag(std::getenv("KT_CPUINFER_INLINE"));
    if (on) {
      printf("[kt] CPUInfer: inline execution ON (KT_CPUINFER_INLINE=1): a submitted task runs on the calling "
             "thread and the TaskQueue worker is bypassed; it still falls back to the queue while anything is "
             "pending\n");
    }
    return on;
  }();
  return enabled;
}

TaskQueue::TaskQueue() : done(false), pending(0) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::run_inline(const std::function<void()>& task) {
  std::exception_ptr task_exception;
  if (task) {
    try {
      task();
    } catch (...) {
      task_exception = std::current_exception();
    }
  }
  if (task_exception) {
    // Same slot the worker uses, so sync() rethrows it exactly as it would
    // have for a queued task. Dropping it here would turn a failed MoE layer
    // into silently stale output.
    std::lock_guard<std::mutex> lock(mtx);
    if (!first_exception) {
      first_exception = task_exception;
    }
  }
}

size_t TaskQueue::pending_tasks() const { return pending.load(std::memory_order_acquire); }

void TaskQueue::sync(size_t allow_n_pending) {
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}
