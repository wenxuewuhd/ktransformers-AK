/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-16 10:43:18
 * @Version   : 1.0.0
 * @LastEditors : chenht
 * @LastEditTime : 2024-10-09 11:08:07
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// KT_CPUINFER_INLINE, parsed and read once (task_queue.cpp explains the why).
// The parse is exposed so it can be checked without starting a queue.
bool kt_parse_inline_flag(const char* value);
bool kt_cpuinfer_inline_enabled();

class TaskQueue {
 public:
  TaskQueue();
  ~TaskQueue();

  void enqueue(std::function<void()>);

  // Run the task on THIS thread instead of handing it to the worker
  // (DSV4.1 offload §4.22 P2: the hand-off costs a measured 10.44 us of pure
  // thread ping-pong per layer). The exception path is kept identical to the
  // worker's: a throwing task is stored and rethrown by the next sync(), not
  // here -- kt's callers have always learnt about a failed task at sync time
  // and a task that raised earlier than that would be a different contract.
  // Only safe while nothing is queued; the caller checks pending_tasks().
  void run_inline(const std::function<void()>&);

  // Tasks enqueued and not yet finished. Reading it is the only way to know
  // that running a task inline would jump the queue.
  size_t pending_tasks() const;

  void sync(size_t allow_n_pending);

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    Node() : task(nullptr), next(nullptr) {}
    Node(const std::function<void()>& t) : task(t), next(nullptr) {}
  };

  std::atomic<Node*> head;
  std::atomic<Node*> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;
  std::mutex mtx;
  std::condition_variable cv;
  std::exception_ptr first_exception;

  void worker();
};

#endif
