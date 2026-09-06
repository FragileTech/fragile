// Fixed-size std::thread pool with a blocking parallel_for.
//
// Emscripten builds run this on pthreads: the pool must not exceed the
// preallocated PTHREAD_POOL_SIZE (workers are created once, at startup).
// The calling thread participates in the work as slot 0; pool workers are
// slots 1..size-1, so parallel_for's fn receives slot ids in [0, size).
#ifndef FRACTAL_GAS_THREAD_POOL_HPP
#define FRACTAL_GAS_THREAD_POOL_HPP

#include <condition_variable>
#include <utility>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace fg {

class ThreadPool {
 public:
  /// n_slots >= 1 total execution slots (caller + n_slots-1 workers).
  explicit ThreadPool(int n_slots);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_slots_; }

  /// Runs fn(i, slot) for every i in [0, n); blocks until all complete.
  /// Each slot id is used by exactly one thread at a time, so fn may use
  /// slot-indexed scratch resources (e.g. one emulator per slot) unlocked.
  /// Work is split by STATIC block partition (index i always maps to the
  /// same slot for a given n and pool size) — never by work stealing. This
  /// keeps runs reproducible: the nes-py emulator does not serialize the
  /// controller strobe/shift-register state, so which emulator instance
  /// services which walker must not depend on thread scheduling.
  void parallel_for(int32_t n, const std::function<void(int32_t, int)>& fn);

 private:
  void worker_loop(int slot);
  std::pair<int32_t, int32_t> block_range(int32_t n, int slot) const;

  int n_slots_;
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(int32_t, int)>* job_fn_ = nullptr;
  int32_t job_n_ = 0;
  int completed_slots_ = 0;
  uint64_t generation_ = 0;
  bool shutdown_ = false;
};

}  // namespace fg

#endif  // FRACTAL_GAS_THREAD_POOL_HPP
