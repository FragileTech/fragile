#include "thread_pool.hpp"

namespace fg {

ThreadPool::ThreadPool(int n_slots) : n_slots_(n_slots < 1 ? 1 : n_slots) {
  workers_.reserve(static_cast<size_t>(n_slots_ - 1));
  for (int slot = 1; slot < n_slots_; ++slot) {
    workers_.emplace_back([this, slot] { worker_loop(slot); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
  }
  cv_start_.notify_all();
  for (auto& w : workers_) w.join();
}

void ThreadPool::parallel_for(int32_t n,
                              const std::function<void(int32_t, int)>& fn) {
  if (n <= 0) return;
  if (n_slots_ == 1) {
    for (int32_t i = 0; i < n; ++i) fn(i, 0);
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_fn_ = &fn;
    job_n_ = n;
    completed_items_ = 0;
    ++generation_;
  }
  cv_start_.notify_all();

  // The caller participates as slot 0, running its static block.
  int32_t done_here = 0;
  const auto range = block_range(n, 0);
  for (int32_t i = range.first; i < range.second; ++i) {
    fn(i, 0);
    ++done_here;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  completed_items_ += done_here;
  if (completed_items_ == job_n_) {
    job_fn_ = nullptr;
  } else {
    cv_done_.wait(lock, [this] { return completed_items_ == job_n_; });
    job_fn_ = nullptr;
  }
}

std::pair<int32_t, int32_t> ThreadPool::block_range(int32_t n, int slot) const {
  const auto s = static_cast<int64_t>(n_slots_);
  const auto lo = static_cast<int32_t>(static_cast<int64_t>(n) * slot / s);
  const auto hi = static_cast<int32_t>(static_cast<int64_t>(n) * (slot + 1) / s);
  return {lo, hi};
}

void ThreadPool::worker_loop(int slot) {
  uint64_t seen_generation = 0;
  for (;;) {
    const std::function<void(int32_t, int)>* fn = nullptr;
    int32_t n = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock, [this, seen_generation] {
        return shutdown_ || (job_fn_ != nullptr && generation_ != seen_generation);
      });
      if (shutdown_) return;
      seen_generation = generation_;
      fn = job_fn_;
      n = job_n_;
    }

    int32_t done_here = 0;
    const auto range = block_range(n, slot);
    for (int32_t i = range.first; i < range.second; ++i) {
      (*fn)(i, slot);
      ++done_here;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      completed_items_ += done_here;
      if (completed_items_ == job_n_) cv_done_.notify_one();
    }
  }
}

}  // namespace fg
