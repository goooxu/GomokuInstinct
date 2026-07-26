// 常驻线程池。
//
// collect/apply 每手要被调用几百次，每次都新建线程的开销不可忽略，
// 所以线程只创建一次，之后靠代次号唤醒。自博弈与批量搜索共用这一份。
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gi {

class ThreadPool {
 public:
  explicit ThreadPool(int n) {
    for (int i = 0; i < n; ++i) {
      threads_.emplace_back([this] { worker(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : threads_) t.join();
  }

  void run(const std::function<void(int)>& fn, int count) {
    if (threads_.empty()) {
      for (int i = 0; i < count; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      active_ = static_cast<int>(threads_.size());
      ++generation_;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] { return active_ == 0; });
  }

 private:
  void worker() {
    uint64_t seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      const std::function<void(int)>* fn = fn_;
      const int count = count_;
      lock.unlock();

      int i;
      while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < count) {
        (*fn)(i);
      }

      lock.lock();
      if (--active_ == 0) cv_done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(int)>* fn_ = nullptr;
  int count_ = 0;
  std::atomic<int> next_{0};
  int active_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace gi
