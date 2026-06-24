#include <ga/thread_pool.hpp>

namespace ga {

    // ─────────────────────────────────────────────
    // Constructor — spawn N worker threads
    // ─────────────────────────────────────────────

    ThreadPool::ThreadPool(unsigned num_threads) {
        workers_.reserve(num_threads);
        for (unsigned i = 0; i < num_threads; ++i)
            workers_.emplace_back(&ThreadPool::worker_loop, this);
    }


    // ─────────────────────────────────────────────
    // Destructor — drain the queue and join all workers
    // ─────────────────────────────────────────────

    ThreadPool::~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopped_ = true;
        }
        task_available_.notify_all();   // wake every sleeping worker so they can exit
        for (auto& w : workers_) w.join();
    }


    // ─────────────────────────────────────────────
    // enqueue — push a task and wake one worker
    // ─────────────────────────────────────────────

    void ThreadPool::enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            tasks_.push(std::move(task));
            ++active_;
        }
        task_available_.notify_one();
    }


    // ─────────────────────────────────────────────
    // wait_all — block until active_ == 0
    // ─────────────────────────────────────────────

    void ThreadPool::wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        all_done_.wait(lock, [this]{ return active_ == 0; });
    }


    // ─────────────────────────────────────────────
    // worker_loop — each thread runs this forever
    // ─────────────────────────────────────────────

    void ThreadPool::worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Sleep until there is a task or the pool is shutting down
                task_available_.wait(lock, [this]{
                    return !tasks_.empty() || stopped_;
                });

                if (stopped_ && tasks_.empty()) return;

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            task();   // execute outside the lock

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                --active_;
            }
            all_done_.notify_one();   // wake wait_all() in case we just hit 0
        }
    }

} // namespace ga
