#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ga {

    // ─────────────────────────────────────────────────────────────────────────────
    // ThreadPool
    //
    // Fixed-size pool of worker threads that pulls tasks from a shared queue.
    // Workers are created once and reused across every generation — no per-round
    // thread creation overhead.
    //
    // Usage:
    //   ThreadPool pool;                   // defaults to hardware_concurrency()
    //   pool.enqueue([&]{ do_work(); });   // submit a task
    //   pool.wait_all();                   // block until all submitted tasks finish
    // ─────────────────────────────────────────────────────────────────────────────
    class ThreadPool {
    public:
        explicit ThreadPool(
            unsigned num_threads = std::thread::hardware_concurrency()
        );

        ~ThreadPool();

        // Submit a callable (lambda, function, etc.) to be executed by a worker.
        // Thread-safe — can be called from any thread.
        void enqueue(std::function<void()> task);

        // Block the calling thread until all enqueued tasks have completed.
        void wait_all();

        ThreadPool(const ThreadPool&)            = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

    private:
        std::vector<std::thread>          workers_;
        std::queue<std::function<void()>> tasks_;

        std::mutex              queue_mutex_;
        std::condition_variable task_available_;   // wakes workers when tasks arrive
        std::condition_variable all_done_;         // wakes wait_all() when active_ hits 0

        int  active_  = 0;    // tasks currently being executed
        bool stopped_ = false;

        void worker_loop();
    };

} // namespace ga
