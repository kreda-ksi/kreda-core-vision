#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace kreda {

struct TimedFrame {
    cv::Mat frame;
    std::int64_t stream_ms = 0;
};

class LatestFrame {
  public:
    void push(cv::Mat &f, std::int64_t stream_ms, bool wait_if_full,
              const std::atomic<bool> &running) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (wait_if_full)
            space_cv_.wait(lock, [&] { return !full_ || !running; });
        if (!running)
            return;
        cv::swap(frame_.frame, f);
        frame_.stream_ms = stream_ms;
        full_ = true;
        lock.unlock();
        cv_.notify_one();
    }

    bool tryTake(TimedFrame &out, const std::atomic<bool> &running) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(100),
                          [&] { return full_ || !running; }))
            return false; // timeout, no frame and no caller loops
        if (!full_)
            return false;

        cv::swap(out.frame, frame_.frame);
        out.stream_ms = frame_.stream_ms;
        full_ = false;
        lock.unlock();
        space_cv_.notify_one();
        return true;
    }

    void wake() {
        cv_.notify_all();
        space_cv_.notify_all();
    }

  private:
    TimedFrame frame_;
    bool full_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable space_cv_;
};

} // namespace kreda
