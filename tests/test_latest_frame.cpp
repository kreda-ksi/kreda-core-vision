#include "doctest.h"
#include "frames.hpp"

#include <atomic>
#include <chrono>
#include <opencv2/core.hpp>
#include <thread>

using namespace kreda;

namespace {
cv::Mat tinyFrame(int val) { return {4, 4, CV_8UC3, cv::Scalar(val, 0, 0)}; }
} // namespace

TEST_CASE("LatestFrame: lossless mode delivers all frames and joins") {
    LatestFrame lf;
    std::atomic<bool> running{true};
    constexpr int N = 200;
    std::atomic<int> received{0};

    std::thread producer([&] {
        for (int i{}; i < N; ++i) {
            cv::Mat m = tinyFrame(i % 256);
            lf.push(m, i, true, running);
        }
    });
    std::thread consumer([&] {
        TimedFrame tf;
        while (received.load() < N)
            if (lf.tryTake(tf, running))
                received++;
    });

    producer.join();
    while (received.load() < N)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    running = false;
    lf.wake();
    consumer.join();

    CHECK(received.load() == N);
}

