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

TEST_CASE("LatestFrame: drop mode keeps only the latest frame") {
    LatestFrame lf;
    std::atomic<bool> running{true};

    cv::Mat a = tinyFrame(1);
    cv::Mat b = tinyFrame(2);
    lf.push(a, 100, false, running);
    lf.push(b, 200, false, running);

    TimedFrame out;
    REQUIRE(lf.tryTake(out, running));
    CHECK(out.stream_ms == 200);
    CHECK(out.frame.at<cv::Vec3b>(0, 0)[0] == 2);

    CHECK_FALSE(lf.tryTake(out, running));
}
