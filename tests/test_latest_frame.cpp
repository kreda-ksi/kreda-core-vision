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

TEST_CASE("LatestFrame: shutdown frees a blocked producer") {
    LatestFrame lf;
    std::atomic<bool> running{true};

    cv::Mat first = tinyFrame(7);
    lf.push(first, 1, true, running); // slot now full

    std::atomic<bool> producer_done{false};
    std::thread producer([&] {
        cv::Mat second = tinyFrame(8);
        lf.push(second, 2, true, running); // blocks
        producer_done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK_FALSE(producer_done.load()); // blocked

    running = false;
    lf.wake();
    producer.join(); // this returns
    CHECK(producer_done.load());

    // blocked push was dropped, so slot still holds the 1st frame
    TimedFrame out;
    std::atomic<bool> running2{true};
    REQUIRE(lf.tryTake(out, running2));
    CHECK(out.stream_ms == 1);
}

TEST_CASE("LatestFrame: timestamps travel with frames") {
    LatestFrame lf;
    std::atomic<bool> running{true};
    cv::Mat m = tinyFrame(3);
    lf.push(m, 42, false, running);
    TimedFrame out;
    REQUIRE(lf.tryTake(out, running));
    CHECK(out.stream_ms == 42);
}
