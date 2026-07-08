#include "config.hpp"
#include "doctest.h"
#include "grid.hpp"
#include <opencv2/opencv.hpp>

using namespace kreda;

namespace {
cv::Mat motionMask() {
    return cv::Mat::zeros(static_cast<int>(MOTION_HEI),
                          static_cast<int>(MOTION_WID), CV_8U);
}
} // namespace

TEST_CASE("grid: full-field slide activates every column") {
    cv::Mat mask(static_cast<int>(MOTION_HEI), static_cast<int>(MOTION_WID),
                 CV_8U, cv::Scalar(255));
    CHECK(countActiveColumns(gridOccupancy(mask)) == GRID_COLS);
}

TEST_CASE("grid: thin edge-band slide activates every column") {
    cv::Mat mask = motionMask();
    cv::rectangle(mask, cv::Point(0, 300),
                  cv::Point(static_cast<int>(MOTION_WID) - 1, 380),
                  cv::Scalar(255), cv::FILLED);
    CHECK(countActiveColumns(gridOccupancy(mask)) == GRID_COLS);
}

TEST_CASE("grid: centered person blob stays under the veto line") {
    cv::Mat mask = motionMask();
    cv::rectangle(mask, cv::Point(490, 135),
                  cv::Point(490 + MOTION_WID / 4, 135 + MOTION_WID / 2),
                  cv::Scalar(255), cv::FILLED);
    CHECK(countActiveColumns(gridOccupancy(mask)) < SLIDE_MIN_ACTIVE_COLS);
}

TEST_CASE("grid: arms out stays under SLIDE_MIN_ACTIVE_COLS") {
    cv::Mat mask = motionMask();
    cv::rectangle(mask, cv::Point(190, 260),
                  cv::Point(190 + MOTION_WID / 2, 460), cv::Scalar(255),
                  cv::FILLED);
    CHECK(countActiveColumns(gridOccupancy(mask)) < SLIDE_MIN_ACTIVE_COLS);
}

TEST_CASE("grid: empty mask activates nothing") {
    CHECK(countActiveColumns(gridOccupancy(motionMask())) == 0);
}

TEST_CASE("grid: occupancy output has contract dimensions and type") {
    const cv::Mat g = gridOccupancy(motionMask());
    CHECK(g.cols == GRID_COLS);
    CHECK(g.rows == GRID_ROWS);
    CHECK(g.type() == CV_8U);
}

TEST_CASE("decay: first grid initializes the state") {
    cv::Mat decayed;
    cv::Mat grid(GRID_ROWS, GRID_COLS, CV_8U, cv::Scalar(100));
    updateDecayedGrid(decayed, grid);
    CHECK(decayed.at<float>(0, 0) == 100.0f);
}

TEST_CASE("decay: constant input is a fixed point (no drift under activity)") {
    cv::Mat decayed;
    cv::Mat grid(GRID_ROWS, GRID_COLS, CV_8U, cv::Scalar(100));
    for (int i = 0; i < 127; ++i)
        updateDecayedGrid(decayed, grid);
    CHECK(decayed.at<float>(0, 0) == 100.0f);
}

TEST_CASE("decay: cell decays by GRID_DECAY^n over n quiet frames") {
    cv::Mat decayed;
    cv::Mat seed = cv::Mat::zeros(GRID_ROWS, GRID_COLS, CV_8U);
    seed.at<std::uint8_t>(5, 5) = 200;
    updateDecayedGrid(decayed, seed);

    const cv::Mat quiet = cv::Mat::zeros(GRID_ROWS, GRID_COLS, CV_8U);
    for (int i = 0; i < 30; ++i)
        updateDecayedGrid(decayed, quiet);

    const float expected = 200.0f * std::pow(GRID_DECAY, 30.0f);
    CHECK(decayed.at<float>(5, 5) == doctest::Approx(expected).epsilon(0.01));
}

TEST_CASE("decay: fresh activity overrides decayed value per-cell") {
    cv::Mat decayed(GRID_ROWS, GRID_COLS, CV_32F, cv::Scalar(50.0f));
    cv::Mat grid = cv::Mat::zeros(GRID_ROWS, GRID_COLS, CV_8U);
    grid.at<std::uint8_t>(2, 2) = 180;
    updateDecayedGrid(decayed, grid);

    CHECK(decayed.at<float>(2, 2) == doctest::Approx(180.0f));
    CHECK(decayed.at<float>(3, 3) == doctest::Approx(50.0f * GRID_DECAY));
}
