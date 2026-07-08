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
