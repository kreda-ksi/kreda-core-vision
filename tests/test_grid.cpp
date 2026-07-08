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
