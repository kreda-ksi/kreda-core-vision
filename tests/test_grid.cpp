#include "config.hpp"
#include "doctest.h"
#include "grid.hpp"
#include <opencv2/opencv.hpp>

using namespace kreda;

TEST_CASE("grid: full-field slide activates every column") {
    cv::Mat mask(static_cast<int>(MOTION_HEI), static_cast<int>(MOTION_WID),
                 CV_8U, cv::Scalar(255));
    CHECK(countActiveColumns(gridOccupancy(mask)) == GRID_COLS);
}
