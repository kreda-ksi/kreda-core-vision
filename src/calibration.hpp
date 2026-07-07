#pragma once

#include "config.hpp"
#include <array>
#include <opencv2/opencv.hpp>

namespace kreda {

struct WarpSet {
    std::array<cv::Mat, COLUMN_CNT> content; // CONTENT_WID x CONTENT_HEI
    std::array<cv::Mat, COLUMN_CNT> motion;  // MOTION_WID  x MOTION_HEI
};

// runs the cal loop and returns the calculated mats
WarpSet runCalibration(const RunConfig &cfg, cv::VideoCapture &cap);
} // namespace kreda
