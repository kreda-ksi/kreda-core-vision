#pragma once

#include "config.hpp"
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

namespace kreda {

struct WarpSet {
    std::array<cv::Mat, COLUMN_CNT> content; // CONTENT_WID x CONTENT_HEI
    std::array<cv::Mat, COLUMN_CNT> motion;  // MOTION_WID  x MOTION_HEI
};

WarpSet computeWarps(const std::vector<cv::Point2f> &src_points);

// runs the cal loop and returns the calculated mats
WarpSet runCalibration(const RunConfig &cfg, cv::VideoCapture &cap);
} // namespace kreda
