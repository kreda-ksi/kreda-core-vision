#pragma once

#include "config.hpp"
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

namespace kreda {

struct WarpSet {
    std::array<cv::Mat, COLUMN_CNT> content;
    std::array<cv::Size, COLUMN_CNT> content_dims;
    std::array<cv::Mat, COLUMN_CNT> motion;
};

WarpSet computeWarps(const std::vector<cv::Point2f> &src_points);
bool validateDrift(const cv::Mat &H, const cv::Size &frame_size);

// runs the cal loop and returns the calculated mats
WarpSet runCalibration(const RunConfig &cfg, cv::VideoCapture &cap);
} // namespace kreda
