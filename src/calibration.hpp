#pragma once

#include "config.hpp"
#include <array>
#include <opencv2/opencv.hpp>

namespace kreda {
// runs the cal loop and returns the calculated mats
std::array<cv::Mat, COLUMN_CNT> runCalibration(cv::VideoCapture &cap);
} // namespace kreda
