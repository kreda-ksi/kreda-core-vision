#pragma once

#include "config.hpp"
#include <array>
#include <opencv2/opencv.hpp>
#include <string>

namespace kreda {

// runs the infinite ingestion -> dewarping -> processing loop
void runIngestionLoop(const RunConfig &cfg, cv::VideoCapture &cap,
                      const std::string &rtsp_url,
                      const std::array<cv::Mat, COLUMN_CNT> &warp_matrices);

bool openStream(cv::VideoCapture &cap, const std::string &url);

} // namespace kreda
