#pragma once

#include "core/config.hpp"
#include "cv/calibration.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace kreda {

// runs the infinite ingestion -> dewarping -> processing loop
void runIngestionLoop(const RunConfig &cfg, cv::VideoCapture &cap,
                      const std::string &rtsp_url, const WarpSet &warps);

bool openStream(cv::VideoCapture &cap, const std::string &url, bool is_file);

} // namespace kreda
