#include "args.hpp"
#include "calibration.hpp"
#include "config.hpp"
#include "ingestion.hpp"
#include "signals.hpp"
#include <csignal>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace kreda;

int main(int argc, char **argv) {
    auto cfg = parseArgs(argc, argv);
    installSigHandlers();

    std::cout << std::format("Connecting to: {}\n", cfg.rtsp_url);

    cv::VideoCapture cap;
    if (!openStream(cap, cfg.rtsp_url, cfg.is_file)) {
        std::cerr << "Could not open the RTSP stream. Is mediamtx running?\n";
        return ERTSP;
    }

    const WarpSet warps = runCalibration(cfg, cap);
    runIngestionLoop(cfg, cap, cfg.rtsp_url, warps);

    // cleanup
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
