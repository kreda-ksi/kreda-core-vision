#include "calibration.hpp"
#include "config.hpp"
#include "ingestion.hpp"
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace kreda;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << std::format("Usage: ./{} <rtsp_ip> [rtsp_port]", argv[0])
                  << std::endl;
        return EARG;
    }

    std::string rtsp_url =
        std::format("rtsp://{}:{}/live", argv[1], argc > 2 ? argv[2] : "8554");

    std::cout << std::format("Connecting to: {}", rtsp_url) << std::endl;

    cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Could not open the RTSP stream. Is mediamtx running?"
                  << std::endl;
        return ERTSP;
    }

    // set buffer size to one frame to prevent any delay buildup
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::array<cv::Mat, COLUMN_CNT> warp_matrices = runCalibration(cap);
    runIngestionLoop(cap, rtsp_url, warp_matrices);

    // cleanup
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
