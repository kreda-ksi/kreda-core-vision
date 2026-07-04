#include "ingestion.hpp"
#include "config.hpp"
#include <chrono>
#include <format>
#include <iostream>
#include <thread>

namespace kreda {

void runIngestionLoop(cv::VideoCapture &cap, const std::string &rtsp_url,
                      const std::array<cv::Mat, COLUMN_CNT> &warp_matrices) {
    cv::Mat frame;
    std::array<cv::Mat, COLUMN_CNT> dewarped_boards;
    int retry_cnt = 0;

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0);

    while (true) {
        bool succ = cap.read(frame);

        if (!succ || frame.empty()) {
            retry_cnt++;
            std::cerr << std::format(
                             "Frame dropped or stream stuttered. Retry {}/{}.",
                             retry_cnt, MAX_RETRIES)
                      << std::endl;

            if (retry_cnt >= MAX_RETRIES) {
                std::cerr << "Stream lost completely. Reconnecting."
                          << std::endl;

                // reboot the connection
                cap.release();
                cap.open(rtsp_url, cv::CAP_FFMPEG);
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

                retry_cnt = 0;

                // wait a second for it to wake up
                cv::waitKey(1000);
                continue;
            }

            cv::waitKey(33);
            continue;
        }

        retry_cnt = 0;

        for (unsigned int i{}; i < COLUMN_CNT; ++i) {
            // homography
            cv::warpPerspective(frame, dewarped_boards[i], warp_matrices[i],
                                cv::Size(OUT_WID, OUT_HEI));
            // grayscale
            cv::Mat gray_board;
            cv::cvtColor(dewarped_boards[i], gray_board, cv::COLOR_BGR2GRAY);

            // clahe
            cv::Mat clahe_board;
            clahe->apply(gray_board, clahe_board);

            // invert
            cv::Mat final_board;
            cv::bitwise_not(clahe_board, final_board);

            // debug display
            if (SHOW_RAW)
                cv::imshow(std::format("KREDA column {} (raw)", i + 1),
                           dewarped_boards[i]);
            cv::imshow(std::format("KREDA column {}", i + 1), final_board);
        }
        cv::pollKey();
    }
}

} // namespace kreda
