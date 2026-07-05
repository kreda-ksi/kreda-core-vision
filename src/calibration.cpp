#include "calibration.hpp"
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace kreda {

static void onMouseClick(int event, int x, int y, int flags, void *userdata) {
    auto *points = static_cast<std::vector<cv::Point2f> *>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (points->size() < POINTS_CNT) {
            points->push_back(
                cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            std::cout << std::format("Point {}/{} recorded at ({},{})",
                                     points->size(), POINTS_CNT, x, y)
                      << std::endl;
        } else {
            std::cout
                << std::format(
                       "All {} points recorded. Press any key to continue.",
                       POINTS_CNT)
                << std::endl;
        }
    }
}

static bool loadCalibration(std::array<cv::Mat, COLUMN_CNT> &warps) {
    cv::FileStorage fs(CALIB_FILE, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    for (unsigned int i{}; i < COLUMN_CNT; ++i) {
        fs[std::format("track_{}", i)] >> warps[i];
        if (warps[i].empty()) {
            std::cout << "Calibration file invalid or outdated. Running manual "
                         "calibration."
                      << std::endl;
            return false;
        }
    }

    return true;
}

std::array<cv::Mat, COLUMN_CNT> runCalibration(cv::VideoCapture &cap) {
    std::array<cv::Mat, COLUMN_CNT> warp_matrices;

    if (loadCalibration(warp_matrices))
        return warp_matrices;

    cv::Mat frame;

    // read frame 1 for calibration
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to grab the first frame for calibration."
                  << std::endl;
        exit(ECAL);
    }

    std::vector<cv::Point2f> src_points;
    cv::namedWindow("Calibration", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Calibration", onMouseClick, &src_points);

    while (true) {
        cv::Mat display_frame = frame.clone();
        for (std::size_t i{}; i < src_points.size(); ++i) {
            cv::circle(display_frame, src_points[i], 5, cv::Scalar(0, 0, 255),
                       -1);

            auto mod_i = i % 4;
            if (mod_i > 0)
                cv::line(display_frame, src_points[i - 1], src_points[i],
                         cv::Scalar(0, 255, 0), 2);
            if (mod_i == 3)
                cv::line(display_frame, src_points[i], src_points[i - 3],
                         cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Calibration", display_frame);

        if (src_points.size() == 8) {
            cv::waitKey(1000);
            break;
        }

        cv::waitKey(30);
    }
    cv::destroyWindow("Calibration");

    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(OUT_WID, 0),
        cv::Point2f(OUT_WID, OUT_HEI),
        cv::Point2f(0, OUT_HEI),
    };

    cv::FileStorage fs_write(CALIB_FILE, cv::FileStorage::WRITE);
    if (!fs_write.isOpened()) {
        std::cerr << std::format("Could not open {} for writing.", CALIB_FILE)
                  << std::endl;
    }

    for (unsigned int i{}; i < COLUMN_CNT; ++i) {
        std::vector<cv::Point2f> track_src(src_points.begin() + i * 4,
                                           src_points.begin() + (i + 1) * 4);
        warp_matrices[i] = cv::getPerspectiveTransform(track_src, dst_points);

        if (fs_write.isOpened())
            fs_write << std::format("track_{}", i) << warp_matrices[i];
    }

    if (fs_write.isOpened())
        fs_write.release();

    return warp_matrices;
}

} // namespace kreda
