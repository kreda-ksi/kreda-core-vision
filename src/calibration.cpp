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

static void saveCalibration(const std::array<cv::Mat, COLUMN_CNT> &warps,
                            const std::vector<cv::Point2f> &src_points) {
    cv::FileStorage fs(CALIB_FILE, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << std::format("Could not open {} for writing.", CALIB_FILE)
                  << std::endl;
        return;
    }

    for (unsigned int i{}; i < COLUMN_CNT; ++i)
        fs << std::format("track_{}", i) << warps[i];
    fs << "src_points" << src_points;
}

static void drawCalibrationOverlay(cv::Mat &display,
                                   const std::vector<cv::Point2f> &points) {
    for (std::size_t i{}; i < points.size(); ++i) {
        cv::circle(display, points[i], 5, cv::Scalar(0, 0, 255), -1);

        auto mod_i = i % 4;
        if (mod_i > 0)
            cv::line(display, points[i - 1], points[i], cv::Scalar(0, 255, 0),
                     2);
        if (mod_i == 3)
            cv::line(display, points[i], points[i - 3], cv::Scalar(0, 255, 0),
                     2);
    }
}

static std::vector<cv::Point2f>
collectPointsInteractively(const cv::Mat &frame) {
    std::vector<cv::Point2f> src_points;
    src_points.reserve(POINTS_CNT);

    cv::namedWindow("Calibration", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Calibration", onMouseClick, &src_points);

    while (true) {
        cv::Mat display = frame.clone();
        drawCalibrationOverlay(display, src_points);
        cv::imshow("Calibration", display);

        if (src_points.size() == POINTS_CNT) {
            cv::waitKey(1000);
            break;
        }
        cv::waitKey(30);
    }

    cv::destroyWindow("Calibration");
    return src_points;
}

static std::array<cv::Mat, COLUMN_CNT>
computeWarps(const std::vector<cv::Point2f> &src_points) {
    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(OUT_WID, 0),
        cv::Point2f(OUT_WID, OUT_HEI),
        cv::Point2f(0, OUT_HEI),
    };

    std::array<cv::Mat, COLUMN_CNT> warps;
    for (unsigned int i{}; i < COLUMN_CNT; ++i) {
        std::vector<cv::Point2f> track_src(src_points.begin() + i * 4,
                                           src_points.begin() + (i + 1) * 4);
        warps[i] = cv::getPerspectiveTransform(track_src, dst_points);
    }

    return warps;
}

std::array<cv::Mat, COLUMN_CNT> runCalibration(cv::VideoCapture &cap) {
    std::array<cv::Mat, COLUMN_CNT> warps;

    if (loadCalibration(warps))
        return warps;

    cv::Mat frame;

    // read frame 1 for calibration
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to grab the first frame for calibration."
                  << std::endl;
        exit(ECAL);
    }


    auto src_points = collectPointsInteractively(frame);
    warps = computeWarps(src_points);
    saveCalibration(warps, src_points);

    return warps;
}

} // namespace kreda
