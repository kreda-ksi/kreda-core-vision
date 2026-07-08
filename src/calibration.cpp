#include "calibration.hpp"
#include "config.hpp"
#include <format>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace kreda {

static void onMouseClick(int event, int x, int y, int /*flags*/,
                         void *userdata) {
    auto *points = static_cast<std::vector<cv::Point2f> *>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (points->size() < POINTS_CNT) {
            points->push_back(
                cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            std::cout << std::format("Point {}/{} recorded at ({},{})\n",
                                     points->size(), POINTS_CNT, x, y);
        } else {
            std::cout << std::format(
                "All {} points recorded. Press any key to continue.\n",
                POINTS_CNT);
        }
    }
}

static bool loadCalibration(const RunConfig &cfg,
                            std::vector<cv::Point2f> &src_points) {
    const cv::FileStorage fs(cfg.calib_file, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    fs["src_points"] >> src_points;
    if (src_points.size() != POINTS_CNT) {
        std::cout << "Calibration file invalid or outdated. Running manual "
                     "calibration.\n";
        return false;
    }

    return true;
}

static void saveCalibration(const RunConfig &cfg,
                            const std::vector<cv::Point2f> &src_points,
                            const cv::Mat &ref_frame) {
    cv::imwrite(cfg.ref_file, ref_frame);

    cv::FileStorage fs(cfg.calib_file, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << std::format("Could not open {} for writing.\n",
                                 cfg.calib_file);
        return;
    }

    fs << "src_points" << src_points;

    for (std::size_t i{}; i < COLUMN_CNT; ++i) {
        auto a = src_points[i * 4], b = src_points[i * 4 + 1],
             c = src_points[i * 4 + 2], d = src_points[i * 4 + 3];
        double top = cv::norm(b - a), bottom = cv::norm(c - d);
        double left = cv::norm(d - a), right = cv::norm(c - b);
        std::cout << std::format("col {}: top {:.0f} bottom {:.0f} "
                                 "left {:.0f} right {:.0f}\n",
                                 i, top, bottom, left, right);
    }
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
warpsFor(const std::vector<cv::Point2f> &src_points, float w, float h) {
    const std::vector<cv::Point2f> dst = {{0, 0}, {w, 0}, {w, h}, {0, h}};
    std::array<cv::Mat, COLUMN_CNT> warps;

    for (std::size_t i{}; i < COLUMN_CNT; ++i) {
        const std::vector<cv::Point2f> track_src(
            src_points.begin() + i * 4, src_points.begin() + (i + 1) * 4);
        warps[i] = cv::getPerspectiveTransform(track_src, dst);
    }

    return warps;
}

WarpSet computeWarps(const std::vector<cv::Point2f> &src_points) {
    return {warpsFor(src_points, CONTENT_WID, CONTENT_HEI),
            warpsFor(src_points, MOTION_WID, MOTION_HEI)};
}

// H maps current frame coords -> ref frame coords
static bool estimateDrift(const cv::Mat &ref, const cv::Mat &curr, cv::Mat &H) {
    cv::Mat ref_gray, curr_gray;
    cv::cvtColor(ref, ref_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr, curr_gray, cv::COLOR_BGR2GRAY);

    auto orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kp_ref, kp_curr;
    cv::Mat desc_ref, desc_curr;
    orb->detectAndCompute(ref_gray, cv::noArray(), kp_ref, desc_ref);
    orb->detectAndCompute(curr_gray, cv::noArray(), kp_curr, desc_curr);

    if (desc_ref.empty() || desc_curr.empty())
        return false;

    const cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc_curr, desc_ref, knn, 2);

    std::vector<cv::Point2f> pts_curr, pts_ref;
    for (const auto &m : knn) {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
            pts_curr.push_back(kp_curr[m[0].queryIdx].pt);
            pts_ref.push_back(kp_ref[m[0].trainIdx].pt);
        }
    }
    if (pts_curr.size() < 4)
        return false;

    cv::Mat inlier_mask;
    H = cv::findHomography(pts_curr, pts_ref, cv::RANSAC, 3.0, inlier_mask);
    if (H.empty())
        return false;

    int inliers = cv::countNonZero(inlier_mask);
    if (inliers < DRIFT_MIN_INLIERS) {
        std::cerr << std::format("Drift estimation rejected, {} inliers.\n",
                                 inliers);
        return false;
    }

    return true;
}

static bool validateDrift(const cv::Mat &H, const cv::Size &frame_size) {
    // scale check via sqrt(|det|) of the upper-left 2x2
    const double det = H.at<double>(0, 0) * H.at<double>(1, 1) -
                       H.at<double>(0, 1) * H.at<double>(1, 0);
    const double scale = std::sqrt(std::abs(det));

    if (std::abs(scale - 1.0) > DRIFT_MAX_SCALE_DEV)
        std::cout << std::format(
            "zoom drift {:+.0f}%, output quality might be degraded.\n",
            (scale - 1.0) * 100.0);

    // 50%+ zoom is not drift
    if (std::abs(scale - 1.0) > 0.5)
        return false;

    // perpsective terms should be tiny
    if (std::abs(H.at<double>(2, 0)) > 1e-3 ||
        std::abs(H.at<double>(2, 1)) > 1e-3)
        return false;

    // frame corners must be roughly inside
    const std::vector<cv::Point2f> corners = {
        {0.0f, 0.0f},
        {static_cast<float>(frame_size.width), 0.0f},
        {static_cast<float>(frame_size.width),
         static_cast<float>(frame_size.height)},
        {0.0f, static_cast<float>(frame_size.height)},
    };
    std::vector<cv::Point2f> mapped;
    cv::perspectiveTransform(corners, mapped, H);
    for (const auto &p : mapped)
        if (p.x < -0.2f * frame_size.width || p.x > 1.2f * frame_size.width ||
            p.y < -0.2f * frame_size.height || p.y > 1.2f * frame_size.height)
            return false;

    return true;
}

static bool applyDriftCorrection(const RunConfig &cfg, WarpSet &warps,
                                 cv::VideoCapture &cap) {
    const cv::Mat ref = cv::imread(cfg.ref_file);
    if (ref.empty())
        return false;

    cv::Mat curr;
    if (!cap.read(curr) || curr.empty())
        return false;

    cv::Mat H;
    if (!estimateDrift(ref, curr, H))
        return false;
    if (!validateDrift(H, curr.size()))
        return false;

    for (auto &w : warps.content)
        w = w * H;
    for (auto &w : warps.motion)
        w = w * H;
    std::cout << "Calibration drift-corrected.\n";
    return true;
}

WarpSet runCalibration(const RunConfig &cfg, cv::VideoCapture &cap) {
    std::vector<cv::Point2f> src_points;

    if (loadCalibration(cfg, src_points) && !cfg.force_recalibrate) {
        WarpSet warps = computeWarps(src_points);
        if (!applyDriftCorrection(cfg, warps, cap))
            std::cout
                << "Drift correction unavailable, using stored calibration.\n";
        return warps;
    }

    if (!cfg.show_gui) {
        std::cerr << "Headless mode requires an existing calibration file.\n";
        exit(ECAL);
    }

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "Failed to grab the first frame for calibration.\n";
        exit(ECAL);
    }

    src_points = collectPointsInteractively(frame);
    saveCalibration(cfg, src_points, frame);
    return computeWarps(src_points);
}

} // namespace kreda
