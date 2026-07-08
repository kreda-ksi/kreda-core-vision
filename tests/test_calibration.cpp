#include "cv/calibration.hpp"
#include "doctest.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

using namespace kreda;

namespace {
std::vector<cv::Point2f> twoColPoints() {
    return {{100, 100}, {500, 120}, {480, 400}, {90, 380},
            {600, 100}, {900, 110}, {890, 390}, {590, 400}};
}
} // namespace

TEST_CASE("warps: content warp maps clicked corners to output corners") {
    const WarpSet ws = computeWarps(twoColPoints());
    std::vector<cv::Point2f> src = {
        {100, 100}, {500, 120}, {480, 400}, {90, 380}};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, ws.content[0]);

    CHECK(dst[0].x == doctest::Approx(0).epsilon(0.01));
    CHECK(dst[0].y == doctest::Approx(0).epsilon(0.01));
    CHECK(dst[1].x == doctest::Approx(ws.content_dims[0].width).epsilon(0.01));
    CHECK(dst[2].y == doctest::Approx(ws.content_dims[0].height).epsilon(0.01));
    CHECK(dst[3].x == doctest::Approx(0).epsilon(0.01));
}

TEST_CASE("warps: content and motion sets differ per column") {
    const WarpSet ws = computeWarps(twoColPoints());
    for (unsigned int i{}; i < COLUMN_CNT; ++i)
        CHECK(cv::norm(ws.content[i], ws.motion[i], cv::NORM_INF) > 1e-9);
}

TEST_CASE("validateDrift: identity passes") {
    CHECK(validateDrift(cv::Mat::eye(3, 3, CV_64F), cv::Size(1920, 1080)));
}

TEST_CASE("validateDrift: 70% scale refuses") {
    cv::Mat h = cv::Mat::eye(3, 3, CV_64F) * 1.7;
    h.at<double>(2, 2) = 1.0;
    CHECK_FALSE(validateDrift(h, cv::Size(1920, 1080)));
}

TEST_CASE("validateDrift: 12% scale warns but accepts") {
    cv::Mat h = cv::Mat::eye(3, 3, CV_64F) * 1.12;
    h.at<double>(2, 2) = 1.0;
    CHECK(validateDrift(h, cv::Size(1920, 1080)));
}

TEST_CASE("validateDrift: fat perspective terms refuse") {
    cv::Mat h = cv::Mat::eye(3, 3, CV_64F);
    h.at<double>(2, 0) = 0.01;
    CHECK_FALSE(validateDrift(h, cv::Size(1920, 1080)));
}

TEST_CASE("validateDrift: frame-exit translation refuses") {
    cv::Mat h = cv::Mat::eye(3, 3, CV_64F);
    h.at<double>(0, 2) = 960.0; // half the frame out of bounds
    CHECK_FALSE(validateDrift(h, cv::Size(1920, 1080)));
}
