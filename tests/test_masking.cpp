#include "doctest.h"
#include "masking.hpp"
#include <opencv2/imgproc.hpp>

using namespace kreda;

TEST_CASE("masking: small blob counts as chalk") {
    cv::Mat mask = cv::Mat::zeros(720, 1280, CV_8U);
    cv::rectangle(mask, {100, 100}, {109, 109}, cv::Scalar(255), cv::FILLED);
    CHECK(countChalkPixels(mask, 1000) == 100);
}
