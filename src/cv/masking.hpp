#pragma once
#include <opencv2/opencv.hpp>

namespace kreda {

int countChalkPixels(const cv::Mat &bin_mask, int max_comp_area) {
    cv::Mat labels, stats, centroids;
    const int n =
        cv::connectedComponentsWithStats(bin_mask, labels, stats, centroids);

    int chalk_pxs = 0;
    for (int i = 1; i < n; ++i) { // 0 is bg
        const int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a < max_comp_area)
            chalk_pxs += a;
    }

    return chalk_pxs;
}

} // namespace kreda
