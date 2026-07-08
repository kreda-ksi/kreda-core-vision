#pragma once
#include "config.hpp"
#include "track_log.hpp"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>

namespace kreda {

cv::Mat gridOccupancy(const cv::Mat &bin_mask) {
    cv::Mat grid;
    cv::resize(bin_mask, grid, cv::Size(GRID_COLS, GRID_ROWS), 0, 0,
               cv::INTER_AREA);
    return grid;
}

int countActiveColumns(const cv::Mat &grid) {
    int active_cols = 0;

    for (int c{}; c < grid.cols; ++c) {
        for (int r{}; r < grid.rows; ++r)
            if (grid.at<std::uint8_t>(r, c) > GRID_CELL_ACTIVE) {
                active_cols++;
                break;
            }
    }

    return active_cols;
}

void decayGrid(cv::Mat &decayed) {
    if (!decayed.empty())
        decayed *= GRID_DECAY;
}

void updateDecayedGrid(cv::Mat &decayed, const cv::Mat &grid) {
    if (decayed.empty()) {
        grid.convertTo(decayed, CV_32F);
        return;
    }

    decayGrid(decayed);
    cv::Mat grid_f32;
    grid.convertTo(grid_f32, CV_32F);
    decayed = cv::max(decayed, grid_f32);
}

void logGrid(const cv::Mat &decayed, const std::string &filename,
             const std::string &reason, std::int64_t stream_ms,
             SidecarLogger &sidecar) {
    if (decayed.empty())
        return;

    cv::Mat grid_8u;
    decayed.convertTo(grid_8u, CV_8U);
    sidecar.logSave(filename, stream_ms, std::format("SAVE_{}", reason),
                    grid_8u);
}

} // namespace kreda
