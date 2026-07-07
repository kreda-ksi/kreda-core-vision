#pragma once
#include "config.hpp"
#include "telemetry.hpp"
#include <format>
#include <opencv2/imgproc.hpp>
#include <string>

namespace kreda {

inline void drawHud(cv::Mat *display, const FrameTelemetry &t) {
    if (!display)
        return;

    cv::rectangle(*display, cv::Point(0, 0), cv::Point(display->cols, 58),
                  cv::Scalar(0, 0, 0), cv::FILLED);

    std::string phase;
    cv::Scalar col;
    if (t.slide_recover) {
        phase =
            std::format("RECOVERING {}/{}", t.recover_cooldown, SLIDE_COOLDOWN);
        col = cv::Scalar(0, 0, 255);
    } else if (t.is_sliding) {
        phase = "SLIDING";
        col = cv::Scalar(0, 165, 255);
    } else if (t.is_moving) {
        phase = "MOVING";
        col = cv::Scalar(0, 255, 255);
    } else {
        phase = std::format("STILL {}/{}", t.still_cnt, STILL_COOLDOWN);
        col = cv::Scalar(0, 255, 0);
    }

    cv::putText(*display, phase, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX,
                0.6, col, 2);

    cv::putText(*display,
                std::format("changed: {} (move>{} slide>{})", t.changed,
                            MOTION_TRIGGER_PXS, SLIDE_TRIGGER_PXS),
                cv::Point(10, 46), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);

    const int bar_w = static_cast<int>(
        display->cols * std::min(1.0, double(t.changed) / SLIDE_TRIGGER_PXS));
    cv::rectangle(*display, cv::Point(0, 54), cv::Point(bar_w, 58), col,
                  cv::FILLED);

    const int tick_x = static_cast<int>(
        display->cols * (double(MOTION_TRIGGER_PXS) / SLIDE_TRIGGER_PXS));
    cv::line(*display, cv::Point(tick_x, 52), cv::Point(tick_x, 58),
             cv::Scalar(255, 255, 255), 1);

    if (t.save_flash > 0)
        cv::putText(*display, "SAVED", cv::Point(display->cols - 110, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

} // namespace kreda
