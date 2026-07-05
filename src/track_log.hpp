#pragma once
#include "config.hpp"
#include "telemetry.hpp"
#include <chrono>
#include <format>
#include <fstream>
#include <string>

namespace kreda {

class TrackLogger {
  public:
    static TrackLogger &instance() {
        static TrackLogger logger;
        return logger;
    }

    void frame(const FrameTelemetry &t) {
        if (!ok_)
            return;
        file_ << std::format("{},{},FRAME,{},{},{},{},{},{}\n", elapsedMs(),
                             t.track_id, t.changed, int(t.is_moving),
                             int(t.is_sliding), int(t.slide_recover),
                             t.still_cnt, t.recover_cooldown);
        maybeFlush();
    }

    void event(unsigned int track_id, const std::string &what,
               long detail = 0) {
        if (!ok_)
            return;

        file_ << std::format("{},{},EVENT,{},{},,,\n", elapsedMs(), track_id,
                             what, detail);

        file_.flush();
    }

  private:
    TrackLogger() : start_(std::chrono::steady_clock::now()) {
        file_.open(LOG_FILE, std::ios::out | std::ios::trunc);
        ok_ = file_.is_open();
        if (ok_)
            file_ << "t_ms,track,type,changed_or_what,moving_or_detail,sliding,"
                     "recover,still_cnt,recover_cd\n";
    }

    long elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_)
            .count();
    }

    void maybeFlush() {
        if (++line_cnt_ % 250 == 0)
            file_.flush();
    }

    std::ofstream file_;
    bool ok_ = false;
    std::chrono::steady_clock::time_point start_;
    unsigned long line_cnt_ = 0;
};

} // namespace kreda
