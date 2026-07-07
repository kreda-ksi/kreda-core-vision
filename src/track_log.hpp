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
    static TrackLogger &instance(const RunConfig &cfg) {
        static TrackLogger logger(cfg);
        return logger;
    }

    void frame(const FrameTelemetry &t) {
        if (!ok_)
            return;

        TrackData &prev = last_[t.track_id];
        auto now = std::chrono::steady_clock::now();

        const bool should_update =
            !prev.valid || t.is_moving != prev.t.is_moving ||
            t.is_sliding != prev.t.is_sliding ||
            t.slide_recover != prev.t.slide_recover ||
            std::abs(t.changed - prev.t.changed) > FRAME_LOG_DELTA;
        if (!should_update)
            return;

        file_ << std::format("{},{},FRAME,{},{},{},{},{},{}\n", elapsedMs(),
                             t.track_id, t.changed, int(t.is_moving),
                             int(t.is_sliding), int(t.slide_recover),
                             t.still_cnt, t.recover_cooldown);
        prev.t = t;
        prev.written_at = now;
        prev.valid = true;
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
    TrackLogger(const RunConfig &cfg)
        : start_(std::chrono::steady_clock::now()) {
        file_.open(cfg.log_file, std::ios::out | std::ios::trunc);
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

    struct TrackData {
        FrameTelemetry t{};
        std::chrono::steady_clock::time_point written_at{};
        bool valid = false;
    };
    std::array<TrackData, COLUMN_CNT> last_{};

    std::ofstream file_;
    bool ok_ = false;
    std::chrono::steady_clock::time_point start_;
    unsigned long line_cnt_ = 0;
};

} // namespace kreda
