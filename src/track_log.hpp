#pragma once
#include "config.hpp"
#include "telemetry.hpp"
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <mutex>
#include <string>

namespace kreda {

class TrackLogger {
  public:
    explicit TrackLogger(const RunConfig &cfg) {
        if (!cfg.log_enabled)
            return;
        file_.open(cfg.log_file, std::ios::out | std::ios::trunc);
        ok_ = file_.is_open();
        if (ok_) {
            file_ << "t_ms,track,type,changed_or_what,detail\n";
            const auto epoch_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
            file_ << std::format("0,0,EVENT,RUN_START,{}\n", epoch_ms);
            file_.flush();
        }
    }

    TrackLogger(const TrackLogger &) = delete;
    TrackLogger &operator=(const TrackLogger &) = delete;

    void frame(const FrameTelemetry &t, std::int64_t stream_ms) {
        if (!ok_)
            return;
        const std::lock_guard<std::mutex> lock(mtx_);

        TrackData &prev = last_[t.track_id];

        const bool should_update =
            !prev.valid || t.is_moving != prev.t.is_moving ||
            t.is_sliding != prev.t.is_sliding ||
            t.slide_recover != prev.t.slide_recover ||
            std::abs(t.changed - prev.t.changed) > FRAME_LOG_DELTA;
        if (!should_update)
            return;

        file_ << std::format("{},{},FRAME,{},{}{}{}\n", stream_ms, t.track_id,
                             t.changed, int(t.is_moving), int(t.is_sliding),
                             int(t.slide_recover));
        prev.t = t;
        prev.written_at = stream_ms;
        prev.valid = true;
        maybeFlush();
    }

    void event(unsigned int track_id, const std::string &what,
               std::int64_t stream_ms, long detail = 0) {
        if (!ok_)
            return;
        const std::lock_guard<std::mutex> lock(mtx_);

        file_ << std::format("{},{},EVENT,{},{}\n", stream_ms, track_id, what,
                             detail);

        file_.flush();
    }

  private:
    void maybeFlush() {
        if (++line_cnt_ % 250 == 0)
            file_.flush();
    }

    struct TrackData {
        FrameTelemetry t{};
        std::int64_t written_at = 0;
        bool valid = false;
    };

    std::array<TrackData, COLUMN_CNT> last_{};
    std::ofstream file_;
    std::mutex mtx_;
    bool ok_ = false;
    unsigned long line_cnt_ = 0;
};

} // namespace kreda
