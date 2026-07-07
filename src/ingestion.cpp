#include "ingestion.hpp"
#include "config.hpp"
#include "debug_hud.hpp"
#include "signals.hpp"
#include "telemetry.hpp"
#include "track_log.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <mutex>
#include <thread>

namespace kreda {

struct TrackState {
    std::deque<cv::Mat> history_buff;
    std::deque<cv::Mat> motion_hist;

    cv::Mat last_saved_frame;
    std::int64_t last_save_ms = -1;
    std::int64_t last_seen_ms = -1;

    int still_cnt = 0;
    bool slide_recover = false;
    int recover_cooldown = 0;
    int save_flash = 0;
    bool was_active = false;
};

struct TimedFrame {
    cv::Mat frame;
    std::int64_t stream_ms = 0;
};

class LatestFrame {
  public:
    void push(TimedFrame &&tf) {
        {
            const std::lock_guard<std::mutex> lock(mtx_);
            cv::swap(frame_.frame, tf.frame);
            frame_.stream_ms = tf.stream_ms;
        }
        cv_.notify_one();
    }

    bool tryTake(TimedFrame &out, const std::atomic<bool> &running) {
        // NOLINTNEXTLINE(misc-const-correctness)
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(100),
                          [&] { return !frame_.frame.empty() || !running; }))
            return false; // timeout, no frame and no caller loops
        if (!running && frame_.frame.empty())
            return false;

        cv::swap(out.frame, frame_.frame);
        out.stream_ms = frame_.stream_ms;

        return true;
    }

    void wake() { cv_.notify_all(); }

  private:
    TimedFrame frame_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

bool openStream(cv::VideoCapture &cap, const std::string &url) {
    cap.release();
    cap.open(url, cv::CAP_FFMPEG);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    return cap.isOpened();
}

static cv::Mat enhanceChalkboard(const cv::Mat &raw_board,
                                 const cv::Ptr<cv::CLAHE> &clahe) {
    cv::Mat gray, enhanced, final;
    cv::cvtColor(raw_board, gray, cv::COLOR_BGR2GRAY);
    clahe->apply(gray, enhanced);
    cv::bitwise_not(enhanced, final);
    return final;
}

static int countChalkPixels(const cv::Mat &bin_mask, int max_comp_area) {
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

static bool saveIfChanged(const RunConfig &cfg, const cv::Mat &frame,
                          TrackState &state, unsigned int track_id,
                          const char *reason, int threshold,
                          std::int64_t stream_ms, TrackLogger &logger) {
    cv::Mat cdiff, cthresh;
    cv::absdiff(frame, state.last_saved_frame, cdiff);
    cv::threshold(cdiff, cthresh, CONTENT_THRESH_INTENSITY, 255,
                  cv::THRESH_BINARY);

    const int raw_pxs = cv::countNonZero(cthresh);

    const int chalk_pxs = countChalkPixels(cthresh, MAX_STROKE_COMP_AREA);

    if (chalk_pxs <= threshold) {
        logger.event(track_id, std::format("SKIP_{}_{}", reason, raw_pxs),
                     stream_ms, chalk_pxs);
        return false;
    }

    const std::string filename = std::format(
        "{}/track_{}_{}_{}.png", cfg.out_dir, track_id, stream_ms, reason);
    cv::imwrite(filename, frame);

    logger.event(track_id, std::format("SAVE_{}_{}", reason, raw_pxs),
                 stream_ms, chalk_pxs);

    state.last_save_ms = stream_ms;
    state.last_saved_frame = frame.clone();
    state.save_flash = 15;
    return true;
}

static int detectMotion(const cv::Mat &motion_frame, const cv::Mat &ref_frame,
                        cv::Mat *display_frame) {
    cv::Mat cdiff, diff, thresh;
    cv::absdiff(motion_frame, ref_frame, cdiff);
    std::vector<cv::Mat> ch(3);
    cv::split(cdiff, ch);
    diff = cv::max(ch[0], cv::max(ch[1], ch[2]));
    cv::threshold(diff, thresh, MOTION_THRESH_INTENSITY, 255,
                  cv::THRESH_BINARY);

    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

    if (display_frame)
        display_frame->setTo(cv::Scalar(0, 255, 255), thresh);

    return cv::countNonZero(thresh);
}

static bool updateSlideRecovery(TrackState &state, const cv::Mat &motion_frame,
                                unsigned int track_id, std::int64_t stream_ms,
                                TrackLogger &logger) {
    if (state.slide_recover) {
        state.recover_cooldown++;
        if (state.recover_cooldown >= SLIDE_COOLDOWN) {
            state.slide_recover = false;
            state.recover_cooldown = 0;
            logger.event(track_id, "RECOVERY_DONE", stream_ms);
        }

        state.motion_hist.push_back(motion_frame.clone());
        if (state.motion_hist.size() > MOTION_HIST_FRAMES)
            state.motion_hist.pop_front();

        return true;
    }

    return false;
}

static void updateActivityState(const RunConfig &cfg,
                                const cv::Mat &content_frame, TrackState &state,
                                unsigned int track_id, bool is_sliding,
                                bool is_moving, int changed,
                                std::int64_t stream_ms, TrackLogger &logger) {
    if (is_sliding) {
        logger.event(track_id, "SLIDE_DETECTED", stream_ms, changed);

        const size_t idx =
            state.history_buff.size() > SLIDE_LOOKBACK_FRAMES
                ? state.history_buff.size() - 1 - SLIDE_LOOKBACK_FRAMES
                : 0;
        const cv::Mat &old_frame = state.history_buff[idx];

        saveIfChanged(cfg, old_frame, state, track_id, "slide", SLIDE_SAVE_PXS,
                      stream_ms, logger);

        state.slide_recover = true;
        state.recover_cooldown = 0;
        state.still_cnt = 0;
        state.was_active = false;
    } else if (is_moving) {
        state.still_cnt = 0;
        state.was_active = true;
    } else if (state.was_active && ++state.still_cnt >= STILL_COOLDOWN) {
        saveIfChanged(cfg, content_frame, state, track_id, "still",
                      STATE_CHANGE_PXS, stream_ms, logger);
        state.still_cnt = 0;
        state.was_active = false;
    }
}

static void evaluateAndExtract(const RunConfig &cfg,
                               const cv::Mat &motion_frame,
                               const cv::Mat &content_frame, TrackState &state,
                               unsigned int track_id, cv::Mat *display_frame,
                               std::int64_t stream_ms, TrackLogger &logger) {
    state.last_seen_ms = stream_ms;
    // update ring buffer
    state.history_buff.push_back(content_frame.clone());
    if (state.history_buff.size() > PRE_SLIDE_BUFFER_FRAMES)
        state.history_buff.pop_front();

    if (state.last_saved_frame.empty()) { // inital state
        state.last_saved_frame = content_frame.clone();
        state.motion_hist.push_back(motion_frame.clone());
        state.last_save_ms = stream_ms;
        return;
    }

    if (updateSlideRecovery(state, motion_frame, track_id, stream_ms, logger))
        return;

    const int changed =
        detectMotion(motion_frame, state.motion_hist.front(), display_frame);

    const bool is_sliding = changed > SLIDE_TRIGGER_PXS;
    const bool is_moving = !is_sliding && changed > MOTION_TRIGGER_PXS;

    const FrameTelemetry t{track_id,
                           changed,
                           is_moving,
                           is_sliding,
                           state.slide_recover,
                           state.still_cnt,
                           state.recover_cooldown,
                           state.save_flash > 0 ? state.save_flash-- : 0};

    logger.frame(t, stream_ms);
    drawHud(display_frame, t);

    if (stream_ms - state.last_save_ms >
        std::chrono::milliseconds(SNAPSHOT_INTERVAL).count()) {
        saveIfChanged(cfg, content_frame, state, track_id, "periodic",
                      STATE_CHANGE_PXS, stream_ms, logger);
        state.last_save_ms = stream_ms;
    }

    updateActivityState(cfg, content_frame, state, track_id, is_sliding,
                        is_moving, changed, stream_ms, logger);

    state.motion_hist.push_back(motion_frame.clone());
    if (state.motion_hist.size() > MOTION_HIST_FRAMES)
        state.motion_hist.pop_front();
}

static void processColumn(const RunConfig &cfg, const cv::Mat &frame,
                          const cv::Mat &warp, TrackState &state,
                          unsigned int track_id,
                          const cv::Ptr<cv::CLAHE> &clahe,
                          std::int64_t stream_ms, TrackLogger &logger) {
    cv::Mat dewarped;

    // homography
    cv::warpPerspective(frame, dewarped, warp, cv::Size(OUT_WID, OUT_HEI));

    // pixel math
    const cv::Mat final = enhanceChalkboard(dewarped, clahe);

    if (!cfg.show_gui) {
        evaluateAndExtract(cfg, dewarped, final, state, track_id, nullptr,
                           stream_ms, logger);
        return;
    }

    cv::Mat display;
    cv::cvtColor(final, display, cv::COLOR_GRAY2BGR);
    evaluateAndExtract(cfg, dewarped, final, state, track_id, &display,
                       stream_ms, logger);

    // debug display
    if (cfg.show_raw)
        cv::imshow(std::format("KREDA column {} (raw)", track_id), dewarped);
    cv::imshow(std::format("KREDA column {}", track_id), display);
}

static void captureLoop(cv::VideoCapture &cap, const std::string &url,
                        LatestFrame &shared, std::atomic<bool> &is_running,
                        TrackLogger &logger) {
    cv::Mat temp_frame;
    unsigned int retry_cnt = 0;
    const auto capture_start = std::chrono::steady_clock::now();

    auto now_ms = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - capture_start)
            .count();
    };

    while (is_running.load(std::memory_order_relaxed)) {
        // temp_frame is swapped via cv::swap. post-push state is a valid Mat.
        // NOLINTNEXTLINE(bugprone-use-after-move)
        if (!cap.read(temp_frame) || temp_frame.empty()) {
            retry_cnt++;
            if (retry_cnt >= MAX_RETRIES) {
                logger.event(0, "STREAM_RECONNECT", now_ms());
                openStream(cap, url);
                retry_cnt = 0;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        retry_cnt = 0;

        shared.push({std::move(temp_frame), now_ms()});
    }
}

static void consumeLoop(const RunConfig &cfg, LatestFrame &shared,
                        const std::array<cv::Mat, COLUMN_CNT> &warp_matrices,
                        std::array<TrackState, COLUMN_CNT> &track_states,
                        const cv::Ptr<cv::CLAHE> &clahe,
                        std::atomic<bool> &is_running, TrackLogger &logger) {
    TimedFrame local;
    const auto run_start = std::chrono::steady_clock::now();

    while (is_running.load(std::memory_order_relaxed)) {
        const bool have_frame = shared.tryTake(local, is_running);

        if (have_frame)
            for (unsigned int i{}; i < COLUMN_CNT; ++i)
                processColumn(cfg, local.frame, warp_matrices[i],
                              track_states[i], i, clahe, local.stream_ms,
                              logger);

        if (cfg.show_gui) {
            if (cv::pollKey() == 'q')
                is_running = false;
        } else {
            if (g_signal_stop.load())
                is_running = false;
            if (cfg.duration.count() > 0 &&
                std::chrono::steady_clock::now() - run_start > cfg.duration)
                is_running = false;
        }
    }
}

void runIngestionLoop(const RunConfig &cfg, cv::VideoCapture &cap,
                      const std::string &rtsp_url,
                      const std::array<cv::Mat, COLUMN_CNT> &warp_matrices) {
    std::filesystem::create_directory(cfg.out_dir);

    const cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0);
    std::array<TrackState, COLUMN_CNT> track_states; // holds state for all cols

    LatestFrame shared;
    std::atomic<bool> is_running{true};

    TrackLogger logger(cfg);

    std::thread capture_thrd(
        [&] { captureLoop(cap, rtsp_url, shared, is_running, logger); });

    consumeLoop(cfg, shared, warp_matrices, track_states, clahe, is_running,
                logger);

    // end of run flush
    for (unsigned int i{}; i < COLUMN_CNT; ++i) {
        TrackState &state = track_states[i];
        if (!state.history_buff.empty() && !state.last_saved_frame.empty())
            saveIfChanged(cfg, state.history_buff.back(), state, i, "final",
                          FINAL_SAVE_PXS, state.last_seen_ms, logger);
    }

    // cleanup
    is_running = false;
    shared.wake();
    if (capture_thrd.joinable())
        capture_thrd.join();
}

} // namespace kreda
