#include "ingestion.hpp"
#include "config.hpp"
#include "debug_hud.hpp"
#include "frames.hpp"
#include "signals.hpp"
#include "telemetry.hpp"
#include "track_log.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <format>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace kreda {

struct TrackState {
    std::deque<cv::Mat> history_buff;
    cv::Mat motion_ref_f32;
    cv::Mat grid_decayed_f32;

    cv::Mat last_saved_frame;
    std::int64_t last_save_ms = -1;
    std::int64_t last_seen_ms = -1;

    int still_cnt = 0;
    bool slide_recover = false;
    int recover_cooldown = 0;
    int save_flash = 0;
    bool was_active = false;
};

bool openStream(cv::VideoCapture &cap, const std::string &url, bool is_file) {
    cap.release();
    cap.open(url, cv::CAP_FFMPEG);
    if (!is_file)
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
                          std::int64_t stream_ms, TrackLogger &logger,
                          SidecarLogger &sidecar) {
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

    if (!state.grid_decayed_f32.empty()) {
        cv::Mat grid_8u;
        state.grid_decayed_f32.convertTo(grid_8u, CV_8U);
        sidecar.logSave(filename, stream_ms, std::format("SAVE_{}", reason),
                        grid_8u);
    }

    state.last_save_ms = stream_ms;
    state.last_saved_frame = frame.clone();
    state.save_flash = 15;
    return true;
}

static cv::Mat gridOccupancy(const cv::Mat &bin_mask) {
    cv::Mat grid;
    cv::resize(bin_mask, grid, cv::Size(GRID_COLS, GRID_ROWS), 0, 0,
               cv::INTER_AREA);
    return grid;
}

static int countActiveColumns(const cv::Mat &grid) {
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

static int detectMotion(const cv::Mat &motion_frame, cv::Mat &ref_f32,
                        cv::Mat *display_frame, cv::Mat &grid_out) {
    cv::Mat ref_8u;
    ref_f32.convertTo(ref_8u, CV_8UC3);

    cv::Mat cdiff, diff, thresh;
    cv::absdiff(motion_frame, ref_8u, cdiff);
    std::vector<cv::Mat> ch(3);
    cv::split(cdiff, ch);
    diff = cv::max(ch[0], cv::max(ch[1], ch[2]));
    cv::threshold(diff, thresh, MOTION_THRESH_INTENSITY, 255,
                  cv::THRESH_BINARY);

    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

    grid_out = gridOccupancy(thresh);

    if (display_frame) {
        cv::Mat thresh_up;
        cv::resize(thresh, thresh_up, display_frame->size(), 0, 0,
                   cv::INTER_NEAREST);
        display_frame->setTo(cv::Scalar(0, 255, 255), thresh_up);
    }

    cv::accumulateWeighted(motion_frame, ref_f32, MOTION_REF_ALPHA);

    return cv::countNonZero(thresh);
}

static bool updateSlideRecovery(TrackState &state, const cv::Mat &motion_frame,
                                unsigned int track_id, std::int64_t stream_ms,
                                TrackLogger &logger) {
    if (state.slide_recover) {
        if (!state.grid_decayed_f32.empty())
            state.grid_decayed_f32 *= GRID_DECAY;

        state.recover_cooldown++;
        if (state.recover_cooldown >= SLIDE_COOLDOWN) {
            state.slide_recover = false;
            state.recover_cooldown = 0;
            motion_frame.convertTo(state.motion_ref_f32, CV_32FC3);
            logger.event(track_id, "RECOVERY_DONE", stream_ms);
        } else {
            cv::accumulateWeighted(motion_frame, state.motion_ref_f32,
                                   MOTION_REF_ALPHA);
        }

        return true;
    }

    return false;
}

static void updateActivityState(const RunConfig &cfg,
                                const cv::Mat &content_frame, TrackState &state,
                                unsigned int track_id, bool is_sliding,
                                bool is_moving, int changed,
                                std::int64_t stream_ms, TrackLogger &logger,
                                SidecarLogger &sidecar) {
    if (is_sliding) {
        logger.event(track_id, "SLIDE_DETECTED", stream_ms, changed);

        const size_t idx =
            state.history_buff.size() > SLIDE_LOOKBACK_FRAMES
                ? state.history_buff.size() - 1 - SLIDE_LOOKBACK_FRAMES
                : 0;
        const cv::Mat &old_frame = state.history_buff[idx];

        saveIfChanged(cfg, old_frame, state, track_id, "slide", LAST_CHANCE_PXS,
                      stream_ms, logger, sidecar);

        state.slide_recover = true;
        state.recover_cooldown = 0;
        state.still_cnt = 0;
        state.was_active = false;
    } else if (is_moving) {
        state.still_cnt = 0;
        state.was_active = true;
    } else if (state.was_active && ++state.still_cnt >= STILL_COOLDOWN) {
        saveIfChanged(cfg, content_frame, state, track_id, "still",
                      STATE_CHANGE_PXS, stream_ms, logger, sidecar);
        state.still_cnt = 0;
        state.was_active = false;
    }
}

static void evaluateAndExtract(const RunConfig &cfg,
                               const cv::Mat &motion_frame,
                               const cv::Mat &content_frame, TrackState &state,
                               unsigned int track_id, cv::Mat *display_frame,
                               std::int64_t stream_ms, TrackLogger &logger,
                               SidecarLogger &sidecar) {
    state.last_seen_ms = stream_ms;
    // update ring buffer
    state.history_buff.push_back(content_frame.clone());
    if (state.history_buff.size() > PRE_SLIDE_BUFFER_FRAMES)
        state.history_buff.pop_front();

    if (state.last_saved_frame.empty()) { // inital state
        state.last_saved_frame = content_frame.clone();
        motion_frame.convertTo(state.motion_ref_f32, CV_32FC3);
        state.last_save_ms = stream_ms;
        return;
    }

    if (updateSlideRecovery(state, motion_frame, track_id, stream_ms, logger))
        return;

    cv::Mat grid;
    const int changed =
        detectMotion(motion_frame, state.motion_ref_f32, display_frame, grid);

    if (state.grid_decayed_f32.empty()) {
        grid.convertTo(state.grid_decayed_f32, CV_32F);
    } else {
        state.grid_decayed_f32 *= GRID_DECAY;
        cv::Mat grid_f32;
        grid.convertTo(grid_f32, CV_32F);
        state.grid_decayed_f32 = cv::max(state.grid_decayed_f32, grid_f32);
    }

    const bool is_sliding = changed > SLIDE_TRIGGER_PXS &&
                            countActiveColumns(grid) >= SLIDE_MIN_ACTIVE_COLS;
    const bool is_moving =
        !is_sliding && (state.was_active ? changed > MOTION_TRIGGER_PXS / 2
                                         : changed > MOTION_TRIGGER_PXS);

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
                      STATE_CHANGE_PXS, stream_ms, logger, sidecar);
        state.last_save_ms = stream_ms;
    }

    updateActivityState(cfg, content_frame, state, track_id, is_sliding,
                        is_moving, changed, stream_ms, logger, sidecar);
}

static void processColumn(const RunConfig &cfg, const cv::Mat &frame,
                          const cv::Mat &content_warp,
                          const cv::Mat &motion_warp, TrackState &state,
                          unsigned int track_id,
                          const cv::Ptr<cv::CLAHE> &clahe,
                          std::int64_t stream_ms, TrackLogger &logger,
                          SidecarLogger &sidecar) {
    cv::Mat content_raw, motion_raw;

    // homography
    cv::warpPerspective(
        frame, content_raw, content_warp,
        cv::Size(static_cast<int>(CONTENT_WID), static_cast<int>(CONTENT_HEI)),
        cv::INTER_CUBIC);
    cv::warpPerspective(
        frame, motion_raw, motion_warp,
        cv::Size(static_cast<int>(MOTION_WID), static_cast<int>(MOTION_HEI)));

    // pixel math
    const cv::Mat final = enhanceChalkboard(content_raw, clahe);

    if (!cfg.show_gui) {
        evaluateAndExtract(cfg, motion_raw, final, state, track_id, nullptr,
                           stream_ms, logger, sidecar);
        return;
    }

    cv::Mat display;
    cv::cvtColor(final, display, cv::COLOR_GRAY2BGR);
    evaluateAndExtract(cfg, motion_raw, final, state, track_id, &display,
                       stream_ms, logger, sidecar);

    // debug display
    if (cfg.show_raw)
        cv::imshow(std::format("KREDA column {} (raw)", track_id), motion_raw);
    cv::imshow(std::format("KREDA column {}", track_id), display);
}

static void captureLoop(const RunConfig &cfg, cv::VideoCapture &cap,
                        const std::string &url, LatestFrame &shared,
                        std::atomic<bool> &is_running, TrackLogger &logger) {
    cv::Mat temp_frame;
    unsigned int retry_cnt = 0;
    const auto capture_start = std::chrono::steady_clock::now();

    auto now_ms = [&] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - capture_start)
            .count();
    };

    while (is_running.load(std::memory_order_relaxed)) {
        if (!cap.read(temp_frame) || temp_frame.empty()) {
            if (cfg.is_file) {
                logger.event(0, "FILE_EOF", now_ms());
                is_running = false;
                break;
            }

            retry_cnt++;
            if (retry_cnt >= MAX_RETRIES) {
                logger.event(0, "STREAM_RECONNECT", now_ms());
                openStream(cap, url, false);
                retry_cnt = 0;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        retry_cnt = 0;

        const std::int64_t stream_ms =
            cfg.is_file
                ? static_cast<std::int64_t>(cap.get(cv::CAP_PROP_POS_MSEC))
                : now_ms();
        shared.push(temp_frame, stream_ms, cfg.is_file, is_running);
    }
}

static void consumeLoop(const RunConfig &cfg, LatestFrame &shared,
                        const WarpSet &warps,
                        std::array<TrackState, COLUMN_CNT> &track_states,
                        const cv::Ptr<cv::CLAHE> &clahe,
                        std::atomic<bool> &is_running, TrackLogger &logger,
                        SidecarLogger &sidecar) {
    TimedFrame local;
    const auto run_start = std::chrono::steady_clock::now();

    while (is_running.load(std::memory_order_relaxed)) {
        const bool have_frame = shared.tryTake(local, is_running);

        if (have_frame)
            for (unsigned int i{}; i < COLUMN_CNT; ++i)
                processColumn(cfg, local.frame, warps.content[i],
                              warps.motion[i], track_states[i], i, clahe,
                              local.stream_ms, logger, sidecar);

        if (cfg.show_gui && cv::pollKey() == 'q')
            is_running = false;
        if (g_signal_stop.load())
            is_running = false;
        if (cfg.duration.count() > 0 &&
            std::chrono::steady_clock::now() - run_start > cfg.duration)
            is_running = false;
    }
}

void runIngestionLoop(const RunConfig &cfg, cv::VideoCapture &cap,
                      const std::string &rtsp_url, const WarpSet &warps) {
    std::filesystem::create_directory(cfg.out_dir);

    const cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0);
    std::array<TrackState, COLUMN_CNT> track_states; // holds state for all cols

    LatestFrame shared;
    std::atomic<bool> is_running{true};

    TrackLogger logger(cfg);
    SidecarLogger sidecar(cfg);

    std::thread capture_thrd(
        [&] { captureLoop(cfg, cap, rtsp_url, shared, is_running, logger); });

    consumeLoop(cfg, shared, warps, track_states, clahe, is_running, logger,
                sidecar);

    is_running = false;
    shared.wake();

    // end of run flush
    for (unsigned int i{}; i < COLUMN_CNT; ++i) {
        TrackState &state = track_states[i];
        if (!state.history_buff.empty() && !state.last_saved_frame.empty())
            saveIfChanged(cfg, state.history_buff.back(), state, i, "final",
                          LAST_CHANCE_PXS, state.last_seen_ms, logger, sidecar);
    }

    // cleanup
    if (capture_thrd.joinable())
        capture_thrd.join();
}

} // namespace kreda
