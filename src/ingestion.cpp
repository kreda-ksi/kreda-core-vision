#include "ingestion.hpp"
#include "config.hpp"
#include "debug_hud.hpp"
#include "telemetry.hpp"
#include "track_log.hpp"
#include <atomic>
#include <chrono>
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

    std::chrono::steady_clock::time_point last_save_time =
        std::chrono::steady_clock::now();

    int still_cnt = 0;
    bool slide_recover = false;
    int recover_cooldown = 0;
    int save_flash = 0;
    bool was_active = false;
};

class LatestFrame {
  public:
    void push(cv::Mat &&f) {
        std::lock_guard<std::mutex> lock(mtx_);
        frame_ = std::move(f);
    }

    bool tryTake(cv::Mat &out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (frame_.empty())
            return false;

        out = std::move(frame_);
        return true;
    }

  private:
    cv::Mat frame_;
    std::mutex mtx_;
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

static bool saveIfChanged(const cv::Mat &frame, TrackState &state,
                          unsigned int track_id, const char *reason) {
    cv::Mat cdiff, cthresh;
    cv::absdiff(frame, state.last_saved_frame, cdiff);
    cv::threshold(cdiff, cthresh, CONTENT_THRESH_INTENSITY, 255,
                  cv::THRESH_BINARY);

    int raw_pxs = cv::countNonZero(cthresh);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(cthresh, labels, stats, centroids);
    int chalk_pxs = 0;
    for (int i = 1; i < n; ++i) { // 0 is bg
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a < MAX_STROKE_COMP_AREA)
            chalk_pxs += a;
    }
    if (chalk_pxs <= STATE_CHANGE_PXS) {
        if (LOG_ENABLED)
            TrackLogger::instance().event(
                track_id, std::format("SKIP_{}_{}", reason, raw_pxs),
                chalk_pxs);
        return false;
    }

    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

    std::string filename =
        std::format("{}/track_{}_{}.png", OUT_DIR, track_id, timestamp);
    cv::imwrite(filename, frame);

    if (LOG_ENABLED)
        TrackLogger::instance().event(
            track_id, std::format("SAVE_{}_{}", reason, raw_pxs), chalk_pxs);

    state.last_save_time = std::chrono::steady_clock::now();
    state.last_saved_frame = frame.clone();
    state.save_flash = 15;
    return true;
}

static int detectMotion(const cv::Mat &motion_frame, const cv::Mat &ref_frame,
                        cv::Mat &display_frame) {
    cv::Mat cdiff, diff, thresh;
    cv::absdiff(motion_frame, ref_frame, cdiff);
    std::vector<cv::Mat> ch(3);
    cv::split(cdiff, ch);
    diff = cv::max(ch[0], cv::max(ch[1], ch[2]));
    cv::threshold(diff, thresh, MOTION_THRESH_INTENSITY, 255,
                  cv::THRESH_BINARY);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

    display_frame.setTo(cv::Scalar(0, 255, 255), thresh);

    return cv::countNonZero(thresh);
}

static bool updateSlideRecovery(TrackState &state, const cv::Mat &motion_frame,
                                unsigned int track_id) {
    if (state.slide_recover) {
        state.recover_cooldown++;
        if (state.recover_cooldown >= SLIDE_COOLDOWN) {
            state.slide_recover = false;
            state.recover_cooldown = 0;
            if (LOG_ENABLED)
                TrackLogger::instance().event(track_id, "RECOVERY_DONE");
        }

        state.motion_hist.push_back(motion_frame.clone());
        if (state.motion_hist.size() > MOTION_HIST_FRAMES)
            state.motion_hist.pop_front();

        return true;
    }

    return false;
}

static void updateActivityState(const cv::Mat &content_frame, TrackState &state,
                                unsigned int track_id, bool is_sliding,
                                bool is_moving, int changed) {
    if (is_sliding) {
        if (LOG_ENABLED)
            TrackLogger::instance().event(track_id, "SLIDE_DETECTED", changed);

        size_t idx = state.history_buff.size() > SLIDE_LOOKBACK_FRAMES
                         ? state.history_buff.size() - 1 - SLIDE_LOOKBACK_FRAMES
                         : 0;
        const cv::Mat &old_frame = state.history_buff[idx];

        saveIfChanged(old_frame, state, track_id, "slide");

        state.slide_recover = true;
        state.recover_cooldown = 0;
        state.still_cnt = 0;
        state.was_active = false;
    } else if (is_moving) {
        state.still_cnt = 0;
        state.was_active = true;
    } else if (state.was_active && ++state.still_cnt >= STILL_COOLDOWN) {
        saveIfChanged(content_frame, state, track_id, "still");
        state.still_cnt = 0;
        state.was_active = false;
    }
}

static void evaluateAndExtract(const cv::Mat &motion_frame,
                               const cv::Mat &content_frame, TrackState &state,
                               unsigned int track_id, cv::Mat &display_frame) {
    // update ring buffer
    state.history_buff.push_back(content_frame.clone());
    if (state.history_buff.size() > PRE_SLIDE_BUFFER_FRAMES)
        state.history_buff.pop_front();

    if (state.last_saved_frame.empty()) { // inital state
        state.last_saved_frame = content_frame.clone();
        state.motion_hist.push_back(motion_frame.clone());
        return;
    }

    if (updateSlideRecovery(state, motion_frame, track_id))
        return;

    int changed =
        detectMotion(motion_frame, state.motion_hist.front(), display_frame);

    bool is_sliding = changed > SLIDE_TRIGGER_PXS;
    bool is_moving = !is_sliding && changed > MOTION_TRIGGER_PXS;

    FrameTelemetry t{track_id,
                     changed,
                     is_moving,
                     is_sliding,
                     state.slide_recover,
                     state.still_cnt,
                     state.recover_cooldown,
                     state.save_flash--};

    if (LOG_ENABLED)
        TrackLogger::instance().frame(t);
    if (SHOW_GUI)
        drawHud(display_frame, t);

    auto since_save = std::chrono::steady_clock::now() - state.last_save_time;
    if (since_save > SNAPSHOT_INTERVAL) {
        saveIfChanged(content_frame, state, track_id, "periodic");
        state.last_save_time = std::chrono::steady_clock::now();
    }

    updateActivityState(content_frame, state, track_id, is_sliding, is_moving,
                        changed);

    state.motion_hist.push_back(motion_frame.clone());
    if (state.motion_hist.size() > MOTION_HIST_FRAMES)
        state.motion_hist.pop_front();
}

static void processColumn(const cv::Mat &frame, const cv::Mat &warp,
                          TrackState &state, unsigned int track_id,
                          const cv::Ptr<cv::CLAHE> &clahe) {
    cv::Mat dewarped;

    // homography
    cv::warpPerspective(frame, dewarped, warp, cv::Size(OUT_WID, OUT_HEI));

    // pixel math
    cv::Mat final = enhanceChalkboard(dewarped, clahe);

    cv::Mat display;
    cv::cvtColor(final, display, cv::COLOR_GRAY2BGR);

    evaluateAndExtract(dewarped, final, state, track_id, display);

    // debug display
    if (SHOW_RAW)
        cv::imshow(std::format("KREDA column {} (raw)", track_id), dewarped);
    cv::imshow(std::format("KREDA column {}", track_id), display);
}

static void captureLoop(cv::VideoCapture &cap, const std::string &url,
                        LatestFrame &shared, std::atomic<bool> &is_running) {
    cv::Mat temp_frame;
    unsigned int retry_cnt = 0;

    while (is_running.load(std::memory_order_relaxed)) {
        if (!cap.read(temp_frame) || temp_frame.empty()) {
            retry_cnt++;
            if (retry_cnt >= MAX_RETRIES) {
                if (LOG_ENABLED)
                    TrackLogger::instance().event(0, "STREAM_RECONNECT", 0);
                openStream(cap, url);
                retry_cnt = 0;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        retry_cnt = 0;
        shared.push(std::move(temp_frame));
    }
}

static void consumeLoop(LatestFrame &shared,
                        const std::array<cv::Mat, COLUMN_CNT> &warp_matrices,
                        std::array<TrackState, COLUMN_CNT> &track_states,
                        const cv::Ptr<cv::CLAHE> &clahe,
                        std::atomic<bool> &is_running) {
    cv::Mat local_frame;

    while (is_running.load(std::memory_order_relaxed)) {
        if (!shared.tryTake(local_frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        for (unsigned int i{}; i < COLUMN_CNT; ++i)
            processColumn(local_frame, warp_matrices[i], track_states[i], i,
                          clahe);

        if (cv::pollKey() == 'q')
            is_running = false;
    }
}

void runIngestionLoop(cv::VideoCapture &cap, const std::string &rtsp_url,
                      const std::array<cv::Mat, COLUMN_CNT> &warp_matrices) {
    std::filesystem::create_directory(OUT_DIR);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0);
    std::array<TrackState, COLUMN_CNT> track_states; // holds state for all cols

    LatestFrame shared;
    std::atomic<bool> is_running{true};

    std::thread capture_thrd(captureLoop, std::ref(cap), std::cref(rtsp_url),
                             std::ref(shared), std::ref(is_running));

    consumeLoop(shared, warp_matrices, track_states, clahe, is_running);

    // cleanup
    if (capture_thrd.joinable())
        capture_thrd.join();
}

} // namespace kreda
