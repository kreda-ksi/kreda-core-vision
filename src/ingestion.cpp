#include "ingestion.hpp"
#include "config.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <mutex>
#include <thread>

namespace kreda {

struct TrackState {
    cv::Mat prev_frame;
    cv::Mat last_saved_frame;
    int still_cnt = 0;
    bool is_moving = false;
};

static cv::Mat enhanceChalkboard(const cv::Mat &raw_board,
                                 const cv::Ptr<cv::CLAHE> &clahe) {
    cv::Mat gray, enhanced, final;
    cv::cvtColor(raw_board, gray, cv::COLOR_BGR2GRAY);
    clahe->apply(gray, enhanced);
    cv::bitwise_not(enhanced, final);
    return final;
}

// evaluates if a frame should be saved based on verification.
// the verification process is as follows:
// 1. on image difference (motion detection) track goes to an 'alert' state,
// 2. if after the 'alert' state cools down there's a visible difference on the
// chalkboard, then proceed to save the frame. otherwise, just continue.
static void evaluateAndExtract(const cv::Mat &curr_frame, TrackState &state,
                               unsigned int track_id) {
    if (state.prev_frame.empty()) { // inital state
        state.prev_frame = curr_frame.clone();
        state.last_saved_frame = curr_frame.clone();
        return;
    }

    cv::Mat diff1, thresh1;
    cv::absdiff(curr_frame, state.prev_frame, diff1);
    cv::threshold(diff1, thresh1, MOTION_THRESH_INTENSITY, 255,
                  cv::THRESH_BINARY);

    if (cv::countNonZero(thresh1) > MOTION_TRIGGER_PXS) {
        state.is_moving = true;
        state.still_cnt = 0;
    } else if (state.is_moving) {
        state.still_cnt++;

        if (state.still_cnt >= STILL_COOLDOWN) {
            cv::Mat diff2, thresh2;
            cv::absdiff(curr_frame, state.last_saved_frame, diff2);
            cv::threshold(diff2, thresh2, MOTION_THRESH_INTENSITY, 255,
                          cv::THRESH_BINARY);

            if (cv::countNonZero(thresh2) > STATE_CHANGE_PXS) {
                auto timestamp =
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

                std::string filename = std::format(
                    "{}/track_{}_{}.png", OUT_DIR, track_id, timestamp);
                cv::imwrite(filename, curr_frame);

                std::cout << std::format("Track {} updated. Saved to {}.",
                                         track_id, filename)
                          << std::endl;

                state.last_saved_frame = curr_frame.clone();
            }

            state.is_moving = false;
            state.still_cnt = 0;
        }
    }

    state.prev_frame = curr_frame.clone();
}

void runIngestionLoop(cv::VideoCapture &cap, const std::string &rtsp_url,
                      const std::array<cv::Mat, COLUMN_CNT> &warp_matrices) {
    std::filesystem::create_directory(OUT_DIR);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0);
    std::array<TrackState, COLUMN_CNT> track_states; // holds state for all cols

    std::mutex frame_mtx;
    cv::Mat shared_frame;
    std::atomic<bool> is_running{true};

    // network i/o
    std::thread capture_thrd([&]() {
        cv::Mat temp_frame;
        unsigned int retry_cnt = 0;

        while (is_running) {
            if (!cap.read(temp_frame) || temp_frame.empty()) {
                retry_cnt++;
                if (retry_cnt >= MAX_RETRIES) {
                    std::cerr << "Stream lost completely. Reconnecting."
                              << std::endl;

                    // reboot the connection
                    cap.release();
                    cap.open(rtsp_url, cv::CAP_FFMPEG);
                    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

                    retry_cnt = 0;

                    // wait a second for it to wake up
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            retry_cnt = 0;

            // lock long enough to drop the new frame in
            {
                std::lock_guard<std::mutex> lock(frame_mtx);
                shared_frame = temp_frame.clone();
            }
        }
    });

    cv::Mat local_frame;

    // cpu math and disk i/o
    while (true) {
        bool have_frame = false;
        {
            std::lock_guard<std::mutex> lock(frame_mtx);
            if (!shared_frame.empty()) {
                local_frame = shared_frame.clone();
                shared_frame.release(); // to not process the same frame twice
                have_frame = true;
            }
        }
        if (!have_frame) {
            // yield cpu if no frame is ready yet
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        for (unsigned int i{}; i < COLUMN_CNT; ++i) {
            cv::Mat dewarped;

            // homography
            cv::warpPerspective(local_frame, dewarped, warp_matrices[i],
                                cv::Size(OUT_WID, OUT_HEI));

            // pixel math
            cv::Mat final = enhanceChalkboard(dewarped, clahe);
            evaluateAndExtract(final, track_states[i], i);

            // debug display
            if (SHOW_RAW)
                cv::imshow(std::format("KREDA column {} (raw)", i + 1),
                           dewarped);
            cv::imshow(std::format("KREDA column {}", i + 1), final);
        }

        cv::pollKey();
    }

    // cleanup
    if (capture_thrd.joinable())
        capture_thrd.join();
}

} // namespace kreda
