#pragma once
#include <chrono>
#include <string>

namespace kreda {

struct RunConfig {
    std::string rtsp_url;
    std::string calib_file = "calibration.xml";
    std::string out_dir = "staging";
    std::string log_file = "run.csv";
    std::string ref_file = "calibration_ref.png";
    bool show_gui = true;
    bool show_raw = true;
    bool log_enabled = true;
    bool force_recalibrate = false;
    bool is_file = false;
    std::chrono::minutes duration{0}; // 0 => run until q
};

// error codes
inline constexpr int EARG = 1;
inline constexpr int ERTSP = 2;
inline constexpr int ECAL = 3;

// vision
inline constexpr unsigned int COLUMN_CNT = 2;
inline constexpr unsigned int POINTS_CNT = COLUMN_CNT * 4;
inline constexpr float OUT_WID = 1280.0f;
inline constexpr float OUT_HEI = 720.0f;
inline constexpr int TOTAL_AREA =
    OUT_WID * OUT_HEI; // only used internally to calc frame area% consts
inline constexpr unsigned int MAX_RETRIES = 30;

inline constexpr int MOTION_THRESH_INTENSITY =
    30; // px intens diff to count as 'changed'
inline constexpr int CONTENT_THRESH_INTENSITY =
    MOTION_THRESH_INTENSITY; // kept as separate for potential fine-tuning
inline constexpr int MOTION_TRIGGER_PXS =
    1000; // how many pxs must change to trigger movement
inline constexpr int SLIDE_TRIGGER_PXS = static_cast<int>(TOTAL_AREA * 0.2);
inline constexpr int MAX_STROKE_COMP_AREA = static_cast<int>(TOTAL_AREA * 0.02);
inline constexpr unsigned int MOTION_HIST_FRAMES = 5;
inline constexpr int STATE_CHANGE_PXS =
    15000; // how many pxs must differ from the last saved board
inline constexpr int SLIDE_SAVE_PXS =
    1500; // same as STATE_CHANGE_PXS, fired if the chalkboard was moved
inline constexpr int FINAL_SAVE_PXS =
    1500; // same as STATE_CHANGE_PXS, fired on the end frame
inline constexpr int STILL_COOLDOWN =
    30; // how many frames of stillness required to capture
inline constexpr int SLIDE_COOLDOWN =
    120; // time after sliding finishes to let board settle
inline constexpr int PRE_SLIDE_BUFFER_FRAMES = 60; // 60 frames of memory
inline constexpr unsigned int SLIDE_LOOKBACK_FRAMES = 7;

inline constexpr std::chrono::seconds SNAPSHOT_INTERVAL{20};

// calibration
inline constexpr int DRIFT_MIN_INLIERS = 15;
inline constexpr double DRIFT_MAX_SCALE_DEV = 0.10; // warn past 10% zoom drift

// logging
inline constexpr int FRAME_LOG_DELTA = 1000;

} // namespace kreda
