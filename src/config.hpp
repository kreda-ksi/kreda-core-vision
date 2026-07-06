#pragma once
#include <chrono>

namespace kreda {
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

inline constexpr bool SHOW_RAW = false;
inline constexpr bool LOG_ENABLED = true;
inline constexpr bool SHOW_GUI = true;

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
inline constexpr int STILL_COOLDOWN =
    30; // how many frames of stillness required to capture
inline constexpr int SLIDE_COOLDOWN =
    120; // time after sliding finishes to let board settle
inline constexpr int PRE_SLIDE_BUFFER_FRAMES = 60; // 60 frames of memory
inline constexpr unsigned int SLIDE_LOOKBACK_FRAMES = 7;

inline constexpr std::chrono::seconds SNAPSHOT_INTERVAL{20};

// i/o
inline constexpr const char *CALIB_FILE = "calibration.xml";
inline constexpr const char *OUT_DIR = "staging";
inline constexpr const char *LOG_FILE = "run.csv";
} // namespace kreda
