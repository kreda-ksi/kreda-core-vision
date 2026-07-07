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
    std::chrono::minutes duration{0}; // 0 => run until q / signal
};

// error codes
inline constexpr int EARG = 1;
inline constexpr int ERTSP = 2;
inline constexpr int ECAL = 3;

// geometry
inline constexpr unsigned int COLUMN_CNT = 2;
inline constexpr unsigned int POINTS_CNT = COLUMN_CNT * 4;

// resolutions
inline constexpr float CONTENT_WID = 1920.0f; // saved frames
inline constexpr float CONTENT_HEI = 1080.0f;
inline constexpr float MOTION_WID = 1280.0f; // detection pipeline
inline constexpr float MOTION_HEI = 720.0f;

inline constexpr int TOTAL_MOTION_AREA =
    static_cast<int>(MOTION_WID * MOTION_HEI);
inline constexpr int TOTAL_CONTENT_AREA =
    static_cast<int>(CONTENT_WID * CONTENT_HEI);

// motion path
inline constexpr int MOTION_THRESH_INTENSITY =
    30; // px intens diff to count as 'changed'
inline constexpr int MOTION_TRIGGER_PXS = static_cast<int>(
    TOTAL_MOTION_AREA * 0.001); // how many pxs must change to trigger movement
inline constexpr int SLIDE_TRIGGER_PXS = static_cast<int>(
    TOTAL_MOTION_AREA *
    0.05); // can be set low, since sliding state is gated behind column
           // detection logic, this is just threshold
inline constexpr double MOTION_REF_ALPHA =
    0.15; // FPS-dependent, EMA ref memory ~1/alpha frames

// content path
inline constexpr int CONTENT_THRESH_INTENSITY =
    MOTION_THRESH_INTENSITY; // kept as separate for potential fine-tuning
inline constexpr int STATE_CHANGE_PXS = static_cast<int>(
    TOTAL_CONTENT_AREA *
    0.015); // how many pxs must differ from the last saved board
inline constexpr int LAST_CHANCE_PXS =
    static_cast<int>(TOTAL_CONTENT_AREA * 0.0015); // slide + final flush save
inline constexpr int MAX_STROKE_COMP_AREA = static_cast<int>(
    TOTAL_CONTENT_AREA * 0.02); // body-vs-stroke ceiling for chalk masking

// grid
inline constexpr int GRID_COLS = static_cast<int>(MOTION_WID / 80.0f);
inline constexpr int GRID_ROWS = static_cast<int>(MOTION_HEI / 80.0f);
inline constexpr int GRID_CELL_ACTIVE =
    32; // INTER_AREA resize to CV8_U, 32/255 ~ 12.5%
inline constexpr int SLIDE_MIN_ACTIVE_COLS = GRID_COLS - 3;

// frame-count timing
inline constexpr int STILL_COOLDOWN =
    30; // how many frames of stillness required to capture
inline constexpr int SLIDE_COOLDOWN =
    30; // time after sliding finishes to let board settle
inline constexpr int PRE_SLIDE_BUFFER_FRAMES =
    60; // ring buffer depth for slide lookback
inline constexpr unsigned int SLIDE_LOOKBACK_FRAMES =
    7; // how far pre-slide the lookback save reaches

// stream
inline constexpr unsigned int MAX_RETRIES = 30;
inline constexpr std::chrono::seconds SNAPSHOT_INTERVAL{20};

// calibration
inline constexpr int DRIFT_MIN_INLIERS = 15;
inline constexpr double DRIFT_MAX_SCALE_DEV = 0.10; // warn past 10% zoom drift

// logging
inline constexpr int FRAME_LOG_DELTA = 1000; // materiality band for FRAME rows

} // namespace kreda
