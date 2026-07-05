#pragma once

namespace kreda {
// error codes
inline constexpr int EARG = 1;
inline constexpr int ERTSP = 2;
inline constexpr int ECAL = 3;

// config
inline constexpr unsigned int COLUMN_CNT = 2;
inline constexpr unsigned int POINTS_CNT = COLUMN_CNT * 4;
inline constexpr float OUT_WID = 1280.0f;
inline constexpr float OUT_HEI = 720.0f;
inline constexpr unsigned int MAX_RETRIES = 30;

inline constexpr bool SHOW_RAW = true;

inline constexpr int MOTION_THRESH_INTENSITY =
    30; // px intens diff to count as 'changed'
inline constexpr int MOTION_TRIGGER_PXS =
    1000; // how many pxs must change to trigger movement
inline constexpr int STATE_CHANGE_PXS =
    15000; // how many pxs must differ from the last saved board
inline constexpr int STILL_COOLDOWN =
    30; // how many frames of stillness required to capture

// i/o
inline constexpr const char *CALIB_FILE = "calibration.xml";
inline constexpr const char *OUT_DIR = "staging";
} // namespace kreda
