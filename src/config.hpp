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

// i/o
inline constexpr const char *CALIB_FILE = "calibration.xml";
} // namespace kreda
