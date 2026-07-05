#pragma once

namespace kreda {

struct FrameTelemetry {
    unsigned int track_id;
    int changed;
    bool is_moving;
    bool is_sliding;
    bool slide_recover;
    int still_cnt;
    int recover_cooldown;
    int save_flash;
};

} // namespace kreda
