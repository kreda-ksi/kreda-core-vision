#pragma once
#include <atomic>
#include <csignal>

namespace kreda {

inline std::atomic<bool> g_signal_stop{false};

inline void installSigHandlers() {
    std::signal(SIGINT, +[](int) { g_signal_stop.store(true); });
    std::signal(SIGTERM, +[](int) { g_signal_stop.store(true); });
}

} // namespace kreda
