#include "args.hpp"
#include "config.hpp"
#include "doctest.h"
#include <array>
#include <fstream>
#include <initializer_list>

using namespace kreda;

namespace {

struct Argv {
    std::vector<std::string> storage;
    std::vector<char *> ptrs;
    explicit Argv(std::initializer_list<std::string> args) {
        storage.emplace_back("kreda");
        storage.insert(storage.end(), args);
        for (auto &s : storage)
            ptrs.push_back(s.data());
    }
    int argc() const { return static_cast<int>(ptrs.size()); }
    char **argv() { return ptrs.data(); }
};

} // namespace

TEST_CASE("args: url only yields defaults") {
    Argv a({"rtsp://cam.local:8554/live"});
    auto cfg = parseArgs(a.argc(), a.argv());

    CHECK(cfg.rtsp_url == "rtsp://cam.local:8554/live");
    CHECK(cfg.show_gui);
    CHECK(cfg.log_enabled);
    CHECK_FALSE(cfg.is_file);
}

TEST_CASE("args: --headless disables both GUI flags") {
    Argv a({"--headless", "rtsp://cam.local:8554/live"});
    auto cfg = parseArgs(a.argc(), a.argv());

    CHECK_FALSE(cfg.show_gui);
    CHECK_FALSE(cfg.show_raw);
}

TEST_CASE("args: value flags parse") {
    Argv a({"-c", "x.xml", "-d", "90", "rtsp://cam.local:8554/live"});
    auto cfg = parseArgs(a.argc(), a.argv());

    CHECK(cfg.calib_file == "x.xml");
    CHECK(cfg.duration == std::chrono::minutes(90));
}

TEST_CASE("args: bare host gets rtsp scheme") {
    Argv a({"cam.local:8554/live"});
    auto cfg = parseArgs(a.argc(), a.argv());

    CHECK(cfg.rtsp_url == "rtsp://cam.local:8554/live");
    CHECK_FALSE(cfg.is_file);
}

TEST_CASE("args: flag after positional parses") {
    Argv a({"rtsp://cam.local:8554/live", "-nl"});
    auto cfg = parseArgs(a.argc(), a.argv());

    CHECK_FALSE(cfg.log_enabled);
}
