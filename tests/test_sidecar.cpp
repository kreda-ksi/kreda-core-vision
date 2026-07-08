#include "config.hpp"
#include "doctest.h"
#include "track_log.hpp"
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <sstream>

using namespace kreda;

namespace {

std::string readAll(const std::filesystem::path &p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool balanced(const std::string &s) {
    int brace = 0;
    int bracket = 0;
    for (char c : s) {
        brace += (c == '{') - (c == '}');
        bracket += (c == '[') - (c == ']');
        if (brace < 0 || bracket < 0)
            return false;
    }
    return brace == 0 && bracket == 0;
}

RunConfig tmpCfg(bool enabled) {
    RunConfig cfg;
    cfg.log_enabled = enabled;
    cfg.grid_file =
        (std::filesystem::temp_directory_path() / "kreda_sidecar_test.json")
            .string();
    return cfg;
}

} // namespace

TEST_CASE("sidecar: lifecycle produces balanced JSON with all entries") {
    const auto cfg = tmpCfg(true);

    {
        SidecarLogger sc(cfg);
        cv::Mat grid(GRID_ROWS, GRID_COLS, CV_8U, cv::Scalar(128));
        sc.logSave("staging/a.png", 1000, "SAVE_periodic", grid);
        sc.logSave("staging/b.png", 2000, "SAVE_still", grid);
    } // destructor closes the JSON

    const std::string s = readAll(cfg.grid_file);
    CHECK(balanced(s));
    CHECK(s.find("staging/a.png") != std::string::npos);
    CHECK(s.find("staging/b.png") != std::string::npos);
    std::filesystem::remove(cfg.grid_file);
}

TEST_CASE("sidecar: comma discipline between entries") {
    const auto cfg = tmpCfg(true);

    {
        SidecarLogger sc(cfg);
        cv::Mat grid = cv::Mat::zeros(GRID_ROWS, GRID_COLS, CV_8U);
        sc.logSave("one.png", 1, "SAVE_still", grid);
    }

    const std::string s = readAll(cfg.grid_file);
    CHECK(s.find(",\n\n") == std::string::npos); // no dangling separator
    CHECK(balanced(s));
    std::filesystem::remove(cfg.grid_file);
}

TEST_CASE("sidecar: disabled logger writes nothing") {
    const auto cfg = tmpCfg(false);
    std::filesystem::remove(cfg.grid_file);

    {
        SidecarLogger sc(cfg);
        cv::Mat grid = cv::Mat::zeros(GRID_ROWS, GRID_COLS, CV_8U);
        sc.logSave("x.png", 1, "SAVE_still", grid);
    }

    CHECK_FALSE(std::filesystem::exists(cfg.grid_file));
}
