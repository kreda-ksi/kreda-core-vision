#include "calibration.hpp"
#include "config.hpp"
#include "ingestion.hpp"
#include "signals.hpp"
#include <csignal>
#include <filesystem>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace kreda;

static inline void printUsage(const char *prog) {
    std::cerr << std::format(
        "Usage: {} [OPTIONS] <rtsp_url>\n"
        "  --headless           no windows (requires existing calibration)\n"
        "  --no-raw             suppress raw debug windows\n"
        "  --no-log             disable CSV telemetry\n"
        "  --calib <path>       calibration file           (default: "
        "calibration.xml)\n"
        "  --out <path>         output directory           (default: staging)\n"
        "  --log-file <path>    CSV path                   (default: "
        "kreda_run.csv)\n"
        "  --ref-file <path>    calibration reference file (default: "
        "calibration_ref.png)\n"
        "  --recalibrate        force manual calibration\n"
        "  --duration <min>     auto-stop after min minutes\n",
        prog);
}

RunConfig parseArgs(int argc, char **argv) {
    RunConfig cfg;

    auto next_val = [&](int &i, std::string_view flag) -> const char * {
        if (i + 1 >= argc) {
            std::cerr << std::format("{} requires a value.\n", flag);
            printUsage(argv[0]);
            exit(EARG);
        }

        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "--headless" || arg == "-h") {
            cfg.show_gui = false;
            cfg.show_raw = false;
        } else if (arg == "--no-raw" || arg == "-nr") {
            cfg.show_raw = false;
        } else if (arg == "--no-log" || arg == "-nl") {
            cfg.log_enabled = false;
        } else if (arg == "--recalibrate" || arg == "-rc") {
            cfg.force_recalibrate = true;
        } else if (arg == "--calib" || arg == "-c") {
            cfg.calib_file = next_val(i, arg);
        } else if (arg == "--out" || arg == "-o") {
            cfg.out_dir = next_val(i, arg);
        } else if (arg == "--log-file" || arg == "-lf") {
            cfg.log_file = next_val(i, arg);
        } else if (arg == "--duration" || arg == "-d") {
            const long mins = std::strtol(next_val(i, arg), nullptr, 10);
            if (mins <= 0) {
                std::cerr << "--duration must be a positive integer.\n";
                exit(EARG);
            }
            cfg.duration = std::chrono::minutes(mins);
        } else if (arg.starts_with("--")) {
            std::cerr << std::format("Unknown option: {}\n", arg);
            printUsage(argv[0]);
            exit(EARG);
        } else if (cfg.rtsp_url.empty()) {
            cfg.rtsp_url = arg;
        } else {
            std::cerr << std::format("Unexpected argument: {}\n", arg);
            printUsage(argv[0]);
            exit(EARG);
        }
    }

    if (std::filesystem::exists(cfg.rtsp_url)) {
        cfg.is_file = true;
    } else if (cfg.rtsp_url.find("://") == std::string::npos) {
        cfg.rtsp_url = "rtsp://" + cfg.rtsp_url;
    }

    if (cfg.rtsp_url.empty()) {
        std::cerr << "Missing rtsp_url.\n";
        printUsage(argv[0]);
        exit(EARG);
    }

    return cfg;
}

int main(int argc, char **argv) {
    auto cfg = parseArgs(argc, argv);
    installSigHandlers();

    std::cout << std::format("Connecting to: {}\n", cfg.rtsp_url);

    cv::VideoCapture cap;
    if (!openStream(cap, cfg.rtsp_url, cfg.is_file)) {
        std::cerr << "Could not open the RTSP stream. Is mediamtx running?\n";
        return ERTSP;
    }

    const WarpSet warps = runCalibration(cfg, cap);
    runIngestionLoop(cfg, cap, cfg.rtsp_url, warps);

    // cleanup
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
