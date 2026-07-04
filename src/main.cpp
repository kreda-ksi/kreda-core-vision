#include <chrono>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

// ERROR CODES

const int KREDA_EARG = 1;
const int KREDA_ERTSP = 2;
const int KREDA_ECAL = 3;

// CONFIG

const unsigned int COLUMN_CNT = 2U;
const unsigned int POINTS_CNT = COLUMN_CNT * 4U;
const float OUT_WID = 1920.0f;
const float OUT_HEI = 1080.0f;
const unsigned int MAX_RETRIES = 30U;

void onMouseClick(int event, int x, int y, int flags, void *userdata) {
  auto *points = static_cast<std::vector<cv::Point2f> *>(userdata);

  if (event == cv::EVENT_LBUTTONDOWN) {
    if (points->size() < POINTS_CNT) {
      points->push_back(
          cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
      std::cout << std::format("Point {}/{} recorded at ({},{})",
                               points->size(), POINTS_CNT, x, y)
                << std::endl;
    } else {
      std::cout << std::format(
                       "All {} points recorded. Press any key to continue.",
                       POINTS_CNT)
                << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << std::format("Usage: ./{} <rtsp_ip> [rtsp_port]", argv[0])
              << std::endl;
    return KREDA_EARG;
  }

  std::string rtsp_url =
      std::format("rtsp://{}:{}/live", argv[1], argc > 2 ? argv[2] : "8554");

  std::cout << std::format("Connecting to: {}", rtsp_url) << std::endl;

  cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);

  if (!cap.isOpened()) {
    std::cerr << "Could not open the RTSP stream. Is mediamtx running?"
              << std::endl;
    return KREDA_ERTSP;
  }

  // set buffer size to one frame to prevent any delay buildup
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  cv::Mat frame;

  // read frame 1 for calibration
  if (!cap.read(frame) || frame.empty()) {
    std::cerr << "Failed to grab the first frame for calibration." << std::endl;
    return KREDA_ECAL;
  }

  std::vector<cv::Point2f> src_points;
  cv::namedWindow("Calibration", cv::WINDOW_NORMAL);
  cv::setMouseCallback("Calibration", onMouseClick, &src_points);

  while (true) {
    cv::Mat display_frame = frame.clone();
    for (std::size_t i{}; i < src_points.size(); ++i) {
      cv::circle(display_frame, src_points[i], 5, cv::Scalar(0, 0, 255), -1);

      auto mod_i = i % 4;
      if (mod_i > 0)
        cv::line(display_frame, src_points[i - 1], src_points[i],
                 cv::Scalar(0, 255, 0), 2);
      if (mod_i == 3)
        cv::line(display_frame, src_points[i], src_points[i - 3],
                 cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Calibration", display_frame);

    if (src_points.size() == 8) {
      cv::waitKey(1000);
      break;
    }

    cv::waitKey(30);
  }
  cv::destroyWindow("Calibration");

  std::vector<cv::Point2f> dst_points = {
      cv::Point2f(0, 0),
      cv::Point2f(OUT_WID, 0),
      cv::Point2f(OUT_WID, OUT_HEI),
      cv::Point2f(0, OUT_HEI),
  };

  std::array<cv::Mat, COLUMN_CNT> warp_matrices;

  for (unsigned int i{}; i < COLUMN_CNT; ++i) {
    std::vector<cv::Point2f> track_src(src_points.begin() + i * 4,
                                       src_points.begin() + (i + 1) * 4);
    warp_matrices[i] = cv::getPerspectiveTransform(track_src, dst_points);
  }

  std::array<cv::Mat, COLUMN_CNT> dewarped_boards;
  int retry_cnt = 0;

  while (true) {
    bool succ = cap.read(frame);

    if (!succ || frame.empty()) {
      retry_cnt++;
      std::cerr << std::format(
                       "Frame dropped or stream stuttered. Retry {}/{}.",
                       retry_cnt, MAX_RETRIES)
                << std::endl;

      if (retry_cnt >= MAX_RETRIES) {
        std::cerr << "Stream lost completely. Reconnecting." << std::endl;

        // reboot the connection
        cap.release();
        cap.open(rtsp_url, cv::CAP_FFMPEG);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        retry_cnt = 0;

        // wait a second for it to wake up
        cv::waitKey(1000);
        continue;
      }

      if (cv::waitKey(33) == 'q')
        break;

      continue;
    }

    retry_cnt = 0;

    for (unsigned int i{}; i < COLUMN_CNT; ++i) {
      cv::warpPerspective(frame, dewarped_boards[i], warp_matrices[i],
                          cv::Size(OUT_WID, OUT_HEI));
      cv::imshow(std::format("KREDA column {}", i + 1), dewarped_boards[i]);
    }

    if (cv::waitKey(1) == 'q')
      break;
  }

  // cleanup
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
