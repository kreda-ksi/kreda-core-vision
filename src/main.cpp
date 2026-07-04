#include <chrono>
#include <format>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

const int KREDA_EARG = 1;
const int KREDA_ERTSP = 2;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << std::format("Usage: ./{} <rtsp_ip> [rtsp_port]", argv[0])
              << std::endl;
    return KREDA_EARG;
  }

  std::string rtsp_url =
      std::format("rtsp://{}:{}/live", argv[1], argc > 2 ? argv[2] : "8554");

  std::cout << "Connecting to: " << rtsp_url << std::endl;

  cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);

  if (!cap.isOpened()) {
    std::cerr << "Could not open the RTSP stream. Is mediamtx running?"
              << std::endl;
    return KREDA_ERTSP;
  }

  // set buffer size to one frame to prevent any delay buildup
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  cv::Mat frame;
  int retry_cnt = 0;
  const int MAX_RETRIES = 30;

  while (true) {
    bool succ = cap.read(frame);

    if (!succ || frame.empty()) {
      retry_cnt++;
      std::cerr << "Frame dropped or stream stuttered. Retry " << retry_cnt
                << "/" << MAX_RETRIES << std::endl;

      if (retry_cnt >= MAX_RETRIES) {
        std::cerr << "Stream lost completely. Exiting." << std::endl;
        break;
      }

      // 33ms ~ 30fps
      std::this_thread::sleep_for(std::chrono::milliseconds(33));
      continue;
    }

    retry_cnt = 0;

    // display the raw footage
    cv::imshow("KREDA", frame);

    if (cv::waitKey(1) == 'q') {
      std::cout << "Shutdown signal received. Exiting." << std::endl;
      break;
    }
  }

  // cleanup
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
