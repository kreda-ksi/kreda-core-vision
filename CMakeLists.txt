cmake_minimum_required(VERSION 3.20)

project(kreda-core-vision LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vision_engine
  src/main.cpp
  src/calibration.cpp
  src/ingestion.cpp
)

target_link_libraries(vision_engine
  ${OpenCV_LIBS}
)
