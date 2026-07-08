cmake_minimum_required(VERSION 3.20)

project(kreda-core-vision LANGUAGES CXX)

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

if (APPLE) # explicitly write apple sdk path
    execute_process(COMMAND xcrun --show-sdk-path
                    OUTPUT_VARIABLE MAC_SYSROOT
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MAC_SYSROOT}")
endif()

add_library(kreda_vision_lib STATIC
    src/calibration.cpp
    src/ingestion.cpp
)
target_link_libraries(kreda_vision_lib PUBLIC 
    opencv_core
    opencv_imgproc
    opencv_videoio
    opencv_highgui
    opencv_features2d
    opencv_calib3d
)
target_include_directories(kreda_vision_lib PUBLIC
    src
)

add_executable(kreda_vision_engine
  src/main.cpp
)
target_link_libraries(kreda_vision_engine PRIVATE
    kreda_vision_lib
)

option(KREDA_BUILD_TESTS "Build unit tests" ON)
if (KREDA_BUILD_TESTS)
    function(kreda_add_test BASE_NAME)
        set(TEST_NAME "test_${BASE_NAME}")
        set(SOURCE_FILE "tests/${TEST_NAME}.cpp")

        add_executable(${TEST_NAME}
            ${SOURCE_FILE}
            tests/test_main.cpp
        )

        target_link_libraries(${TEST_NAME} PRIVATE
            kreda_vision_lib
        )

        target_include_directories(${TEST_NAME} PRIVATE
            tests
        )

        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 60)
    endfunction()

    kreda_add_test(latest_frame)
    kreda_add_test(grid)
    kreda_add_test(calibration)
endif()
