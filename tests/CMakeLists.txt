cmake_minimum_required(VERSION 3.21)

# Standalone builds use the same configuration as the root project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/../aarch64-linux-toolchain.cmake"
      CACHE FILEPATH "Target toolchain" FORCE)
  project(autoqnn_tests LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CXX_WARNS "-Wall -Wextra -Wno-unused-but-set-variable -pedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNS} -Wl,-lstdc++")

  set(QNN_SDK_ROOT "" CACHE PATH "Path to QNN API include headers")
  if(DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT} CACHE PATH "QNN API include" FORCE)
    message(STATUS "${PROJECT_NAME}: QNN_SDK_ROOT=${QNN_SDK_ROOT}")
  else()
    message(FATAL_ERROR "${PROJECT_NAME}: QNN_SDK_ROOT not set (QAIRT), run envsetup.sh first")
  endif()

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-audio-1.0 gstreamer-video-1.0 gstreamer-app-1.0)
  enable_testing()
endif()

find_package(Threads REQUIRED)

add_library(mock_model SHARED mocks/mock_model.cpp)
add_library(missing_symbols SHARED mocks/mock_model.cpp)
target_compile_definitions(missing_symbols PRIVATE MISSING_SYMBOLS)
add_executable(runtime_tests
  runtime_tests.cpp
  core/camera_tests.cpp
  core/model_tests.cpp
  utils/buffer_tests.cpp
  utils/casts_tests.cpp
  utils/tensor_tests.cpp
  mocks/qnn_mock.cpp
  stubs/allocation_stubs.cpp
  stubs/camera_stubs.cpp
  ../src/core/model.cpp
  ../src/core/camera.cpp
  ../src/utils/tensor.cpp
  ../src/utils/buffer.cpp
  ../src/utils/dynload.cpp
  ../src/log/logger.cpp
  ../src/log/log_utils.cpp
)

foreach(TEST_TARGET mock_model missing_symbols runtime_tests)
  target_include_directories(${TEST_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QNN_SDK_ROOT}/include/QNN
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/qnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/log
    ${GST_INCLUDE_DIRS}
  )
  target_compile_options(${TEST_TARGET} PRIVATE -UNDEBUG)
endforeach()

target_compile_definitions(runtime_tests PRIVATE ENABLE_CAMERA_TESTS)
target_link_libraries(runtime_tests PRIVATE dl ${CMAKE_THREAD_LIBS_INIT} ${GST_LIBRARIES})
target_link_directories(runtime_tests PRIVATE ${GST_LIBRARY_DIRS})
target_link_options(runtime_tests PRIVATE -Wl,--wrap=calloc -Wl,--wrap=strdup
  -Wl,--wrap=gst_video_frame_map)
add_dependencies(runtime_tests mock_model missing_symbols)
add_test(NAME runtime_tests COMMAND runtime_tests
  $<TARGET_FILE:mock_model> $<TARGET_FILE:missing_symbols>
)
