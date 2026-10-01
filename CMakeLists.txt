cmake_minimum_required (VERSION 3.21)

# Halt if no toolchain is selected
if(NOT CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "No toolchain file selected!")
endif()

# Define project
project(autoqnn)

find_package(PkgConfig REQUIRED)

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For clangd
set(CXX_WARNS "-Wall -Wextra -Wno-unused-but-set-variable -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNS} -Wl,-lstdc++")

# QNN options
set(QNN_TARGET "aarch64-linux")
set(QNN_SDK_ROOT "" CACHE PATH "Path to QNN API include headers")

message(STATUS "${PROJECT_NAME}: QNN_TARGET=${QNN_TARGET}")
message(STATUS "${PROJECT_NAME}: QNN_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
if(DEFINED ENV{QNN_SDK_ROOT})
	set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT} CACHE PATH "QNN API include" FORCE)
	message(STATUS "${PROJECT_NAME}: QNN_SDK_ROOT=${QNN_SDK_ROOT}")
else()
    message(FATAL_ERROR "${PROJECT_NAME}: QNN_SDK_ROOT not set (QAIRT), run envsetup.sh first")
endif()

# GStreamer check
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-audio-1.0 gstreamer-video-1.0 gstreamer-app-1.0)
message("GST_INCLUDE_DIRS = ${GST_INCLUDE_DIRS}")
message("GST_LIBRARY_DIRS = ${GST_LIBRARY_DIRS}")
message("GST_LIBRARIES = ${GST_LIBRARIES}")

# Compile app
add_subdirectory(src)

# Regression tests use the same SDK, toolchain and GStreamer configuration
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
