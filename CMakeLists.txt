cmake_minimum_required (VERSION 3.21)

# Halt if no toolchain is selected
if(NOT CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "No toolchain file selected!")
endif()

# Define project
project(autoqnn)

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

# Compile app
add_subdirectory(src)
