cmake_minimum_required (VERSION 3.21)

# Create target executable
set(APP ${PROJECT_NAME})
add_executable(${APP} ${APP_SRCS})

# Set compile definitions
if(CMAKE_BUILD_TYPE EQUAL Debug)
  target_compile_definitions(${APP} PRIVATE QNN_DEBUG_ENABLE)
else()
  target_compile_definitions(${APP} PRIVATE "QNN_API=__attribute__((visibility(\"default\")))")
endif()

# Source files
target_sources(${APP}
  PRIVATE
  # Application
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  # Core
  ${CMAKE_CURRENT_SOURCE_DIR}/core/backend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/camera.cpp
  # Utilities
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/dynload.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/buffer.cpp
  # Logging
  ${CMAKE_CURRENT_SOURCE_DIR}/log/logger.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/log/log_utils.cpp
)

# Header files
target_include_directories(${APP}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/qnn
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/log
  ${QNN_SDK_ROOT}/include/QNN
  ${GST_INCLUDE_DIRS}
)

# Platform-specific libraries
find_package(Threads REQUIRED)
target_link_libraries(${APP} PRIVATE dl ${CMAKE_THREAD_LIBS_INIT} ${GST_LIBRARIES})
target_link_directories(${APP} PRIVATE ${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc11.2 ${GST_LIBRARY_DIRS})
#target_link_directories(${APP} PRIVATE ${QNN_SDK_ROOT}/lib/aarch64-ubuntu-gcc9.4)

# Target Execute (w/ssh)
# The following must be set when configuring build (e.g. -DTARGET_KEY="~/.ssh/..").
# The SSH keys must be generated from the host and copied to .ssh/authorized_keys
# on the target device.
set(TARGET_KEY "~/.ssh/rubikpi" CACHE STRING "Path to SSH private key")
set(TARGET_USER "ubuntu" CACHE STRING "Target SSH Username")
set(TARGET_IP "192.168.1.17" CACHE STRING "Target Device IP")
set(TARGET_DIR "/tmp" CACHE STRING "Target execution directory")

# Copy and execute on remote target
add_custom_target(target_run
  COMMAND printf '\\033[36m' # Change text color to cyan
  COMMAND scp -q -i ${TARGET_KEY} $<TARGET_FILE:${CMAKE_PROJECT_NAME}> ${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}
  COMMAND scp -q -i ${TARGET_KEY} ${CMAKE_SOURCE_DIR}/libefficientnet-lite4_quant.so ${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}
  COMMAND scp -q -i ${TARGET_KEY} ${CMAKE_SOURCE_DIR}/imagenet_classes.txt ${TARGET_USER}@${TARGET_IP}:${TARGET_DIR}
  COMMAND ssh -tt -i ${TARGET_KEY} ${TARGET_USER}@${TARGET_IP} "${TARGET_DIR}/${CMAKE_PROJECT_NAME}"
  COMMAND printf '\\033[0m' # Revert text color
  DEPENDS ${PROJECT_NAME}
  COMMENT "Executing program in target hardware"
)