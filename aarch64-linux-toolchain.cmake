# The following toolchain was designed for aarch64 QNN Linux targets (e.g. QCS6490).

# Check if a Qualcomm eSDK has been set up properly.
if(NOT DEFINED ENV{SDKTARGETSYSROOT})
  message(FATAL_ERROR "SDKTARGETSYSROOT is not defined. Is the eSDK environment set up?")
endif()

# Target info
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(LINK_FLAGS "-fPIC -ldl -Wl,-Map=output.map")

# Toolchain settings
set(CMAKE_C_COMPILER      "aarch64-qcom-linux-gcc")
set(CMAKE_CXX_COMPILER    "aarch64-qcom-linux-g++")
set(CMAKE_SYSROOT         "$ENV{SDKTARGETSYSROOT}")
set(CMAKE_FIND_ROOT_PATH  "${CMAKE_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# this makes the test compiles use static library option so that we don't need to pre-set linker flags and scripts
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# Provide pthreads values so `find_package(Threads)` succeeds.
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_THREAD_LIBS_INIT_RELEASE "${CMAKE_THREAD_LIBS_INIT}")
set(Threads_FOUND TRUE)
set(THREADS_HAVE_PTHREAD_ARG 1)
set(CMAKE_HAVE_THREADS_LIBRARY 1)

# Default compiler / linker flags
SET(CMAKE_C_FLAGS   "-fPIC -pg" CACHE INTERNAL "c compiler flags")
set(CMAKE_CXX_FLAGS "-fPIC -fno-exceptions -fno-rtti -pg" CACHE INTERNAL "cxx compiler flags")
set(CMAKE_ASM_FLAGS "-x assembler-with-cpp" CACHE INTERNAL "asm compiler flags")
set(CMAKE_EXE_LINKER_FLAGS "${LINK_FLAGS}" CACHE INTERNAL "exe link flags")

SET(CMAKE_C_FLAGS_DEBUG   "-Og -g -ggdb3" CACHE INTERNAL "c debug compiler flags")
SET(CMAKE_CXX_FLAGS_DEBUG "-Og -g -ggdb3" CACHE INTERNAL "cxx debug compiler flags")
SET(CMAKE_ASM_FLAGS_DEBUG "-g -ggdb3" CACHE INTERNAL "asm debug compiler flags")

SET(CMAKE_C_FLAGS_RELEASE   "-O3" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE INTERNAL "cxx release compiler flags")
SET(CMAKE_ASM_FLAGS_RELEASE "" CACHE INTERNAL "asm release compiler flags")