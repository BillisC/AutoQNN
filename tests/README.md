Regression tests use the same QAIRT/eSDK environment and aarch64 toolchain as
the application. They are included in the normal project build:

```sh
cmake -S . -B build --toolchain aarch64-linux-toolchain.cmake
cmake --build build
```

Existing module regression tests mirror the source folders. The shared runner
and any system-wide tests stay at the top level:

```text
tests/
  core/
    camera_tests.cpp
    model_tests.cpp
  utils/
    buffer_tests.cpp
    casts_tests.cpp
    tensor_tests.cpp
  mocks/
    mock_model.cpp
    qnn_mock.cpp
    qnn_mock.h
  stubs/
    allocation_stubs.cpp
    allocation_stubs.h
    camera_stubs.cpp
    camera_stubs.h
  runtime_tests.cpp
  test_suites.h
```

This layout reorganizes the existing regression coverage. New unit tests can
be added beside their corresponding modules later.

To build only the regression tests in a configured project:

```sh
cmake --build build --target runtime_tests
```

A standalone test build also selects the toolchain automatically:

```sh
cmake -S tests -B build/regression
cmake --build build/regression
```

`QNN_SDK_ROOT` and `SDKTARGETSYSROOT` come from the sourced SDK environment.
No compiler, SDK path, camera-test, or sanitizer parameters are required.
Camera tests use synthetic GStreamer frames and require the `videotestsrc`
plugin; no camera or QNN device is needed.

Run the aarch64 binaries on the target. Copy `runtime_tests`, `libmock_model.so`,
and `libmissing_symbols.so` from the test build directory into one directory:

```sh
./runtime_tests ./libmock_model.so ./libmissing_symbols.so
```

For a native target build, `ctest --test-dir build --output-on-failure`
also runs them. Cross builds need a configured CMake emulator to use CTest on
the host.

The toolchain enables AddressSanitizer and UndefinedBehaviorSanitizer in Debug
builds when their target runtimes are available. It reports missing runtimes
and builds without sanitizers otherwise.
