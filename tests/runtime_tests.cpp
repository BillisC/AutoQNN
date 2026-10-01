#include "test_suites.h"
#include "logger.hpp"

#ifdef ENABLE_CAMERA_TESTS
#include <gst/gst.h>
#endif

#include <cassert>

int main(int argc, char **argv) {
  assert(argc == 3);
  assert(qnn::log::initializeLogging());
  test_buffer();
#ifdef ENABLE_CAMERA_TESTS
  gst_init(nullptr, nullptr);
  test_camera();
#endif
  test_tensor();
  test_casts();
  test_model(argv[1], argv[2]);
  return 0;
}
