#include "test_suites.h"

#ifdef ENABLE_CAMERA_TESTS
#include "camera.h"
#include "stubs/camera_stubs.h"

#include <cassert>

void test_camera() {
  using Result = camera::Camera::Result;
  camera::Camera camera(3, 2,
                        "videotestsrc num-buffers=3 pattern=solid-color "
                        "foreground-color=0xff112233 ! "
                        "video/x-raw,format=RGB,width=3,height=2 ! "
                        "appsink name=sink sync=false async=false",
                        "sink");
  assert(camera.start() == Result::OK);
  std::vector<uint8_t> frame{99};
  assert(camera.frame(frame) == Result::OK);
  /* RGB rows have 9 bytes of pixels and 3 bytes of padding */
  assert(frame == std::vector<uint8_t>({17, 34, 51, 17, 34, 51, 17, 34, 51, 17,
                                        34, 51, 17, 34, 51, 17, 34, 51}));
  camera_stub::fail_video_map = true;
  assert(camera.frame(frame) == Result::SINK_ERROR && frame.empty());
  camera_stub::fail_video_map = false;
  assert(camera.frame(frame) == Result::OK && frame.size() == 18);
  assert(camera.frame(frame) != Result::OK && frame.empty());
  assert(camera.stop() == Result::OK);
  assert(camera.stop() == Result::OK);

  camera::Camera wrong_format(3, 2,
                              "videotestsrc num-buffers=1 ! "
                              "video/x-raw,format=BGR,width=3,height=2 ! "
                              "appsink name=sink sync=false async=false",
                              "sink");
  assert(wrong_format.start() == Result::OK);
  frame = {99};
  assert(wrong_format.frame(frame) == Result::SINK_ERROR && frame.empty());

  camera::Camera wrong_dimensions(4, 2,
                                  "videotestsrc num-buffers=1 ! "
                                  "video/x-raw,format=RGB,width=3,height=2 ! "
                                  "appsink name=sink sync=false async=false",
                                  "sink");
  assert(wrong_dimensions.start() == Result::OK);
  assert(wrong_dimensions.frame(frame) == Result::SINK_ERROR && frame.empty());
}
#endif
