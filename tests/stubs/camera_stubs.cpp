#include "camera_stubs.h"

#ifdef ENABLE_CAMERA_TESTS
#include <gst/video/video.h>

namespace camera_stub {

bool fail_video_map = false;

} // namespace camera_stub
extern "C" gboolean __real_gst_video_frame_map(GstVideoFrame *,
                                               const GstVideoInfo *,
                                               GstBuffer *, GstMapFlags);
extern "C" gboolean __wrap_gst_video_frame_map(GstVideoFrame *frame,
                                               const GstVideoInfo *info,
                                               GstBuffer *buffer,
                                               GstMapFlags flags) {
  return camera_stub::fail_video_map
             ? FALSE
             : __real_gst_video_frame_map(frame, info, buffer, flags);
}
#endif
