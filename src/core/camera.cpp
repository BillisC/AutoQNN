/**
 * @file    camera.cpp
 * @brief   Camera class function definitions
 *
 * This file contains class function definitions for the GStreamer camera
 * wrapper class, under the camera namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include "camera.h"
#include "logger.hpp"

#include <gst/video/video.h>
#include <cstring>

namespace camera {

Camera::Camera(size_t width, size_t height, const char *pipeline_desc,
               const char *sink_name)
    : m_cam_width(width), m_cam_height(height), m_pipeline_desc(pipeline_desc),
      m_sink_name(sink_name) {}

Camera::Result Camera::start() {
  stop();
  GError *gerr = nullptr;
  m_pipeline = gst_parse_launch(m_pipeline_desc, &gerr);

  if (!m_pipeline || gerr) {
    QNN_ERROR("Failed to build pipeline: %s\n",
              gerr ? gerr->message : "unknown");
    if (gerr) {
      g_error_free(gerr);
    }
    return Result::PIPELINE_ERROR;
  }

  /* Start video sink */
  m_sink = gst_bin_get_by_name(GST_BIN(m_pipeline), m_sink_name);
  if (!m_sink) {
    return Result::SINK_ERROR;
  }

  if (gst_element_set_state(m_pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    return Result::PIPELINE_ERROR;
  }

  /* Clear error flags */
  m_msg_fail = false;
  m_running = true;

  /* Start threads */
  m_bus_thread = std::thread(&Camera::watch_bus, this);

  return Result::OK;
}

Camera::Result Camera::stop() {
  m_running = false;
  /* Stop pipeline */
  if (m_pipeline != nullptr) {
    gst_element_set_state(m_pipeline, GST_STATE_NULL);
  }

  /* Join threads */
  if (m_bus_thread.joinable()) {
    m_bus_thread.join();
  }

  /* Free pipeline */
  if (m_sink != nullptr) {
    gst_object_unref(m_sink);
    m_sink = nullptr;
  }
  if (m_pipeline != nullptr) {
    gst_object_unref(m_pipeline);
    m_pipeline = nullptr;
  }

  return Result::OK;
}

Camera::Result Camera::frame(std::vector<uint8_t> &v_buffer) {
  v_buffer.clear();
  /* Check if the stream is alive  */
  if (!m_sink || gst_app_sink_is_eos(GST_APP_SINK(m_sink))) {
    return Result::SINK_EOS; // EOS or sink uninitialized
  } else if (m_msg_fail) {
    return Result::MESSAGE_ERROR;
  }

  /* Pull latest frame from stream  */
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(m_sink));
  if (!sample) {
    return Result::SINK_ERROR; // pipeline stopped
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstCaps *caps = gst_sample_get_caps(sample);
  GstVideoInfo info;
  gst_video_info_init(&info);
  if (buffer == nullptr || caps == nullptr ||
      !gst_video_info_from_caps(&info, caps) ||
      GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_RGB ||
      GST_VIDEO_INFO_WIDTH(&info) != m_cam_width ||
      GST_VIDEO_INFO_HEIGHT(&info) != m_cam_height || m_cam_width <= 0 ||
      m_cam_height <= 0) {
    QNN_ERROR("Camera frame does not match the configured RGB dimensions");
    gst_sample_unref(sample);
    return Result::SINK_ERROR;
  }

  /* Map video rows, excluding any padding in the camera buffer */
  GstVideoFrame videoFrame;
  if (!gst_video_frame_map(&videoFrame, &info, buffer, GST_MAP_READ)) {
    QNN_ERROR("Failed to map camera frame");
    gst_sample_unref(sample);
    return Result::SINK_ERROR;
  }

  size_t rowSize = static_cast<size_t>(m_cam_width) * 3;
  int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&videoFrame, 0);
  const uint8_t *data =
      static_cast<const uint8_t *>(GST_VIDEO_FRAME_PLANE_DATA(&videoFrame, 0));
  if (data == nullptr || stride < 0 || static_cast<size_t>(stride) < rowSize) {
    gst_video_frame_unmap(&videoFrame);
    gst_sample_unref(sample);
    return Result::SINK_ERROR;
  }
  v_buffer.resize(rowSize * m_cam_height);
  for (int row = 0; row < m_cam_height; row++) {
    memcpy(v_buffer.data() + row * rowSize,
           data + static_cast<size_t>(row) * stride, rowSize);
  }
  gst_video_frame_unmap(&videoFrame);
  gst_sample_unref(sample);

  return Result::OK;
}

void Camera::watch_bus() {
  GstBus *bus = gst_element_get_bus(m_pipeline);
  while (m_running) {
    GstMessage *msg = gst_bus_timed_pop_filtered(
        bus, 100 * GST_MSECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (!msg) {
      continue;
    }
    /* Read filtered message */
    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError *err = nullptr;
        gchar *dbg = nullptr;
        gst_message_parse_error(msg, &err, &dbg);
        QNN_ERROR("GStreamer error: %s\n", err ? err->message : "unknown");
        m_msg_fail = true;
        m_running = false;
        g_clear_error(&err);
        g_free(dbg);
        break;
      }

      case GST_MESSAGE_EOS: {
        QNN_INFO("GStreamer: end of stream\n");
        m_running = false;
        break;
      }

      default: break;
    }

    gst_message_unref(msg);
  }

  gst_object_unref(bus);
}

} // namespace camera
