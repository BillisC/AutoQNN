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

namespace camera {

Camera::Camera(size_t width, size_t height, const char *pipeline_desc,
               const char *sink_name)
    : m_cam_width(width), m_cam_height(height), m_pipeline_desc(pipeline_desc),
      m_sink_name(sink_name) {}

Camera::Result Camera::start() {
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

  gst_element_set_state(m_pipeline, GST_STATE_PLAYING);

  /* Start threads */
  m_bus_thread = std::thread(&Camera::watch_bus, this);

  return Result::OK;
}

Camera::Result Camera::stop() {
  /* Join threads */
  m_bus_thread.join();

  /* Free pipeline */
  gst_element_set_state(m_pipeline, GST_STATE_NULL);
  gst_object_unref(m_pipeline);

  return Result::OK;
}

Camera::Result Camera::frame(std::vector<uint8_t> v_buffer) {
  /* Check if the stream is alive  */
  if (!m_sink || !gst_app_sink_is_eos(GST_APP_SINK(m_sink))) {
    return Result::SINK_EOS; // EOS or sink uninitialized
  }

  /* Pull latest frame from stream  */
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(m_sink));
  if (!sample) {
    return Result::SINK_ERROR; // pipeline stopped
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);

  /* Map buffer data and copy to vector */
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    v_buffer.assign(map.data, map.data + map.size);
    gst_buffer_unmap(buffer, &map);
  }
  gst_sample_unref(sample);

  return Result::OK;
}

void Camera::watch_bus() {
  GstBus *bus = gst_element_get_bus(m_pipeline);
  GstMessage *msg = gst_bus_timed_pop_filtered(
      bus, GST_CLOCK_TIME_NONE,
      static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  if (msg) {
    /* Read filtered message */
    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError *err = nullptr;
        gchar *dbg = nullptr;
        gst_message_parse_error(msg, &err, &dbg);
        QNN_ERROR("GStreamer error: %s\n", err ? err->message : "unknown");
        g_clear_error(&err);
        g_free(dbg);
        break;
      }

      case GST_MESSAGE_EOS: {
        QNN_INFO("GStreamer: end of stream\n");
        break;
      }

      default: break;
    }

    gst_message_unref(msg);
  }

  gst_object_unref(bus);
}

} // namespace camera