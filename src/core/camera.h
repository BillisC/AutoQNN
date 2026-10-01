/**
 * @file    camera.h
 * @brief   Camera class header
 *
 * This header contains enums, and forward declarations for the GStreamer Camera
 * wrapper class, under the camera namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#ifndef CAMERA_H
#define CAMERA_H

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <atomic>
#include <vector>

namespace camera {

/**
 * @brief Camera class for gstreamer video capture.
 */
class Camera {
public:
  /**
   * @brief Camera pipeline result enum class.
   */
  enum class Result {
    OK,
    SINK_EOS,
    SINK_ERROR,
    PIPELINE_ERROR,
    MESSAGE_ERROR
  };

private:
  /* Stream info */
  const int m_cam_width;
  const int m_cam_height;

  /* Flags */
  std::atomic<bool> m_msg_fail{false}; /**< Sink message fail state */
  std::atomic<bool> m_running{false};  /**< Bus watcher state */

  /* Pipeline control */
  GstElement *m_pipeline{nullptr}; /**< Pipeline handle */
  GstElement *m_sink{nullptr};     /**< Sink handle */
  const char *m_pipeline_desc;     /**< Pipeline description */
  const char *m_sink_name;

  std::thread m_bus_thread;

public:
  /* Class constructors */
  /**
   * @brief Camera class constructor.
   *
   * @param[in] width Width of the video stream
   * @param[in] height Height of the video stream
   * @param[in] pipeline_desc The pipeline description string
   * @param[in] sink_name The sink identifier name
   */
  Camera(size_t width, size_t height, const char *pipeline_desc,
         const char *sink_name);
  /**
   * @brief Camera class destructor.
   */
  ~Camera() { stop(); };

  /* Actions */
  /**
   * @brief Initialize and start the GStreamer pipeline.
   *
   * This function initializes the pipeline / sink with respect to the passed
   * description, and then changes the state to play. The start time depends on
   * the hardware and should be executed before inference starts.
   *
   * @return result enum value
   */
  Result start();

  /**
   * @brief Stop the GStreamer pipeline and cleanup.
   * @note This is NOT a pause function.
   * @return result enum value
   */
  Result stop();

  /**
   * @brief Fetch latest frame from GStreamer pipeline sink.
   *
   * This function receives the latest sink frame from the GStreamer pipeline
   * and copies packed RGB rows to the passed vector buffer. Caps must match
   * the configured width and height. On failure the vector is cleared.
   * The vector buffer should have reserved width*height*channels bytes to
   * avoid unwanted overheads.
   *
   * @param[out] v_buffer Vector buffer to store the frame
   * @return result enum value
   */
  Result frame(std::vector<uint8_t> &v_buffer);

private:
  /* Threaded functions */
  /**
   * @brief Handle continuous GStreamer messages.
   * @note Should be threaded ideally.
   */
  void watch_bus();
};

} // namespace camera

#endif
