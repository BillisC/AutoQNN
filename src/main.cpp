/**
 * @file    main.cpp
 * @brief   The main app
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <thread>
#include <vector>
#include <chrono>

// QNN Logging
#include "logger.hpp"

#include "backend.h"
#include "model.h"
#include "camera.h"

int main(int argc, char *argv[]) {
  /* Initialize logging */
  qnn::log::initializeLogging();
  qnn::log::setLogLevel(QNN_LOG_LEVEL_DEBUG);

  QNN_INFO("hello qualcomm\n");

  /* Initialize GStreamer */
  gst_init(&argc, &argv);

  const char *pipelineDesc =
      //"qtiqmmfsrc ! "
      //"video/x-raw,format=NV12,width=224,height=224,framerate=30/1 ! "
      "v4l2src device=/dev/video0 ! "
      "video/x-raw,width=320,height=240,framerate=30/1 ! "
      "videoconvert ! "
      "videocrop left=40 right=40 top=0 bottom=0 ! "
      "videoscale ! "
      "appsink name=mysink emit-signals=true sync=false drop=true "
      "max-buffers=1 caps=video/x-raw,format=BGR,width=224,height=224";

  camera::Camera cam1(224, 224, pipelineDesc, "mysink");
  if (camera::Camera::Result::OK != cam1.start()) {
    QNN_ERROR("Camera 1 initialization failed");
    return -1;
  }

  /* Initialize backend */
  backend::DeviceBackend back(backend::DeviceBackend::DeviceType::HTP, true);
  if (backend::DeviceBackend::Result::OK != back.init()) {
    QNN_ERROR("Backend initialization failed");
    return -1;
  }

  /* Load model graph from shared lib */
  model::Model mod(back.backend(), back.device(), back.profile(),
                   back.interface());
  if (model::Model::Result::OK !=
      mod.load_model_lib("/tmp/libefficientnet-lite4_quant.so")) {
    QNN_ERROR("Model load failed");
    return -1;
  }

  std::vector<uint8_t> i_buffer;
  i_buffer.reserve(224 * 224 * 3);

  auto start = std::chrono::steady_clock::now();
  uint32_t frame_cnt = 0;

  /* Blocking parse frame */
  while (camera::Camera::Result::OK == cam1.frame(i_buffer)) {
    mod.fill_input(i_buffer);
    mod.execute();

    frame_cnt++;

    if (frame_cnt % 60 == 0) {
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = end - start;

      double fps = static_cast<double>(frame_cnt) / elapsed.count();
      QNN_INFO("Pipeline speed: %.2f FPS", fps);

      frame_cnt = 0;
      start = std::chrono::steady_clock::now();
    }
  }

  QNN_ERROR("Camera frame parsing stopped");

  QNN_INFO("Finished");

  return 0;
}