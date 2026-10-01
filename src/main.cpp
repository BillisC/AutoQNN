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
#include "buffer.h"

#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>

bool load_labels(const std::string &path,
                 std::vector<std::string> &out_labels) {
  std::ifstream file(path);
  if (!file.is_open()) {
    QNN_ERROR("load_labels: failed to open %s", path.c_str());
    return false;
  }

  out_labels.clear();
  std::string line;
  while (std::getline(file, line)) {
    /* Strip trailing carriage return, in case the file has CRLF endings */
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    out_labels.push_back(line);
  }

  if (out_labels.empty()) {
    QNN_ERROR("load_labels: %s contained no labels", path.c_str());
    return false;
  }

  return true;
}

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
      "max-buffers=1 caps=video/x-raw,format=RGB,width=224,height=224";

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

  /* Contract of the bundled EfficientNet model, checked before inference */
  const model::Model::TensorSpec inputSpec = {
      {1, 224, 3, 224}, QNN_DATATYPE_UFIXED_POINT_8, 1.0f / 128.0f, -127};
  const model::Model::TensorSpec outputSpec = {
      {1, 1000}, QNN_DATATYPE_UFIXED_POINT_8, 1.0f / 256.0f, 0};
  if (mod.validate_io(inputSpec, outputSpec) != model::Model::Result::OK) {
    QNN_ERROR("Model is incompatible with the camera/classification pipeline");
    return -1;
  }

  /* Load labels once at startup */
  std::vector<std::string> class_labels;
  if (!load_labels("/tmp/imagenet_classes.txt", class_labels)) {
    QNN_ERROR("Could not load labels, aborting");
    return -1;
  }
  if (class_labels.size() != outputSpec.dimensions[1]) {
    QNN_ERROR("Label count does not match the model output");
    return -1;
  }

  std::vector<uint8_t> frame_buffer;
  frame_buffer.reserve(224 * 224 * 3);
  std::vector<uint8_t> i_buffer;
  i_buffer.reserve(224 * 224 * 3);

  std::vector<float> o_buffer;

  auto start = std::chrono::steady_clock::now();
  uint32_t frame_cnt = 0;

  /* Blocking parse frame */
  while (camera::Camera::Result::OK == cam1.frame(frame_buffer)) {
    /* Invert the model's [0,3,1,2] transpose for packed NHWC camera data.
     * Pixels already encode the expected (pixel - 127) / 128 input range. */
    if (!buffer::reorder_buffer(frame_buffer, {1, 224, 224, 3}, {0, 2, 3, 1},
                                i_buffer)) {
      QNN_ERROR("Failed to reorder camera input!");
      continue;
    }
    if (mod.fill_input(i_buffer) != model::Model::Result::OK) {
      QNN_ERROR("Failed to fill input!");
      continue;
    }

    if (mod.execute() != model::Model::Result::OK) {
      QNN_ERROR("Failed to execute!");
      continue;
    }

    if (mod.output(o_buffer) != model::Model::Result::OK) {
      QNN_ERROR("Failed to get output buffer!");
      continue;
    }

    if (o_buffer.size() != class_labels.size() ||
        !std::all_of(o_buffer.begin(), o_buffer.end(),
                     [](float score) { return std::isfinite(score); })) {
      QNN_ERROR("Invalid classification output!");
      continue;
    }

    /* Print output */
    auto best_it = std::max_element(o_buffer.begin(), o_buffer.end());
    size_t best_idx = std::distance(o_buffer.begin(), best_it);
    const std::string &label = class_labels[best_idx];

    QNN_INFO("Predicted: %s (class=%zu, score=%.4f)", label.c_str(), best_idx,
             *best_it);

    /* FPS count */
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

  QNN_INFO("Camera frame parsing stopped");

  QNN_INFO("Finished");

  return 0;
}
