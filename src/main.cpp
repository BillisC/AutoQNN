/**
 * @file    main.cpp
 * @brief   The main app
 *
 * @author  BillisC (Vasileios Ch.)
 */

// QNN Logging
#include "logger.hpp"

#include "backend.h"
#include "model.h"

int main() {
  /* Initialize logging */
  qnn::log::initializeLogging();
  qnn::log::setLogLevel(QNN_LOG_LEVEL_DEBUG);

  QNN_INFO("hello qualcomm\n");

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
  mod.execute();
  QNN_INFO("Finished");

  return 0;
}
