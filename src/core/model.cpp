/**
 * @file    model.cpp
 * @brief   Model class function definitions
 *
 * This file contains static and class function definitions for the QNN Model
 * wrapper class, under the model namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include "model.h"

#include <fstream>
#include <vector>

#include "logger.hpp"
#include "dynload.h"
#include "tensor.h"

namespace model {

/* -- Static helper functions -- */

// static Model::Result readFile(const char *path, std::vector<uint8_t> &buf) {
//   /* Open file */
//   std::ifstream file = std::ifstream(path, std::ios::binary | std::ios::ate);
//   if (!file)
//     return Model::Result::FILE_ERROR;

//   /* Prepare read buffer */
//   const auto size = file.tellg();
//   buf.resize(size);

//   /* Store file content in buffer */
//   file.seekg(0, std::ios::beg);
//   file.read(reinterpret_cast<char *>(buf.data()), size);
//   file.close();

//   return Model::Result::OK;
// }

/* Class Function definitions */

Model::Model(Qnn_BackendHandle_t *backend_handle,
             Qnn_DeviceHandle_t *device_handle,
             Qnn_ProfileHandle_t *profile_handle,
             QNN_INTERFACE_VER_TYPE *qnn_interface) {
  m_backend_handle = backend_handle;
  m_device_handle = device_handle;
  m_profile_handle = profile_handle;
  m_qnn_interface = qnn_interface;
}

Model::Result Model::load_model_lib(const char *lib_path) {
  /* Parameter checking  */
  if (m_backend_handle == nullptr || m_device_handle == nullptr ||
      m_qnn_interface == nullptr || lib_path == nullptr) {
    return Model::Result::PARAMS_ERROR;
  }

  /* 1. Create inference context */
  const QnnContext_Config_t *ctxCfg = nullptr;
  if (QNN_SUCCESS != m_qnn_interface->contextCreate(*m_backend_handle,
                                                    *m_device_handle, &ctxCfg,
                                                    &m_context_handle)) {
    return Model::Result::CONTEXT_ERROR;
  }

  /* 2. Load model library and resolve symbols */
  m_libModelHandle = dl::dl_open(lib_path, RTLD_NOW | RTLD_LOCAL);
  if (m_libModelHandle == nullptr) {
    return Model::Result::FILE_ERROR;
  }

  ComposeGraphsFnHandleType_t composeGraphsFnHandle =
      dl::resolve_symbol<ComposeGraphsFnHandleType_t>(m_libModelHandle,
                                                      "QnnModel_composeGraphs");
  /* Store to free graph later */
  m_freeGraphsInfoFn = dl::resolve_symbol<FreeGraphInfoFnHandleType_t>(
      m_libModelHandle, "QnnModel_freeGraphsInfo");
  if (composeGraphsFnHandle == nullptr || m_freeGraphsInfoFn == nullptr) {
    return Model::Result::FILE_ERROR;
  }

  /* 3. Compose loaded graphs */
  if (Model::Result::OK !=
      composeGraphsFnHandle(*m_backend_handle, *m_qnn_interface,
                            m_context_handle, nullptr, 0, &m_graphs_info,
                            &m_graphs_count, false, qnn::log::getLogCallback(),
                            QNN_LOG_LEVEL_DEBUG)) {
    return Model::Result::GRAPH_ERROR;
  }

  if (m_graphs_count == 0 || m_graphs_info == nullptr) {
    return Model::Result::GRAPH_ERROR;
  }

  /* 4. Finalize graphs */
  for (size_t graphIdx = 0; graphIdx < m_graphs_count; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnn_interface->graphFinalize((*m_graphs_info)[graphIdx].graph,
                                       *m_profile_handle, nullptr)) {
      return Model::Result::GRAPH_ERROR;
    }
  }

  /* 5. Initialize tensor structures for all graphs */
  if (Model::Result::OK != init_tensors()) {
    return Model::Result::TENSOR_ERROR;
  }

  return Model::Result::OK;
}

Model::Result Model::execute() {
  GraphInfo &g = *m_graphs_info[0];

  /* Perform model inference */
  QNN_INFO("Running inference..");
  m_qnn_interface->graphExecute(g.graph, m_tensor_inputs[0], g.numInputTensors,
                                m_tensor_outputs[0], g.numOutputTensors,
                                *m_profile_handle, nullptr);
  return Model::Result::OK;
}

Model::Result Model::init_tensors() {
  if (m_graphs_info == nullptr || m_graphs_count == 0) {
    return Model::Result::GRAPH_ERROR;
  }

  m_tensor_inputs = new Qnn_Tensor_t *[m_graphs_count] {};
  m_tensor_outputs = new Qnn_Tensor_t *[m_graphs_count] {};

  /* Setup input / output tensors for all graphs */
  for (size_t gIdx = 0; gIdx < m_graphs_count; gIdx++) {
    GraphInfo &g = *m_graphs_info[gIdx];

    tensor::setup_tensors(&m_tensor_inputs[gIdx], g.numInputTensors,
                          g.inputTensors);
    tensor::setup_tensors(&m_tensor_outputs[gIdx], g.numOutputTensors,
                          g.outputTensors);
  }

  return Model::Result::OK;
}

Model::Result Model::close() {
  /* Tensor teardown */
  if (m_graphs_info != nullptr) {
    /* Free tensor inputs */
    if (m_tensor_inputs != nullptr) {
      for (size_t gIdx = 0; gIdx < m_graphs_count; gIdx++) {
        if (m_tensor_inputs[gIdx] != nullptr) {
          GraphInfo &g = *m_graphs_info[gIdx];
          tensor::free_tensors(m_tensor_inputs[gIdx], g.numInputTensors);
        }
      }
      delete[] m_tensor_inputs;
      m_tensor_inputs = nullptr;
    }

    /* Free tensor outputs */
    if (m_tensor_outputs != nullptr) {
      for (size_t gIdx = 0; gIdx < m_graphs_count; gIdx++) {
        if (m_tensor_outputs[gIdx] != nullptr) {
          GraphInfo &g = *m_graphs_info[gIdx];
          tensor::free_tensors(m_tensor_outputs[gIdx], g.numOutputTensors);
        }
      }
      delete[] m_tensor_outputs;
      m_tensor_outputs = nullptr;
    }
  }

  /* Free graphs info */
  if (m_graphs_info != nullptr && m_freeGraphsInfoFn != nullptr) {
    m_freeGraphsInfoFn(&m_graphs_info, m_graphs_count);
    m_graphs_info = nullptr;
    m_graphs_count = 0;
  }

  /* Free QNN context */
  if (m_context_handle != nullptr && m_qnn_interface != nullptr) {
    m_qnn_interface->contextFree(m_context_handle, *m_profile_handle);
    m_context_handle = nullptr;
  }

  /* Close dynamic model library */
  if (m_libModelHandle != nullptr) {
    dl::dl_close(m_libModelHandle);
    m_libModelHandle = nullptr;
  }

  return Result::OK;
}

} // namespace model