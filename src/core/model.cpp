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
#include <new>

#include "logger.hpp"
#include "dynload.h"
#include "qnn_type_macros.hpp"
#include "tensor.h"

namespace model {

/* -- Static helper functions -- */

static bool matches_spec(Qnn_Tensor_t *tensor, const Model::TensorSpec &spec) {
  if (QNN_TENSOR_GET_DATA_FORMAT(tensor) != QNN_TENSOR_DATA_FORMAT_DENSE ||
      QNN_TENSOR_GET_DATA_TYPE(tensor) != spec.data_type ||
      QNN_TENSOR_GET_RANK(tensor) != spec.dimensions.size()) {
    return false;
  }

  uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
  uint8_t *dynamicDims = QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor);
  if (dims == nullptr || spec.dimensions.empty()) {
    return false;
  }
  for (size_t d = 0; d < spec.dimensions.size(); d++) {
    if (dims[d] != spec.dimensions[d] ||
        (dynamicDims != nullptr && dynamicDims[d] != 0)) {
      return false;
    }
  }

  if (spec.data_type == QNN_DATATYPE_UFIXED_POINT_8 ||
      spec.data_type == QNN_DATATYPE_SFIXED_POINT_8 ||
      spec.data_type == QNN_DATATYPE_UFIXED_POINT_16 ||
      spec.data_type == QNN_DATATYPE_SFIXED_POINT_16 ||
      spec.data_type == QNN_DATATYPE_UFIXED_POINT_32 ||
      spec.data_type == QNN_DATATYPE_SFIXED_POINT_32) {
    tensor::QuantParams params{};
    return tensor::get_quant_params(tensor, params) &&
           params.scale == spec.scale && params.offset == spec.offset;
  }
  return true;
}

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
  m_inputs_ready = false;
  m_output_valid = false;
  /* Parameter checking  */
  if (m_backend_handle == nullptr || m_device_handle == nullptr ||
      m_qnn_interface == nullptr || lib_path == nullptr ||
      m_qnn_interface->contextCreate == nullptr ||
      m_qnn_interface->contextFree == nullptr ||
      m_qnn_interface->graphFinalize == nullptr ||
      m_qnn_interface->graphExecute == nullptr) {
    return Model::Result::PARAMS_ERROR;
  }

  /* Release any earlier load before creating a new context */
  close();

  /* 1. Create inference context */
  const QnnContext_Config_t *ctxCfg = nullptr;
  if (QNN_SUCCESS != m_qnn_interface->contextCreate(*m_backend_handle,
                                                    *m_device_handle, &ctxCfg,
                                                    &m_context_handle)) {
    close();
    return Model::Result::CONTEXT_ERROR;
  }

  /* 2. Load model library and resolve symbols */
  m_libModelHandle = dl::dl_open(lib_path, RTLD_NOW | RTLD_LOCAL);
  if (m_libModelHandle == nullptr) {
    close();
    return Model::Result::FILE_ERROR;
  }

  ComposeGraphsFnHandleType_t composeGraphsFnHandle =
      dl::resolve_symbol<ComposeGraphsFnHandleType_t>(m_libModelHandle,
                                                      "QnnModel_composeGraphs");
  /* Store to free graph later */
  m_freeGraphsInfoFn = dl::resolve_symbol<FreeGraphInfoFnHandleType_t>(
      m_libModelHandle, "QnnModel_freeGraphsInfo");
  if (composeGraphsFnHandle == nullptr || m_freeGraphsInfoFn == nullptr) {
    close();
    return Model::Result::FILE_ERROR;
  }

  /* 3. Compose loaded graphs */
  if (Model::Result::OK !=
      composeGraphsFnHandle(*m_backend_handle, *m_qnn_interface,
                            m_context_handle, nullptr, 0, &m_graphs_info,
                            &m_graphs_count, false, qnn::log::getLogCallback(),
                            QNN_LOG_LEVEL_DEBUG)) {
    close();
    return Model::Result::GRAPH_ERROR;
  }

  if (m_graphs_count == 0 || m_graphs_info == nullptr) {
    close();
    return Model::Result::GRAPH_ERROR;
  }

  /* 4. Finalize graphs */
  for (size_t graphIdx = 0; graphIdx < m_graphs_count; graphIdx++) {
    if (m_graphs_info[graphIdx] == nullptr ||
        QNN_GRAPH_NO_ERROR !=
            m_qnn_interface->graphFinalize(
                m_graphs_info[graphIdx]->graph,
                m_profile_handle == nullptr ? nullptr : *m_profile_handle,
                nullptr)) {
      close();
      return Model::Result::GRAPH_ERROR;
    }
  }

  /* 5. Initialize tensor structures for all graphs */
  if (Model::Result::OK != init_tensors()) {
    close();
    return Model::Result::TENSOR_ERROR;
  }

  GraphInfo &g = *m_graphs_info[0];
  for (uint32_t t = 0; t < g.numInputTensors; t++) {
    uint32_t rank = QNN_TENSOR_GET_RANK(g.inputTensors[t]);
    uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(g.inputTensors[t]);
    std::string dimStr;
    for (uint32_t d = 0; d < rank; d++) dimStr += std::to_string(dims[d]) + " ";
    QNN_INFO("Input tensor %u dims: %s", t, dimStr.c_str());
  }

  m_initialized = true;
  return Model::Result::OK;
}

Model::Result Model::validate_io(const TensorSpec &input,
                                 const TensorSpec &output) const {
  if (!m_initialized) {
    return Model::Result::SETUP_ERROR;
  }
  if (m_graphs_count != 1 || m_graphs_info[0]->numInputTensors != 1 ||
      m_graphs_info[0]->numOutputTensors != 1) {
    return Model::Result::INVALID_ARGUMENT_ERROR;
  }
  if (!matches_spec(m_tensor_inputs[0], input) ||
      !matches_spec(m_tensor_outputs[0], output)) {
    QNN_ERROR("Model input/output does not match the application contract");
    return Model::Result::TENSOR_ERROR;
  }
  return Model::Result::OK;
}

Model::Result Model::execute() {
  m_output_valid = false;
  if (!m_initialized || !m_inputs_ready) {
    return Model::Result::SETUP_ERROR;
  }

  /* Perform inference for every graph with independently filled inputs */
  for (size_t i = 0; i < m_graphs_count; i++) {
    GraphInfo &g = *m_graphs_info[i];
    Qnn_ErrorHandle_t status = m_qnn_interface->graphExecute(
        g.graph, m_tensor_inputs[i], g.numInputTensors, m_tensor_outputs[i],
        g.numOutputTensors,
        m_profile_handle == nullptr ? nullptr : *m_profile_handle, nullptr);
    if (status != QNN_GRAPH_NO_ERROR) {
      QNN_ERROR("graphExecute failed on graph %zu: %llu", i,
                static_cast<unsigned long long>(status));
      return Model::Result::GRAPH_ERROR;
    }
  }
  m_output_valid = true;
  return Model::Result::OK;
}

Model::Result Model::fill_input(std::vector<uint8_t> &i_buffer) {
  m_inputs_ready = false;
  m_output_valid = false;
  if (!m_initialized) {
    return Model::Result::SETUP_ERROR;
  }
  if (m_graphs_count != 1 || m_graphs_info[0]->numInputTensors != 1) {
    return Model::Result::INVALID_ARGUMENT_ERROR;
  }

  /* Fill single model input */
  if (!tensor::fill_tensor(m_tensor_inputs[0], i_buffer)) {
    return Model::Result::TENSOR_ERROR;
  }

  m_inputs_ready = true;
  return Model::Result::OK;
}

Model::Result
Model::fill_inputs(std::vector<std::vector<std::vector<uint8_t>>> &i_buffer) {
  m_inputs_ready = false;
  m_output_valid = false;
  if (!m_initialized) {
    return Model::Result::SETUP_ERROR;
  }
  if (i_buffer.size() != m_graphs_count) {
    return Model::Result::INVALID_ARGUMENT_ERROR;
  }

  /* Fill tensors for each graph */
  for (size_t i = 0; i < m_graphs_count; i++) {
    GraphInfo &g = *m_graphs_info[i];
    if (!tensor::fill_tensors(m_tensor_inputs[i], g.numInputTensors,
                              i_buffer[i])) {
      return Model::Result::TENSOR_ERROR;
    }
  }

  m_inputs_ready = true;
  return Model::Result::OK;
}

Model::Result Model::output(std::vector<uint8_t> &o_buffer) {
  o_buffer.clear();
  if (!m_initialized || !m_output_valid) {
    return Model::Result::SETUP_ERROR;
  }
  if (m_graphs_count != 1 || m_graphs_info[0]->numOutputTensors != 1) {
    return Model::Result::INVALID_ARGUMENT_ERROR;
  }

  /* Read single model output */
  if (!tensor::read_tensor(m_tensor_outputs[0], o_buffer)) {
    return Model::Result::TENSOR_ERROR;
  }

  return Model::Result::OK;
}

Model::Result Model::output(std::vector<float> &o_buffer) {
  o_buffer.clear();
  if (!m_initialized || !m_output_valid) {
    return Model::Result::SETUP_ERROR;
  }
  if (m_graphs_count != 1 || m_graphs_info[0]->numOutputTensors != 1) {
    return Model::Result::INVALID_ARGUMENT_ERROR;
  }

  if (!tensor::read_tensor(m_tensor_outputs[0], o_buffer)) {
    return Model::Result::TENSOR_ERROR;
  }
  return Model::Result::OK;
}

Model::Result
Model::outputs(std::vector<std::vector<std::vector<uint8_t>>> &o_buffer) {
  o_buffer.clear();
  if (!m_initialized || !m_output_valid) {
    return Model::Result::SETUP_ERROR;
  }

  /* Ensure destination has exactly one slot for each graph */
  o_buffer.resize(m_graphs_count);

  /* Read tensors for each graph */
  for (size_t i = 0; i < m_graphs_count; i++) {
    GraphInfo &g = *m_graphs_info[i];
    if (!tensor::read_tensors(m_tensor_outputs[i], g.numOutputTensors,
                              o_buffer[i])) {
      o_buffer.clear();
      return Model::Result::TENSOR_ERROR;
    }
  }

  return Model::Result::OK;
}

Model::Result
Model::outputs(std::vector<std::vector<std::vector<float>>> &o_buffer) {
  o_buffer.clear();
  if (!m_initialized || !m_output_valid) {
    return Model::Result::SETUP_ERROR;
  }

  o_buffer.resize(m_graphs_count);

  for (size_t i = 0; i < m_graphs_count; i++) {
    GraphInfo &g = *m_graphs_info[i];
    if (!tensor::read_tensors(m_tensor_outputs[i], g.numOutputTensors,
                              o_buffer[i])) {
      o_buffer.clear();
      return Model::Result::TENSOR_ERROR;
    }
  }
  return Model::Result::OK;
}

Model::Result Model::init_tensors() {
  if (m_graphs_info == nullptr || m_graphs_count == 0) {
    return Model::Result::GRAPH_ERROR;
  }

  m_tensor_inputs = new (std::nothrow) Qnn_Tensor_t *[m_graphs_count] {};
  m_tensor_outputs = new (std::nothrow) Qnn_Tensor_t *[m_graphs_count] {};
  if (m_tensor_inputs == nullptr || m_tensor_outputs == nullptr) {
    return Model::Result::MEMORY_ALLOCATE_ERROR;
  }

  /* Setup input / output tensors for all graphs */
  for (size_t gIdx = 0; gIdx < m_graphs_count; gIdx++) {
    if (m_graphs_info[gIdx] == nullptr) {
      return Model::Result::GRAPH_ERROR;
    }
    GraphInfo &g = *m_graphs_info[gIdx];

    if ((g.numInputTensors > 0 &&
         !tensor::setup_tensors(&m_tensor_inputs[gIdx], g.numInputTensors,
                                g.inputTensors)) ||
        (g.numOutputTensors > 0 &&
         !tensor::setup_tensors(&m_tensor_outputs[gIdx], g.numOutputTensors,
                                g.outputTensors))) {
      return Model::Result::TENSOR_ERROR;
    }
  }

  return Model::Result::OK;
}

Model::Result Model::close() {
  m_initialized = false;
  m_inputs_ready = false;
  m_output_valid = false;

  /* Tensor teardown */
  for (size_t gIdx = 0; gIdx < m_graphs_count; gIdx++) {
    if (m_graphs_info == nullptr || m_graphs_info[gIdx] == nullptr) {
      continue;
    }
    GraphInfo &g = *m_graphs_info[gIdx];
    if (m_tensor_inputs != nullptr) {
      tensor::free_tensors(m_tensor_inputs[gIdx], g.numInputTensors);
    }
    if (m_tensor_outputs != nullptr) {
      tensor::free_tensors(m_tensor_outputs[gIdx], g.numOutputTensors);
    }
  }
  delete[] m_tensor_inputs;
  delete[] m_tensor_outputs;
  m_tensor_inputs = nullptr;
  m_tensor_outputs = nullptr;

  /* Free graphs info */
  if (m_graphs_info != nullptr && m_freeGraphsInfoFn != nullptr) {
    m_freeGraphsInfoFn(&m_graphs_info, m_graphs_count);
  }
  m_graphs_info = nullptr;
  m_graphs_count = 0;

  /* Free QNN context */
  if (m_context_handle != nullptr && m_qnn_interface != nullptr) {
    m_qnn_interface->contextFree(m_context_handle, m_profile_handle == nullptr
                                                       ? nullptr
                                                       : *m_profile_handle);
    m_context_handle = nullptr;
  }

  /* Close dynamic model library */
  if (m_libModelHandle != nullptr) {
    dl::dl_close(m_libModelHandle);
    m_libModelHandle = nullptr;
  }
  m_freeGraphsInfoFn = nullptr;

  return Result::OK;
}

} // namespace model
