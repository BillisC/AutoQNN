#include "model.h"
#include "qnn_type_macros.hpp"

#include <cstdlib>
#include <cstring>

#ifndef MISSING_SYMBOLS

static bool mode(const char *name) {
  const char *value = getenv("AUTOQNN_TEST_MODEL_MODE");
  return value != nullptr && strcmp(value, name) == 0;
}

extern "C" model::Model::Result
QnnModel_composeGraphs(Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE,
                       Qnn_ContextHandle_t,
                       const model::Model::GraphConfigInfo_t **, const uint32_t,
                       model::Model::GraphInfo_t ***graphs, uint32_t *count,
                       bool, QnnLog_Callback_t, QnnLog_Level_t) {
  if (mode("empty")) {
    return model::Model::Result::OK;
  }

  /* Two separate graph objects expose incorrect graph-pointer indexing */
  static uint32_t dims[] = {2};
  static uint32_t imageDims[] = {1, 2, 3, 2};
  static uint32_t outputDims[] = {1, 2};
  *count = (mode("single") || mode("image")) ? 1 : 2;
  *graphs = new model::Model::GraphInfo_t *[*count] {};
  for (uint32_t g = 0; g < *count; g++) {
    auto *info = new model::Model::GraphInfo_t{};
    (*graphs)[g] = info;
    info->graph =
        reinterpret_cast<Qnn_GraphHandle_t>(static_cast<uintptr_t>(g + 1));
    info->numInputTensors = info->numOutputTensors = 1;
    info->inputTensors = new Qnn_Tensor_t[1];
    info->outputTensors = new Qnn_Tensor_t[1];
    Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
    QNN_TENSOR_SET_NAME(tensor, "mock");
    QNN_TENSOR_SET_DATA_FORMAT(tensor, QNN_TENSOR_DATA_FORMAT_DENSE);
    QNN_TENSOR_SET_RANK(tensor, 1);
    QNN_TENSOR_SET_DIMENSIONS(tensor, dims);
    QNN_TENSOR_SET_DATA_TYPE(tensor, QNN_DATATYPE_UINT_8);
    info->inputTensors[0] = tensor;
    info->outputTensors[0] = tensor;
    if (mode("image")) {
      Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
      q.encodingDefinition = QNN_DEFINITION_DEFINED;
      q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      q.scaleOffsetEncoding = {1.0f / 128.0f, -127};
      QNN_TENSOR_SET_RANK(info->inputTensors[0], 4);
      QNN_TENSOR_SET_DIMENSIONS(info->inputTensors[0], imageDims);
      QNN_TENSOR_SET_DATA_TYPE(info->inputTensors[0],
                               QNN_DATATYPE_UFIXED_POINT_8);
      QNN_TENSOR_SET_QUANT_PARAMS(info->inputTensors[0], q);
      q.scaleOffsetEncoding = {1.0f / 256.0f, 0};
      QNN_TENSOR_SET_RANK(info->outputTensors[0], 2);
      QNN_TENSOR_SET_DIMENSIONS(info->outputTensors[0], outputDims);
      QNN_TENSOR_SET_DATA_TYPE(info->outputTensors[0],
                               QNN_DATATYPE_UFIXED_POINT_8);
      QNN_TENSOR_SET_QUANT_PARAMS(info->outputTensors[0], q);
    }
  }
  if (mode("tensor_error")) {
    QNN_TENSOR_SET_DATA_TYPE((*graphs)[1]->outputTensors[0],
                             QNN_DATATYPE_UNDEFINED);
  }
  return mode("compose_error") ? model::Model::Result::GRAPH_ERROR
                               : model::Model::Result::OK;
}

extern "C" model::Model::Result
QnnModel_freeGraphsInfo(model::Model::GraphInfo_t ***graphs, uint32_t count) {
  for (uint32_t g = 0; g < count; g++) {
    delete[](*graphs)[g]->inputTensors;
    delete[](*graphs)[g]->outputTensors;
    delete (*graphs)[g];
  }
  delete[] * graphs;
  *graphs = nullptr;
  return model::Model::Result::OK;
}

#endif
