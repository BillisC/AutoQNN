/**
 * @file    tensor.cpp
 * @brief   Tensor function definitions
 *
 * This file contains static and normal function definitions for QNN tensors
 * initialization, population, and freeing, under the tensor namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include "tensor.h"

#include <cstdlib>
#include <vector>

#include "QnnTypes.h"
#include "qnn_type_macros.hpp"
#include "logger.hpp"

#include "buffer.h"
#include "casts.h"

/* Measure QNN_DATATYPE_* enum bytes */
#define QNN_BYTES(val) ((((val)&0xFF) - 6 * (((val)&0xFF) >> 4)) >> 3)

static bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src) {
    QNN_ERROR("Received nullptr");
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require
  // the version to be set to correctly assign values
  dst->version = src->version;
  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, strndup(tensorName, strlen(tensorName)));
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition =
      QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding =
        QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.scaleOffsetEncoding =
        QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
  } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding =
        QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis =
        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        QNN_TENSOR_GET_QUANT_PARAMS(src)
            .axisScaleOffsetEncoding.numScaleOffsets;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src)
            .axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
          QNN_TENSOR_GET_QUANT_PARAMS(src)
              .axisScaleOffsetEncoding.numScaleOffsets *
          sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(src)
                                       .axisScaleOffsetEncoding.numScaleOffsets;
             idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              QNN_TENSOR_GET_QUANT_PARAMS(src)
                  .axisScaleOffsetEncoding.scaleOffset[idx]
                  .scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              QNN_TENSOR_GET_QUANT_PARAMS(src)
                  .axisScaleOffsetEncoding.scaleOffset[idx]
                  .offset;
        }
      }
    }
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
  QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
  QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
  if (QNN_TENSOR_GET_RANK(src) > 0) {
    QNN_TENSOR_SET_DIMENSIONS(
        dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
    if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
      memcpy(QNN_TENSOR_GET_DIMENSIONS(dst), QNN_TENSOR_GET_DIMENSIONS(src),
             QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
    }
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
          dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
      memcpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
             QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
             QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

bool tensor::setup_tensors(Qnn_Tensor_t **tensors, uint32_t tensor_count,
                           Qnn_Tensor_t *tensor_wrappers) {
  if (tensor_wrappers == nullptr || tensor_count == 0) {
    QNN_ERROR("setupTensors parameters null");
    return false;
  }

  *tensors = new Qnn_Tensor_t[tensor_count]{};

  for (size_t tensorIdx = 0; tensorIdx < tensor_count; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = tensor_wrappers[tensorIdx];
    std::vector<size_t> dims;

    /* Copy wrapper structure */
    uint32_t *wrapperDims = QNN_TENSOR_GET_DIMENSIONS(wrapperTensor);
    uint32_t wrapperRank = QNN_TENSOR_GET_RANK(wrapperTensor);
    for (size_t r = 0; r < wrapperRank; r++) dims.push_back(wrapperDims[r]);
    (*tensors)[tensorIdx] = QNN_TENSOR_INIT;

    if (deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor) ==
        true) {
      QNN_DEBUG("deepCopyQnnTensorInfo successful");
      QNN_TENSOR_SET_MEM_TYPE(((*tensors) + tensorIdx), QNN_TENSORMEMTYPE_RAW);
    } else {
      return false;
    }

    /* Setup client buffer */
    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
    buffer::allocate_buffer(reinterpret_cast<uint8_t **>(&clientBuffer.data),
                            dims,
                            QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));

    size_t length = QNN_BYTES(QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));
    length *= buffer::count_elements(dims);
    clientBuffer.dataSize = length;
    QNN_TENSOR_SET_CLIENT_BUF(((*tensors) + tensorIdx), clientBuffer);
  }

  return true;
}

bool tensor::fill_tensor(Qnn_Tensor_t *tensor,
                         const std::vector<uint8_t> &input_data) {
  if (tensor == nullptr) {
    QNN_ERROR("fill_tensor: tensor is nullptr");
    return false;
  }

  /* Get allocated client buffer */
  Qnn_ClientBuffer_t clientBuf = QNN_TENSOR_GET_CLIENT_BUF(*tensor);
  if (clientBuf.data == nullptr) {
    QNN_ERROR("fill_tensor: tensor has null client buffer");
    return false;
  }

  /* Get tensor data type */
  Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);

  /* Perform copy operation */
  if (dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
      dataType == QNN_DATATYPE_UINT_8) {

    /* Verify size */
    size_t dstSize = clientBuf.dataSize;
    if (input_data.size() != dstSize) {
      QNN_ERROR("fill_tensor: expects %zu bytes but input has %zu", dstSize,
                input_data.size());
      return false;
    }

    /* Direct input mapping */
    memcpy(reinterpret_cast<uint8_t *>(clientBuf.data), input_data.data(),
           dstSize);
    return true;

  } else {
    QNN_ERROR("fill_tensor: unsupported tensor input type!");
    return false;
  }
}

bool tensor::fill_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                          std::vector<std::vector<uint8_t>> &input_data) {
  /* Check parameter inputs */
  if (tensors == nullptr) {
    QNN_ERROR("fill tensors are nullptr");
    return false;
  }

  if (input_data.size() < tensor_count) {
    QNN_ERROR("not enough input buffers provided (%zu < %u)", input_data.size(),
              tensor_count);
    return false;
  }

  /* Fill each tensor input */
  for (uint32_t t = 0; t < tensor_count; ++t) {
    if (!fill_tensor(tensors + t, input_data[t])) {
      QNN_ERROR("fill_tensors: failed on tensor %u", t);
      return false;
    }
  }

  return true;
}

bool tensor::free_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count) {
  for (size_t tensorIdx = 0; tensorIdx < tensor_count; tensorIdx++) {
    QNN_DEBUG("freeing resources for tensor: %d", tensorIdx);

    if (nullptr != QNN_TENSOR_GET_NAME(tensors[tensorIdx])) {
      QNN_DEBUG("freeing tensor name");
      delete[] const_cast<char *>(QNN_TENSOR_GET_NAME(tensors[tensorIdx]));
      QNN_TENSOR_SET_NAME(tensors[tensorIdx], nullptr);
    }

    if (nullptr != QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx])) {
      QNN_DEBUG("freeing dimensions");
      delete[] QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx]);
      QNN_TENSOR_SET_DIMENSIONS(tensors[tensorIdx], nullptr);
    }

    if (nullptr != QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data) {
      QNN_DEBUG("freeing clientBuf.data");
      delete[] static_cast<uint8_t *>(
          QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data);
    }
  }

  delete[] tensors;

  return true;
}