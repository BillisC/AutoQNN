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
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

#include "QnnTypes.h"
#include "qnn_type_macros.hpp"
#include "logger.hpp"

#include "buffer.h"
#include "casts.h"

static bool valid_client_buffer(const Qnn_Tensor_t &tensor) {
  if (!validateTensorVersion(tensor) ||
      QNN_TENSOR_GET_DATA_FORMAT(tensor) != QNN_TENSOR_DATA_FORMAT_DENSE ||
      QNN_TENSOR_GET_MEM_TYPE(tensor) != QNN_TENSORMEMTYPE_RAW) {
    return false;
  }
  uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
  uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
  if (rank == 0 || dims == nullptr) {
    return false;
  }
  size_t size = 0;
  Qnn_ClientBuffer_t clientBuf = QNN_TENSOR_GET_CLIENT_BUF(tensor);
  return clientBuf.data != nullptr &&
         buffer::buffer_size(std::vector<size_t>(dims, dims + rank),
                             QNN_TENSOR_GET_DATA_TYPE(tensor), size) &&
         size == clientBuf.dataSize;
}

static bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
  if (nullptr == dst || nullptr == src || !validateTensorVersion(*src)) {
    QNN_ERROR("Received nullptr or invalid tensor version");
    return false;
  }

  /* Initialize the selected version before assigning tensor fields */
  dst->version = src->version;
  if (dst->version == QNN_TENSOR_VERSION_2) {
    dst->v2 = QNN_TENSOR_V2_INIT;
  }

  const char *tensorName = QNN_TENSOR_GET_NAME(src);
  if (tensorName != nullptr) {
    char *name = strdup(tensorName);
    if (name == nullptr) {
      return false;
    }
    QNN_TENSOR_SET_NAME(dst, name);
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

  Qnn_QuantizeParams_t srcParams = QNN_TENSOR_GET_QUANT_PARAMS(src);
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition = srcParams.encodingDefinition;
  qParams.quantizationEncoding = srcParams.quantizationEncoding;
  if (srcParams.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.scaleOffsetEncoding = srcParams.scaleOffsetEncoding;
  } else if (srcParams.quantizationEncoding ==
             QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    const Qnn_AxisScaleOffset_t &axisParams = srcParams.axisScaleOffsetEncoding;
    uint32_t rank = QNN_TENSOR_GET_RANK(src);
    uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(src);
    if (axisParams.axis < 0 || static_cast<uint32_t>(axisParams.axis) >= rank ||
        dims == nullptr || axisParams.numScaleOffsets == 0 ||
        axisParams.numScaleOffsets != dims[axisParams.axis] ||
        axisParams.scaleOffset == nullptr) {
      QNN_ERROR("invalid per-axis quantization parameters");
      return false;
    }
    qParams.axisScaleOffsetEncoding.axis = axisParams.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets =
        axisParams.numScaleOffsets;
    qParams.axisScaleOffsetEncoding.scaleOffset =
        static_cast<Qnn_ScaleOffset_t *>(
            calloc(axisParams.numScaleOffsets, sizeof(Qnn_ScaleOffset_t)));
    if (qParams.axisScaleOffsetEncoding.scaleOffset == nullptr) {
      return false;
    }
    memcpy(qParams.axisScaleOffsetEncoding.scaleOffset, axisParams.scaleOffset,
           axisParams.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
  } else if (srcParams.quantizationEncoding !=
             QNN_QUANTIZATION_ENCODING_UNDEFINED) {
    QNN_ERROR("unsupported quantization encoding");
    return false;
  }
  QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);

  uint32_t rank = QNN_TENSOR_GET_RANK(src);
  QNN_TENSOR_SET_RANK(dst, rank);
  if (rank > 0) {
    if (QNN_TENSOR_GET_DIMENSIONS(src) == nullptr) {
      return false;
    }
    uint32_t *dims = static_cast<uint32_t *>(calloc(rank, sizeof(uint32_t)));
    if (dims == nullptr) {
      return false;
    }
    QNN_TENSOR_SET_DIMENSIONS(dst, dims);
    memcpy(dims, QNN_TENSOR_GET_DIMENSIONS(src), rank * sizeof(uint32_t));

    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src) != nullptr) {
      uint8_t *dynamicDims =
          static_cast<uint8_t *>(calloc(rank, sizeof(uint8_t)));
      if (dynamicDims == nullptr) {
        return false;
      }
      QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(dst, dynamicDims);
      memcpy(dynamicDims, QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src), rank);
    }
  }
  QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
  return true;
}

bool tensor::get_quant_params(Qnn_Tensor_t *tensor, QuantParams &params) {
  if (tensor == nullptr) {
    QNN_ERROR("get_quant_params: tensor is nullptr");
    return false;
  }

  Qnn_QuantizeParams_t q = QNN_TENSOR_GET_QUANT_PARAMS(*tensor);
  if (q.encodingDefinition == QNN_DEFINITION_DEFINED &&
      q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
      std::isfinite(q.scaleOffsetEncoding.scale) &&
      q.scaleOffsetEncoding.scale > 0.0f) {
    params = {q.scaleOffsetEncoding.scale, q.scaleOffsetEncoding.offset};
    return true;
  }

  QNN_ERROR("get_quant_params: unsupported/missing quant encoding");
  return false;
}

bool tensor::setup_tensors(Qnn_Tensor_t **tensors, uint32_t tensor_count,
                           Qnn_Tensor_t *tensor_wrappers) {
  if (tensors == nullptr) {
    QNN_ERROR("setupTensors output pointer null");
    return false;
  }
  *tensors = nullptr;
  if (tensor_wrappers == nullptr || tensor_count == 0) {
    QNN_ERROR("setupTensors parameters null");
    return false;
  }

  Qnn_Tensor_t *newTensors = new (std::nothrow) Qnn_Tensor_t[tensor_count];
  if (newTensors == nullptr) {
    return false;
  }
  for (uint32_t t = 0; t < tensor_count; t++) newTensors[t] = QNN_TENSOR_INIT;

  for (size_t tensorIdx = 0; tensorIdx < tensor_count; tensorIdx++) {
    Qnn_Tensor_t *tensor = newTensors + tensorIdx;
    if (!deepCopyQnnTensorInfo(tensor, tensor_wrappers + tensorIdx)) {
      free_tensors(newTensors, tensor_count);
      return false;
    }
    QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_RAW);

    std::vector<size_t> dims;
    uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    uint32_t *tensorDims = QNN_TENSOR_GET_DIMENSIONS(tensor);
    for (size_t r = 0; r < rank; r++) dims.push_back(tensorDims[r]);

    /* Setup client buffer only after validating its size */
    size_t length = 0;
    uint8_t *data = nullptr;
    Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);
    if (!buffer::buffer_size(dims, dataType, length) ||
        !buffer::allocate_buffer(&data, dims, dataType)) {
      free_tensors(newTensors, tensor_count);
      return false;
    }

    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
    clientBuffer.data = data;
    clientBuffer.dataSize = static_cast<uint32_t>(length);
    QNN_TENSOR_SET_CLIENT_BUF(tensor, clientBuffer);
  }

  *tensors = newTensors;
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
  if (!valid_client_buffer(*tensor)) {
    QNN_ERROR("fill_tensor: invalid client buffer or tensor dimensions");
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
  if (tensors == nullptr && tensor_count != 0) {
    QNN_ERROR("fill tensors are nullptr");
    return false;
  }

  if (input_data.size() != tensor_count) {
    QNN_ERROR("input buffer count does not match (%zu != %u)",
              input_data.size(), tensor_count);
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

bool tensor::read_tensor(Qnn_Tensor_t *tensor,
                         std::vector<uint8_t> &output_data) {
  output_data.clear();
  if (tensor == nullptr) {
    QNN_ERROR("read_tensor: tensor is nullptr");
    return false;
  }

  /* Get allocated client buffer */
  Qnn_ClientBuffer_t clientBuf = QNN_TENSOR_GET_CLIENT_BUF(*tensor);
  if (!valid_client_buffer(*tensor)) {
    QNN_ERROR("read_tensor: invalid client buffer or tensor dimensions");
    return false;
  }

  /* Get tensor data type */
  Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);

  /* Perform copy operation */
  if (dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
      dataType == QNN_DATATYPE_UINT_8) {

    /* Resize destination to match tensor buffer */
    output_data.resize(clientBuf.dataSize);

    /* Direct output mapping */
    memcpy(output_data.data(), reinterpret_cast<uint8_t *>(clientBuf.data),
           clientBuf.dataSize);
    return true;

  } else {
    QNN_ERROR("read_tensor: unsupported tensor output type!");
    return false;
  }
}

bool tensor::read_tensor(Qnn_Tensor_t *tensor,
                         std::vector<float> &output_data) {
  output_data.clear();
  if (tensor == nullptr) {
    QNN_ERROR("read_tensor: tensor is nullptr");
    return false;
  }

  Qnn_ClientBuffer_t clientBuf = QNN_TENSOR_GET_CLIENT_BUF(*tensor);
  if (!valid_client_buffer(*tensor)) {
    QNN_ERROR("read_tensor: invalid client buffer or tensor dimensions");
    return false;
  }

  Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);

  if (dataType == QNN_DATATYPE_FLOAT_32) {
    /* Already float, direct copy */
    if (clientBuf.dataSize % sizeof(float) != 0) {
      QNN_ERROR("read_tensor: invalid float buffer size");
      return false;
    }
    size_t count = clientBuf.dataSize / sizeof(float);
    output_data.resize(count);
    memcpy(output_data.data(), clientBuf.data, clientBuf.dataSize);
    return true;

  } else if (dataType == QNN_DATATYPE_UINT_8) {
    /* Plain integers have no quantization scale or offset */
    const uint8_t *raw = static_cast<const uint8_t *>(clientBuf.data);
    output_data.assign(raw, raw + clientBuf.dataSize);
    return true;

  } else if (dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
             dataType == QNN_DATATYPE_SFIXED_POINT_8) {
    Qnn_QuantizeParams_t q = QNN_TENSOR_GET_QUANT_PARAMS(*tensor);
    QuantParams qp{};
    size_t stride = 1;
    bool perAxis =
        q.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
    if (perAxis) {
      const Qnn_AxisScaleOffset_t &axisParams = q.axisScaleOffsetEncoding;
      uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
      uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
      if (q.encodingDefinition != QNN_DEFINITION_DEFINED ||
          axisParams.axis < 0 ||
          static_cast<uint32_t>(axisParams.axis) >= rank || dims == nullptr ||
          axisParams.numScaleOffsets == 0 ||
          axisParams.numScaleOffsets != dims[axisParams.axis] ||
          axisParams.scaleOffset == nullptr) {
        QNN_ERROR("read_tensor: invalid per-axis quantization parameters");
        return false;
      }

      std::vector<size_t> tensorDims(dims, dims + rank);
      if (buffer::count_elements(tensorDims) != clientBuf.dataSize) {
        QNN_ERROR("read_tensor: dimensions do not match quantized buffer");
        return false;
      }
      for (uint32_t d = axisParams.axis + 1; d < rank; d++) stride *= dims[d];
      for (uint32_t c = 0; c < axisParams.numScaleOffsets; c++) {
        float scale = axisParams.scaleOffset[c].scale;
        if (!std::isfinite(scale) || scale <= 0.0f) {
          QNN_ERROR("read_tensor: invalid per-axis quantization scale");
          return false;
        }
      }
    } else if (!get_quant_params(tensor, qp)) {
      return false;
    }

    /* Dequantize using QNN's scale * (q + offset) convention */
    const uint8_t *raw = static_cast<const uint8_t *>(clientBuf.data);
    const int8_t *signedRaw = static_cast<const int8_t *>(clientBuf.data);
    size_t count = clientBuf.dataSize;
    output_data.resize(count);
    for (size_t i = 0; i < count; i++) {
      if (perAxis) {
        const Qnn_AxisScaleOffset_t &axisParams = q.axisScaleOffsetEncoding;
        size_t channel = (i / stride) % axisParams.numScaleOffsets;
        qp = {axisParams.scaleOffset[channel].scale,
              axisParams.scaleOffset[channel].offset};
      }
      int32_t value =
          dataType == QNN_DATATYPE_SFIXED_POINT_8 ? signedRaw[i] : raw[i];
      output_data[i] = qp.scale * (static_cast<double>(value) + qp.offset);
    }
    return true;

  } else {
    QNN_ERROR("read_tensor: unsupported tensor output type for dequant!");
    return false;
  }
}

bool tensor::read_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                          std::vector<std::vector<uint8_t>> &output_data) {
  output_data.clear();
  /* Check parameter inputs */
  if (tensors == nullptr && tensor_count != 0) {
    QNN_ERROR("read tensors are nullptr");
    return false;
  }

  /* Ensure destination has exactly one slot for each tensor */
  output_data.resize(tensor_count);

  /* Read each tensor output */
  for (uint32_t t = 0; t < tensor_count; ++t) {
    if (!read_tensor(tensors + t, output_data[t])) {
      output_data.clear();
      QNN_ERROR("read_tensors: failed on tensor %u", t);
      return false;
    }
  }

  return true;
}

bool tensor::read_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                          std::vector<std::vector<float>> &output_data) {
  output_data.clear();
  if (tensors == nullptr && tensor_count != 0) {
    QNN_ERROR("read tensors are nullptr");
    return false;
  }
  output_data.resize(tensor_count);

  for (uint32_t t = 0; t < tensor_count; ++t) {
    if (!read_tensor(tensors + t, output_data[t])) {
      output_data.clear();
      QNN_ERROR("read_tensors: failed on tensor %u", t);
      return false;
    }
  }
  return true;
}

bool tensor::free_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count) {
  if (tensors == nullptr) {
    return true;
  }

  for (size_t tensorIdx = 0; tensorIdx < tensor_count; tensorIdx++) {
    QNN_DEBUG("freeing resources for tensor: %zu", tensorIdx);

    free(const_cast<char *>(QNN_TENSOR_GET_NAME(tensors[tensorIdx])));
    free(QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx]));
    free(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensors[tensorIdx]));
    free(QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data);

    Qnn_QuantizeParams_t qParams =
        QNN_TENSOR_GET_QUANT_PARAMS(tensors[tensorIdx]);
    if (qParams.quantizationEncoding ==
        QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      free(qParams.axisScaleOffsetEncoding.scaleOffset);
    }
    tensors[tensorIdx] = QNN_TENSOR_INIT;
  }

  delete[] tensors;

  return true;
}
