/**
 * @file    buffer.cpp
 * @brief   Buffer function definitions
 *
 * This file contains templates and function definitions for QNN
 * buffer management, under the buffer namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include "buffer.h"

#include <numeric>
#include <vector>

#include "QnnTypes.h"
#include "logger.hpp"

namespace buffer {

template <typename T>
static void allocate_buffer(T **buffer, size_t &n_elements) {
  *buffer = new T[n_elements]{};
}

bool allocate_buffer(uint8_t **buffer, const std::vector<size_t> dims,
                     Qnn_DataType_t dataType) {
  if (dims.empty()) {
    QNN_ERROR("allocation dims are empty");
    return false;
  }

  size_t n_element = count_elements(dims);

  switch (dataType) {
    case QNN_DATATYPE_FLOAT_32:
      QNN_DEBUG("allocating float buffer");
      allocate_buffer<float>(reinterpret_cast<float **>(buffer), n_element);
      break;

    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      QNN_DEBUG("allocating uint8_t buffer");
      allocate_buffer<uint8_t>(reinterpret_cast<uint8_t **>(buffer), n_element);
      break;

    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      QNN_DEBUG("allocating uint16_t buffer");
      allocate_buffer<uint16_t>(reinterpret_cast<uint16_t **>(buffer),
                                n_element);
      break;

    case QNN_DATATYPE_UINT_32:
      QNN_DEBUG("allocating uint32_t buffer");
      allocate_buffer<uint32_t>(reinterpret_cast<uint32_t **>(buffer),
                                n_element);
      break;

    case QNN_DATATYPE_UINT_64:
      QNN_DEBUG("allocating uint64_t buffer");
      allocate_buffer<uint64_t>(reinterpret_cast<uint64_t **>(buffer),
                                n_element);
      break;

    case QNN_DATATYPE_INT_8:
      QNN_DEBUG("allocating int8_t buffer");
      allocate_buffer<int8_t>(reinterpret_cast<int8_t **>(buffer), n_element);
      break;

    case QNN_DATATYPE_INT_16:
      QNN_DEBUG("allocating int16_t buffer");
      allocate_buffer<int16_t>(reinterpret_cast<int16_t **>(buffer), n_element);
      break;

    case QNN_DATATYPE_INT_32:
      QNN_DEBUG("allocating int32_t buffer");
      allocate_buffer<int32_t>(reinterpret_cast<int32_t **>(buffer), n_element);
      break;

    case QNN_DATATYPE_INT_64:
      QNN_DEBUG("allocating int64_t buffer");
      allocate_buffer<int64_t>(reinterpret_cast<int64_t **>(buffer), n_element);
      break;

    case QNN_DATATYPE_BOOL_8:
      QNN_DEBUG("allocating bool buffer");
      allocate_buffer<uint8_t>(reinterpret_cast<uint8_t **>(buffer), n_element);
      break;

    default: QNN_ERROR("Datatype not supported yet!"); return false;
  }

  if (buffer == nullptr) {
    QNN_ERROR("failed to allocate buffer");
    return false;
  }

  return true;
}

size_t count_elements(const std::vector<size_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<size_t>());
}

} // namespace buffer