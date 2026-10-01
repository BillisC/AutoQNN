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

#include <cstdlib>
#include <limits>
#include <vector>

#include "QnnTypes.h"
#include "logger.hpp"

namespace buffer {

bool buffer_size(const std::vector<size_t> &dims, Qnn_DataType_t dataType,
                 size_t &size) {
  size = 0;
  if (dims.empty()) {
    QNN_ERROR("allocation dims are empty");
    return false;
  }

  size_t element_size = 0;
  switch (dataType) {
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8: element_size = 1; break;

    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: element_size = 2; break;

    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32: element_size = 4; break;

    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_INT_64: element_size = 8; break;

    default: QNN_ERROR("Datatype not supported yet!"); return false;
  }

  size_t n_elements = count_elements(dims);
  if (n_elements == 0 ||
      n_elements > std::numeric_limits<uint32_t>::max() / element_size) {
    QNN_ERROR("tensor buffer size is invalid or too large");
    return false;
  }

  size = n_elements * element_size;
  return true;
}

bool allocate_buffer(uint8_t **buffer, const std::vector<size_t> dims,
                     Qnn_DataType_t dataType) {
  if (buffer == nullptr) {
    QNN_ERROR("buffer pointer is null");
    return false;
  }
  *buffer = nullptr;

  size_t size = 0;
  if (!buffer_size(dims, dataType, size)) {
    return false;
  }

  /* Allocate aligned, zero-initialized storage for every data type */
  *buffer = static_cast<uint8_t *>(calloc(size, 1));
  if (*buffer == nullptr) {
    QNN_ERROR("failed to allocate buffer");
    return false;
  }

  return true;
}

size_t count_elements(const std::vector<size_t> &dims) {
  size_t count = 1;
  for (size_t dim : dims) {
    if (dim == 0 || count > std::numeric_limits<size_t>::max() / dim) {
      QNN_ERROR("tensor element count is invalid or too large");
      return 0;
    }
    count *= dim;
  }
  return count;
}

bool reorder_buffer(const std::vector<uint8_t> &input_data,
                    const std::vector<size_t> &dims,
                    const std::vector<size_t> &order,
                    std::vector<uint8_t> &output_data) {
  size_t count = count_elements(dims);
  if (dims.empty() || order.size() != dims.size() || count == 0 ||
      input_data.size() != count) {
    output_data.clear();
    return false;
  }

  std::vector<bool> used(dims.size(), false);
  for (size_t axis : order) {
    if (axis >= dims.size() || used[axis]) {
      output_data.clear();
      return false;
    }
    used[axis] = true;
  }

  /* Build strides for the original dense buffer */
  std::vector<size_t> strides(dims.size(), 1);
  for (size_t d = dims.size() - 1; d > 0; d--) {
    strides[d - 1] = strides[d] * dims[d];
  }

  std::vector<uint8_t> reordered(count);
  for (size_t i = 0; i < count; i++) {
    size_t index = i, source = 0;
    for (size_t d = order.size(); d > 0; d--) {
      size_t axis = order[d - 1];
      source += (index % dims[axis]) * strides[axis];
      index /= dims[axis];
    }
    reordered[i] = input_data[source];
  }
  output_data.swap(reordered);
  return true;
}

} // namespace buffer
