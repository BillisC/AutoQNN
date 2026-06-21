/**
 * @file    buffer.h
 * @brief   Buffer functions header
 *
 * This header contains forward declarations for the management of QNN buffers,
 * under the buffer namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#ifndef BUFFER_H
#define BUFFER_H

#include <vector>

#include "QnnTypes.h"

namespace buffer {

/**
 * @brief Allocate memory for tensor buffer of type Qnn_Datatype_t.
 *
 * @param[out] buffer Pointer to the allocated buffer
 * @param[in] dims Tensor dims
 * @param[in] dataType Tensor buffer data type
 * @return true on success
 */
bool allocate_buffer(uint8_t **buffer, const std::vector<size_t> dims,
                     Qnn_DataType_t dataType);

/**
 * @brief Count buffer elements.
 *
 * @param[in] dims Number of buffer dimensions
 * @param[in] dataType Tensor dims
 * @return number of elements
 */
size_t count_elements(const std::vector<size_t> &dims);

} // namespace buffer

#endif