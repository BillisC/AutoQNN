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
 * @note Release the allocated buffer with free().
 *
 * @param[out] buffer Pointer to the allocated buffer
 * @param[in] dims Tensor dims
 * @param[in] dataType Tensor buffer data type
 * @return true on success
 */
bool allocate_buffer(uint8_t **buffer, const std::vector<size_t> dims,
                     Qnn_DataType_t dataType);

/**
 * @brief Calculate a buffer size that fits QNN's client buffer.
 *
 * @param[in] dims Tensor dims
 * @param[in] dataType Tensor buffer data type
 * @param[out] size Buffer size in bytes
 * @return true on success
 */
bool buffer_size(const std::vector<size_t> &dims, Qnn_DataType_t dataType,
                 size_t &size);

/**
 * @brief Count buffer elements.
 *
 * @param[in] dims Tensor dims
 * @return number of elements, or zero for zero dimensions / overflow
 */
size_t count_elements(const std::vector<size_t> &dims);

/**
 * @brief Reorder an 8-bit buffer using an explicit axis permutation.
 *
 * @param[in] input_data Dense input buffer
 * @param[in] dims Input dimensions
 * @param[in] order Input axis for each output axis
 * @param[out] output_data Reordered buffer, or empty on failure
 * @return true on success
 */
bool reorder_buffer(const std::vector<uint8_t> &input_data,
                    const std::vector<size_t> &dims,
                    const std::vector<size_t> &order,
                    std::vector<uint8_t> &output_data);

} // namespace buffer

#endif
