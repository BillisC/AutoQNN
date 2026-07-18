/**
 * @file    tensor.h
 * @brief   Tensor function definitions
 *
 * This header contains forward declarations for the management of QNN tensors,
 * under the tensor namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

#include "QnnTypes.h"

namespace tensor {

/**
 * @brief Initialize a copy of the tensors from the model.
 *
 * @param[out] tensors Pointer for the new copy of tensors
 * @param[in] tensor_count Number of tensors
 * @param[in] tensor_wrappers Pointer to the model's tensor structure
 * @return true on success
 */
bool setup_tensors(Qnn_Tensor_t **tensors, uint32_t tensor_count,
                   Qnn_Tensor_t *tensor_wrappers);

/**
 * @brief Fill a single tensor with input data.
 *
 * @param[in] tensor Pointer to tensor
 * @param[in] input_data Pointer to the input's data
 * @return true on success
 */
bool fill_tensor(Qnn_Tensor_t *tensor, const std::vector<uint8_t> &input_data);

/**
 * @brief Fill multiple tensors with input data.
 *
 * @param[in] tensors Pointer to array of tensors
 * @param[in] tensor_count Number of tensors
 * @param[in] input_data Pointer to the input's data
 * @return true on success
 */
bool fill_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                  std::vector<std::vector<uint8_t>> &input_data);

/**
 * @brief Fully deallocate tensor array.
 *
 * @param[in] tensors Pointer to array of tensors
 * @param[in] tensor_count Number of tensors
 * @return true on success
 */
bool free_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count);

} // namespace tensor

#endif