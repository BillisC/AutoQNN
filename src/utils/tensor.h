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
 * @brief Scale/offset for dequantizing a QNN tensor.
 */
struct QuantParams {
  float scale;
  int32_t offset;
};

/**
 * @brief Extract scale/offset quantization parameters from a tensor.
 *
 * @param[in] tensor Pointer to tensor
 * @param[out] params Scale/offset parameters
 * @return true for valid per-tensor scale/offset quantization
 */
bool get_quant_params(Qnn_Tensor_t *tensor, QuantParams &params);

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
 * @note Bytes must already match the tensor's layout and quantization.
 *
 * @param[in] tensor Pointer to tensor
 * @param[in] input_data Vector with the input's data
 * @return true on success
 */
bool fill_tensor(Qnn_Tensor_t *tensor, const std::vector<uint8_t> &input_data);

/**
 * @brief Fill multiple tensors with input data.
 *
 * @param[in] tensors Pointer to array of tensors
 * @param[in] tensor_count Number of tensors
 * @param[in] input_data Vector with the inputs' data
 * @return true on success
 */
bool fill_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                  std::vector<std::vector<uint8_t>> &input_data);

/**
 * @brief Read a single tensor's output data into a raw uint8 buffer.
 *
 * @param[in] tensor Pointer to tensor
 * @param[out] output_data Vector to receive the tensor's data
 * @return true on success
 */
bool read_tensor(Qnn_Tensor_t *tensor, std::vector<uint8_t> &output_data);

/**
 * @brief Read a single tensor's output data into a float buffer.
 * @note If the output type is fixed-point 8-bit, dequantization is applied.
 * @param[in] tensor Pointer to tensor
 * @param[out] output_data Vector to receive the tensor's data
 * @return true on success
 */
bool read_tensor(Qnn_Tensor_t *tensor, std::vector<float> &output_data);

/**
 * @brief Read multiple tensors' output data into raw uint8 buffers.
 * @param[in] tensors Pointer to array of tensors
 * @param[in] tensor_count Number of tensors
 * @param[out] output_data Vector to receive each tensor's data
 * @return true on success
 */
bool read_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                  std::vector<std::vector<uint8_t>> &output_data);

/**
 * @brief Read multiple tensors' output data into float buffers.
 * @note If the output type is fixed-point 8-bit, dequantization is applied.
 * @param[in] tensors Pointer to array of tensors
 * @param[in] tensor_count Number of tensors
 * @param[out] output_data Vector to receive each tensor's data
 * @return true on success
 */
bool read_tensors(Qnn_Tensor_t *tensors, uint32_t tensor_count,
                  std::vector<std::vector<float>> &output_data);

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
