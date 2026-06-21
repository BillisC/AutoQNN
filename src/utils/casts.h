/**
 * @file    casts.h
 * @brief   Cast functions header
 *
 * This header contains templates for the conversion (quantization) of input
 * buffers, under the casts namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#ifndef CASTS_H
#define CASTS_H

#include <vector>
#include <math.h>

namespace casts {

/**
 * @brief Convert float input to quantized type.
 *
 * @tparam T Unsigned integer type for the quantized output
 * @param[out] out Output buffer of type T
 * @param[in] in Input vector of type float
 * @param[in] offset Quantization offset
 * @param[in] scale Quantization scale
 * @param[in] n_elements Buffer element count
 * @return true on success
 */
template <typename T>
static bool floatToTfN(T *out, const std::vector<float> &in, int32_t offset,
                       float scale, size_t n_elements) {
  /* Parameter checks */
  static_assert(std::is_unsigned<T>::value,
                "floatToTfN supports unsigned only!");

  if (nullptr == out || in.empty()) {
    return false;
  }

  /* Quantize all elements */
  size_t dataTypeSizeInBytes = sizeof(T);
  size_t bitWidth = dataTypeSizeInBytes * sizeof(uint8_t);
  double trueBitWidthMax = pow(2, bitWidth) - 1;
  double encodingMin = offset * scale;
  double encodingMax = (trueBitWidthMax + offset) * scale;
  double encodingRange = encodingMax - encodingMin;

  for (size_t i = 0; i < n_elements; ++i) {
    int quantizedValue =
        round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);
    if (quantizedValue < 0)
      quantizedValue = 0;
    else if (quantizedValue > (int)trueBitWidthMax)
      quantizedValue = (int)trueBitWidthMax;
    out[i] = static_cast<T>(quantizedValue);
  }
  return true;
}

} // namespace casts

#endif