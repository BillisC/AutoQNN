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
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

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

  if (nullptr == out || in.empty() || n_elements > in.size() ||
      !std::isfinite(scale) || scale <= 0.0f) {
    return false;
  }

  for (size_t i = 0; i < n_elements; ++i) {
    if (!std::isfinite(in[i])) {
      return false;
    }
  }

  /* Quantize all elements using QNN's scale * (q + offset) convention */
  double max_value = static_cast<double>(std::numeric_limits<T>::max());
  for (size_t i = 0; i < n_elements; ++i) {
    double quantizedValue =
        std::round(static_cast<double>(in[i]) / scale - offset);
    if (quantizedValue <= 0.0)
      out[i] = 0;
    else if (quantizedValue >= max_value)
      out[i] = std::numeric_limits<T>::max();
    else
      out[i] = static_cast<T>(quantizedValue);
  }
  return true;
}

} // namespace casts

#endif
