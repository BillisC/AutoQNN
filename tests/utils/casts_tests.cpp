#include "test_suites.h"
#include "casts.h"

#include <cassert>
#include <limits>

void test_casts() {
  uint8_t quantized[4]{};
  assert(casts::floatToTfN(quantized, {1.0f, -2.0f, 5.0f, 0.0f}, 0, 0.01f, 4));
  assert(quantized[0] == 100 && quantized[1] == 0 && quantized[2] == 255);
  assert(casts::floatToTfN(quantized, {1.0f}, -2, 0.5f, 1));
  assert(quantized[0] == 4);
  assert(!casts::floatToTfN(quantized, {1.0f}, 0, 0.0f, 1));
  assert(!casts::floatToTfN(quantized, {1.0f}, 0, -1.0f, 1));
  assert(!casts::floatToTfN(quantized, {1.0f}, 0, 1.0f, 2));
  assert(!casts::floatToTfN(
      quantized, {std::numeric_limits<float>::quiet_NaN()}, 0, 1.0f, 1));
  uint16_t wide = 0;
  assert(casts::floatToTfN(&wide, {1000.0f}, 0, 0.01f, 1));
  assert(wide == 65535);
  uint32_t wider = 0;
  assert(casts::floatToTfN(&wider, {1.0e20f}, 0, 0.01f, 1));
  assert(wider == std::numeric_limits<uint32_t>::max());
}
