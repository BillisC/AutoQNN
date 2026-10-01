#include "test_suites.h"
#include "buffer.h"

#include <cassert>
#include <limits>

static void test_reorder() {
  std::vector<uint8_t> rgb{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<uint8_t> input;
  assert(buffer::reorder_buffer(rgb, {1, 2, 2, 3}, {0, 2, 3, 1}, input));
  assert(input == std::vector<uint8_t>({0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11}));
  std::vector<uint8_t> restored;
  assert(buffer::reorder_buffer(input, {1, 2, 3, 2}, {0, 3, 1, 2}, restored));
  assert(restored == rgb);

  /* Non-square dimensions expose accidental height/width swapping */
  rgb.resize(24);
  for (size_t i = 0; i < rgb.size(); i++) rgb[i] = i;
  assert(buffer::reorder_buffer(rgb, {1, 2, 4, 3}, {0, 2, 3, 1}, input));
  assert(buffer::reorder_buffer(input, {1, 4, 3, 2}, {0, 3, 1, 2}, restored));
  assert(restored == rgb);
  assert(!buffer::reorder_buffer(rgb, {1, 2, 4, 3}, {0, 2, 2, 1}, input));
  assert(input.empty());
  assert(!buffer::reorder_buffer(rgb, {1, 2, 4, 3}, {0, 2, 4, 1}, input));
  assert(!buffer::reorder_buffer(rgb, {1, 2, 2, 3}, {0, 2, 3, 1}, input));
  assert(!buffer::reorder_buffer(rgb, {}, {}, input));
}

static void test_allocations() {
  assert(buffer::count_elements({65536, 65536}) == size_t{4294967296ULL});
  assert(buffer::count_elements({std::numeric_limits<size_t>::max(), 2}) == 0);
  assert(buffer::count_elements({2, 0}) == 0);
  uint8_t *data = nullptr;
  assert(!buffer::allocate_buffer(&data, {65536, 65536}, QNN_DATATYPE_UINT_8));
  assert(data == nullptr);
  assert(!buffer::allocate_buffer(&data, {std::numeric_limits<uint32_t>::max()},
                                  QNN_DATATYPE_FLOAT_32));
  assert(!buffer::allocate_buffer(nullptr, {2}, QNN_DATATYPE_UINT_8));
}

void test_buffer() {
  test_reorder();
  test_allocations();
}
