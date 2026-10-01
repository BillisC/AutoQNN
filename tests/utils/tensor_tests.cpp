#include "test_suites.h"
#include "tensor.h"
#include "qnn_type_macros.hpp"
#include "stubs/allocation_stubs.h"

#include <cassert>
#include <cstring>

static Qnn_Tensor_t make_tensor(Qnn_DataType_t type, uint32_t *dims,
                                uint32_t rank) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  QNN_TENSOR_SET_NAME(tensor, "test");
  QNN_TENSOR_SET_DATA_TYPE(tensor, type);
  QNN_TENSOR_SET_RANK(tensor, rank);
  QNN_TENSOR_SET_DIMENSIONS(tensor, dims);
  return tensor;
}

static void test_allocations() {
  uint32_t dims[] = {2};
  const Qnn_DataType_t types[] = {
      QNN_DATATYPE_FLOAT_32,        QNN_DATATYPE_UINT_8,
      QNN_DATATYPE_UFIXED_POINT_8,  QNN_DATATYPE_UINT_16,
      QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UINT_32,
      QNN_DATATYPE_UFIXED_POINT_32, QNN_DATATYPE_UINT_64,
      QNN_DATATYPE_INT_8,           QNN_DATATYPE_SFIXED_POINT_8,
      QNN_DATATYPE_INT_16,          QNN_DATATYPE_SFIXED_POINT_16,
      QNN_DATATYPE_INT_32,          QNN_DATATYPE_SFIXED_POINT_32,
      QNN_DATATYPE_INT_64,          QNN_DATATYPE_BOOL_8};
  for (Qnn_DataType_t type : types) {
    Qnn_Tensor_t wrapper = make_tensor(type, dims, 1);
    Qnn_Tensor_t *owned = nullptr;
    assert(tensor::setup_tensors(&owned, 1, &wrapper));
    auto client = QNN_TENSOR_GET_CLIENT_BUF(owned[0]);
    assert(client.data != nullptr && client.dataSize > 0);
    for (uint32_t b = 0; b < client.dataSize; b++) {
      assert(static_cast<uint8_t *>(client.data)[b] == 0);
    }
    assert(QNN_TENSOR_GET_NAME(owned[0]) != QNN_TENSOR_GET_NAME(wrapper));
    assert(QNN_TENSOR_GET_DIMENSIONS(owned[0]) != dims);
    assert(tensor::free_tensors(owned, 1));
  }

  Qnn_Tensor_t wrapper = make_tensor(QNN_DATATYPE_UINT_8, dims, 1);
  Qnn_Tensor_t *owned = nullptr;
  allocation_stub::fail_name = true;
  assert(!tensor::setup_tensors(&owned, 1, &wrapper) && owned == nullptr);
  allocation_stub::fail_name = false;

  /* The second wrapper fails after the first has acquired its resources */
  Qnn_Tensor_t wrappers[] = {wrapper, wrapper};
  QNN_TENSOR_SET_DATA_TYPE(wrappers[1], QNN_DATATYPE_UNDEFINED);
  assert(!tensor::setup_tensors(&owned, 2, wrappers) && owned == nullptr);
  QNN_TENSOR_SET_DIMENSIONS(wrapper, nullptr);
  assert(!tensor::setup_tensors(&owned, 1, &wrapper) && owned == nullptr);
  assert(!tensor::setup_tensors(nullptr, 1, wrappers));
  assert(tensor::free_tensors(nullptr, 0));
}

static void test_quantization() {
  uint32_t dims[] = {2, 2, 2};
  uint8_t dynamicDims[] = {0, 1, 0};
  Qnn_Tensor_t wrapper = QNN_TENSOR_INIT;
  wrapper.version = QNN_TENSOR_VERSION_2;
  wrapper.v2 = QNN_TENSOR_V2_INIT;
  QNN_TENSOR_SET_NAME(wrapper, "axis");
  QNN_TENSOR_SET_DATA_TYPE(wrapper, QNN_DATATYPE_UFIXED_POINT_8);
  QNN_TENSOR_SET_RANK(wrapper, 3);
  QNN_TENSOR_SET_DIMENSIONS(wrapper, dims);
  QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(wrapper, dynamicDims);
  Qnn_ScaleOffset_t encodings[] = {{0.5f, -2}, {2.0f, -2}};
  Qnn_QuantizeParams_t q = QNN_QUANTIZE_PARAMS_INIT;
  q.encodingDefinition = QNN_DEFINITION_DEFINED;
  q.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  q.axisScaleOffsetEncoding = {1, 2, encodings};
  QNN_TENSOR_SET_QUANT_PARAMS(wrapper, q);

  /* Fail each metadata / client-buffer allocation, including a later tensor */
  Qnn_Tensor_t wrappers[] = {wrapper, wrapper};
  Qnn_Tensor_t *owned = nullptr;
  for (int budget = 0; budget < 8; budget++) {
    allocation_stub::calloc_budget = budget;
    assert(!tensor::setup_tensors(&owned, 2, wrappers) && owned == nullptr);
    allocation_stub::calloc_budget = -1;
  }
  assert(tensor::setup_tensors(&owned, 1, &wrapper));
  assert(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(owned[0]) != dynamicDims);
  assert(QNN_TENSOR_GET_QUANT_PARAMS(owned[0])
             .axisScaleOffsetEncoding.scaleOffset != encodings);
  assert(tensor::fill_tensor(owned, std::vector<uint8_t>(8, 10)));
  std::vector<float> output;
  assert(tensor::read_tensor(owned, output));
  assert(output == std::vector<float>({4, 4, 16, 16, 4, 4, 16, 16}));
  tensor::free_tensors(owned, 1);

  /* Reproduce the original two-element per-axis decoding error */
  wrapper = make_tensor(QNN_DATATYPE_UFIXED_POINT_8, dims, 1);
  q.axisScaleOffsetEncoding.axis = 0;
  QNN_TENSOR_SET_QUANT_PARAMS(wrapper, q);
  assert(tensor::setup_tensors(&owned, 1, &wrapper));
  assert(tensor::fill_tensor(owned, {10, 10}));
  assert(tensor::read_tensor(owned, output));
  assert(output == std::vector<float>({4, 16}));

  Qnn_Tensor_t invalid = owned[0];
  q.axisScaleOffsetEncoding.axis = -1;
  QNN_TENSOR_SET_QUANT_PARAMS(invalid, q);
  assert(!tensor::read_tensor(&invalid, output) && output.empty());
  q.axisScaleOffsetEncoding.axis = 0;
  q.axisScaleOffsetEncoding.numScaleOffsets = 1;
  QNN_TENSOR_SET_QUANT_PARAMS(invalid, q);
  assert(!tensor::read_tensor(&invalid, output) && output.empty());

  Qnn_QuantizeParams_t scalar = QNN_QUANTIZE_PARAMS_INIT;
  scalar.encodingDefinition = QNN_DEFINITION_DEFINED;
  scalar.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  scalar.scaleOffsetEncoding = {0.5f, -2};
  /* Release the owned per-axis metadata before changing its encoding */
  tensor::free_tensors(owned, 1);
  QNN_TENSOR_SET_QUANT_PARAMS(wrapper, scalar);
  assert(tensor::setup_tensors(&owned, 1, &wrapper));
  assert(tensor::fill_tensor(owned, {10, 10}));
  assert(tensor::read_tensor(owned, output));
  assert(output == std::vector<float>({4, 4}));
  scalar.scaleOffsetEncoding.scale = 0;
  QNN_TENSOR_SET_QUANT_PARAMS(owned[0], scalar);
  assert(!tensor::read_tensor(owned, output) && output.empty());
  scalar.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  QNN_TENSOR_SET_QUANT_PARAMS(owned[0], scalar);
  assert(!tensor::read_tensor(owned, output) && output.empty());
  scalar.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  QNN_TENSOR_SET_QUANT_PARAMS(owned[0], scalar);
  assert(!tensor::read_tensor(owned, output) && output.empty());
  tensor::free_tensors(owned, 1);
  QNN_TENSOR_SET_QUANT_PARAMS(wrapper, scalar);
  assert(!tensor::setup_tensors(&owned, 1, &wrapper) && owned == nullptr);

  wrapper = make_tensor(QNN_DATATYPE_FLOAT_32, dims, 1);
  assert(tensor::setup_tensors(&owned, 1, &wrapper));
  float values[] = {1.25f, -2.5f};
  memcpy(QNN_TENSOR_GET_CLIENT_BUF(owned[0]).data, values, sizeof(values));
  assert(tensor::read_tensor(owned, output));
  assert(output == std::vector<float>({1.25f, -2.5f}));
  auto client = QNN_TENSOR_GET_CLIENT_BUF(owned[0]);
  client.dataSize--;
  QNN_TENSOR_SET_CLIENT_BUF(owned[0], client);
  assert(!tensor::read_tensor(owned, output) && output.empty());
  tensor::free_tensors(owned, 1);

  wrapper = make_tensor(QNN_DATATYPE_SFIXED_POINT_8, dims, 1);
  scalar.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  scalar.scaleOffsetEncoding = {0.5f, -2};
  QNN_TENSOR_SET_QUANT_PARAMS(wrapper, scalar);
  assert(tensor::setup_tensors(&owned, 1, &wrapper));
  int8_t signedValues[] = {-10, 10};
  memcpy(QNN_TENSOR_GET_CLIENT_BUF(owned[0]).data, signedValues,
         sizeof(signedValues));
  assert(tensor::read_tensor(owned, output));
  assert(output == std::vector<float>({-6, 4}));
  tensor::free_tensors(owned, 1);
}

void test_tensor() {
  test_allocations();
  test_quantization();
}
