#include "test_suites.h"
#include "model.h"
#include "buffer.h"
#include "mocks/qnn_mock.h"
#include "stubs/allocation_stubs.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

void test_model(const char *model_path, const char *missing_symbols_path) {
  using Result = model::Model::Result;
  QNN_INTERFACE_VER_TYPE api{};
  api.contextCreate = qnn_mock::create_context;
  api.contextFree = qnn_mock::free_context;
  api.graphFinalize = qnn_mock::finalize_graph;
  api.graphExecute = qnn_mock::execute_graph;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;

  {
    model::Model invalid(nullptr, nullptr, nullptr, nullptr);
    assert(invalid.load_model_lib(model_path) == Result::PARAMS_ERROR);
    assert(invalid.execute() == Result::SETUP_ERROR);
  }
  {
    model::Model model(&backend, &device, nullptr, &api);
    std::vector<uint8_t> input{10, 20};
    std::vector<float> output{99};
    assert(model.execute() == Result::SETUP_ERROR);
    assert(model.fill_input(input) == Result::SETUP_ERROR);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());
    qnn_mock::fail_context = true;
    assert(model.load_model_lib(model_path) == Result::CONTEXT_ERROR);
    qnn_mock::fail_context = false;
    assert(model.load_model_lib("/nonexistent/autoqnn-test.so") ==
           Result::FILE_ERROR);
    assert(qnn_mock::created == qnn_mock::freed);
    assert(model.load_model_lib(missing_symbols_path) == Result::FILE_ERROR);
    assert(qnn_mock::created == qnn_mock::freed);
    for (const char *mode : {"compose_error", "empty", "tensor_error"}) {
      setenv("AUTOQNN_TEST_MODEL_MODE", mode, 1);
      qnn_mock::finalized = 0;
      Result expected = strcmp(mode, "tensor_error") == 0 ? Result::TENSOR_ERROR
                                                          : Result::GRAPH_ERROR;
      assert(model.load_model_lib(model_path) == expected);
      assert(qnn_mock::created == qnn_mock::freed);
      assert(model.execute() == Result::SETUP_ERROR);
    }
    unsetenv("AUTOQNN_TEST_MODEL_MODE");
    qnn_mock::finalized = 0;
    qnn_mock::fail_finalize = true;
    assert(model.load_model_lib(model_path) == Result::GRAPH_ERROR);
    assert(qnn_mock::created == qnn_mock::freed);
    qnn_mock::fail_finalize = false;
    qnn_mock::finalized = 0;
    allocation_stub::calloc_budget = 0;
    assert(model.load_model_lib(model_path) == Result::TENSOR_ERROR);
    allocation_stub::calloc_budget = -1;
    assert(qnn_mock::created == qnn_mock::freed);
    assert(model.execute() == Result::SETUP_ERROR);
    qnn_mock::finalized = 0;
    assert(model.load_model_lib(model_path) == Result::OK);
    assert(qnn_mock::finalized == 2);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());
    assert(model.execute() == Result::SETUP_ERROR);
    std::vector<std::vector<std::vector<uint8_t>>> too_few_inputs;
    assert(model.fill_inputs(too_few_inputs) == Result::INVALID_ARGUMENT_ERROR);
    assert(model.fill_input(input) == Result::INVALID_ARGUMENT_ERROR);
    std::vector<std::vector<std::vector<uint8_t>>> inputs{{{10, 20}},
                                                          {{30, 40}}};
    assert(model.fill_inputs(inputs) == Result::OK);
    qnn_mock::fail_execute = true;
    assert(model.execute() == Result::GRAPH_ERROR);
    qnn_mock::fail_execute = false;
    qnn_mock::executed.clear();
    assert(model.execute() == Result::OK);
    assert(qnn_mock::executed == std::vector<uintptr_t>({1, 2}));
    assert(model.output(output) == Result::INVALID_ARGUMENT_ERROR);
    std::vector<std::vector<std::vector<float>>> outputs(3, {{99}, {99}});
    assert(model.outputs(outputs) == Result::OK);
    assert(outputs == std::vector<std::vector<std::vector<float>>>(
                          {{{10, 20}}, {{30, 40}}}));
    std::vector<std::vector<std::vector<uint8_t>>> raw_outputs(3, {{99}, {99}});
    assert(model.outputs(raw_outputs) == Result::OK && raw_outputs == inputs);
    qnn_mock::fail_graph = 2;
    assert(model.execute() == Result::GRAPH_ERROR);
    assert(model.outputs(outputs) == Result::SETUP_ERROR && outputs.empty());
    assert(model.outputs(raw_outputs) == Result::SETUP_ERROR &&
           raw_outputs.empty());
    qnn_mock::fail_graph = 0;
    qnn_mock::invalid_output_graph = 2;
    assert(model.execute() == Result::OK);
    assert(model.outputs(outputs) == Result::TENSOR_ERROR && outputs.empty());
    assert(model.outputs(raw_outputs) == Result::TENSOR_ERROR &&
           raw_outputs.empty());
    qnn_mock::invalid_output_graph = 0;
    assert(model.execute() == Result::OK);
    inputs[1].push_back({50, 60});
    assert(model.fill_inputs(inputs) == Result::TENSOR_ERROR);
    assert(model.execute() == Result::SETUP_ERROR);
    assert(model.outputs(outputs) == Result::SETUP_ERROR && outputs.empty());

    /* The single-tensor API requires fresh, successfully qnn_mock::executed
     * input */
    setenv("AUTOQNN_TEST_MODEL_MODE", "single", 1);
    qnn_mock::finalized = 0;
    assert(model.load_model_lib(model_path) == Result::OK);
    assert(model.fill_input(input) == Result::OK);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());
    assert(model.execute() == Result::OK);
    assert(model.output(output) == Result::OK);
    assert(output == std::vector<float>({10, 20}));
    std::vector<uint8_t> raw_output;
    assert(model.output(raw_output) == Result::OK && raw_output == input);
    qnn_mock::fail_execute = true;
    assert(model.execute() == Result::GRAPH_ERROR);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());
    assert(model.output(raw_output) == Result::SETUP_ERROR &&
           raw_output.empty());
    qnn_mock::fail_execute = false;
    assert(model.execute() == Result::OK);
    assert(model.fill_input(input) == Result::OK);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());
    assert(model.execute() == Result::OK);
    input.pop_back();
    assert(model.fill_input(input) == Result::TENSOR_ERROR);
    assert(model.execute() == Result::SETUP_ERROR);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());

    /* Quantized image IO must match shape, type, scale and offset */
    setenv("AUTOQNN_TEST_MODEL_MODE", "image", 1);
    qnn_mock::finalized = 0;
    assert(model.load_model_lib(model_path) == Result::OK);
    model::Model::TensorSpec imageSpec = {
        {1, 2, 3, 2}, QNN_DATATYPE_UFIXED_POINT_8, 1.0f / 128.0f, -127};
    model::Model::TensorSpec outputSpec = {
        {1, 2}, QNN_DATATYPE_UFIXED_POINT_8, 1.0f / 256.0f, 0};
    assert(model.validate_io(imageSpec, outputSpec) == Result::OK);
    auto invalid = imageSpec;
    invalid.dimensions = {1, 2, 2, 3};
    assert(model.validate_io(invalid, outputSpec) == Result::TENSOR_ERROR);
    invalid = imageSpec;
    invalid.data_type = QNN_DATATYPE_UINT_8;
    assert(model.validate_io(invalid, outputSpec) == Result::TENSOR_ERROR);
    invalid = imageSpec;
    invalid.scale = 1.0f;
    assert(model.validate_io(invalid, outputSpec) == Result::TENSOR_ERROR);
    invalid = imageSpec;
    invalid.offset = 0;
    assert(model.validate_io(invalid, outputSpec) == Result::TENSOR_ERROR);
    auto invalid_output = outputSpec;
    invalid_output.dimensions = {1, 3};
    assert(model.validate_io(imageSpec, invalid_output) ==
           Result::TENSOR_ERROR);
    std::vector<uint8_t> rgb{0, 127, 255, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    assert(buffer::reorder_buffer(rgb, {1, 2, 2, 3}, {0, 2, 3, 1}, input));
    assert(model.fill_input(input) == Result::OK);
    assert(model.execute() == Result::OK && qnn_mock::last_input == input);
    assert(model.output(output) == Result::OK);
    assert(output == std::vector<float>({0.0f, 6.0f / 256.0f}));
    assert(model.load_model_lib(nullptr) == Result::PARAMS_ERROR);
    assert(model.execute() == Result::SETUP_ERROR);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());

    /* Reload releases the earlier context and buffers */
    qnn_mock::finalized = 0;
    assert(model.load_model_lib(model_path) == Result::OK);
    assert(qnn_mock::created == qnn_mock::freed + 1);
    assert(model.execute() == Result::SETUP_ERROR);
    assert(model.output(output) == Result::SETUP_ERROR && output.empty());
    unsetenv("AUTOQNN_TEST_MODEL_MODE");
  }
  assert(qnn_mock::created == qnn_mock::freed);
}
