#include "qnn_mock.h"
#include "qnn_type_macros.hpp"

#include <cassert>
#include <cstring>

namespace qnn_mock {

uint32_t created = 0, freed = 0, finalized = 0;
bool fail_context = false, fail_finalize = false, fail_execute = false;
uintptr_t fail_graph = 0, invalid_output_graph = 0;
std::vector<uintptr_t> executed;
std::vector<uint8_t> last_input;

Qnn_ErrorHandle_t create_context(Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                 const QnnContext_Config_t **,
                                 Qnn_ContextHandle_t *context) {
  if (fail_context)
    return QNN_CONTEXT_ERROR_INVALID_ARGUMENT;
  *context = new uint8_t;
  created++;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t free_context(Qnn_ContextHandle_t context,
                               Qnn_ProfileHandle_t) {
  delete static_cast<uint8_t *>(context);
  freed++;
  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t finalize_graph(Qnn_GraphHandle_t graph, Qnn_ProfileHandle_t,
                                 Qnn_SignalHandle_t) {
  assert(reinterpret_cast<uintptr_t>(graph) == ++finalized);
  return fail_finalize ? QNN_GRAPH_ERROR_INVALID_ARGUMENT : QNN_SUCCESS;
}

Qnn_ErrorHandle_t execute_graph(Qnn_GraphHandle_t graph,
                                const Qnn_Tensor_t *inputs, uint32_t,
                                Qnn_Tensor_t *outputs, uint32_t,
                                Qnn_ProfileHandle_t, Qnn_SignalHandle_t) {
  uintptr_t id = reinterpret_cast<uintptr_t>(graph);
  executed.push_back(id);
  if (fail_execute || id == fail_graph)
    return QNN_GRAPH_ERROR_INVALID_ARGUMENT;
  auto client = QNN_TENSOR_GET_CLIENT_BUF(inputs[0]);
  const uint8_t *data = static_cast<const uint8_t *>(client.data);
  last_input.assign(data, data + client.dataSize);
  auto output = QNN_TENSOR_GET_CLIENT_BUF(outputs[0]);
  output.dataSize = id == invalid_output_graph ? 1 : 2;
  QNN_TENSOR_SET_CLIENT_BUF(outputs[0], output);
  memcpy(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data,
         QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data, 2);
  return QNN_SUCCESS;
}

} // namespace qnn_mock
