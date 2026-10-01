#ifndef QNN_MOCK_H
#define QNN_MOCK_H

#include "QnnInterface.h"

#include <cstdint>
#include <vector>

namespace qnn_mock {

extern uint32_t created, freed, finalized;
extern bool fail_context, fail_finalize, fail_execute;
extern uintptr_t fail_graph, invalid_output_graph;
extern std::vector<uintptr_t> executed;
extern std::vector<uint8_t> last_input;

Qnn_ErrorHandle_t create_context(Qnn_BackendHandle_t, Qnn_DeviceHandle_t,
                                 const QnnContext_Config_t **,
                                 Qnn_ContextHandle_t *);
Qnn_ErrorHandle_t free_context(Qnn_ContextHandle_t, Qnn_ProfileHandle_t);
Qnn_ErrorHandle_t finalize_graph(Qnn_GraphHandle_t, Qnn_ProfileHandle_t,
                                 Qnn_SignalHandle_t);
Qnn_ErrorHandle_t execute_graph(Qnn_GraphHandle_t, const Qnn_Tensor_t *,
                                uint32_t, Qnn_Tensor_t *, uint32_t,
                                Qnn_ProfileHandle_t, Qnn_SignalHandle_t);

} // namespace qnn_mock

#endif
