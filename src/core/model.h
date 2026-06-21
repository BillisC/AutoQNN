/**
 * @file    model.h
 * @brief   Model class header
 *
 * This header contains enums, structs and forward declarations for the
 * QNN Model wrapper class, under the model namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */
#ifndef GRAPHS_H
#define GRAPHS_H

#include "QnnCommon.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnTypes.h"

namespace model {

/**
 * @brief Model class for graph inference.
 */
class Model {
public:
  /**
   * @brief Model result enum class.
   */
  enum class Result {
    OK = 0,
    TENSOR_ERROR = 1,
    PARAMS_ERROR = 2,
    NODES_ERROR = 3,
    GRAPH_ERROR = 4,
    CONTEXT_ERROR = 5,
    GENERATION_ERROR = 6,
    SETUP_ERROR = 7,
    INVALID_ARGUMENT_ERROR = 8,
    FILE_ERROR = 9,
    MEMORY_ALLOCATE_ERROR = 10,
    UNKNOWN_ERROR = 0x7FFFFFFF
  };

  /**
   * @brief Model graph struct.
   */
  typedef struct GraphInfo {
    Qnn_GraphHandle_t graph;
    char *graphName;
    Qnn_Tensor_t *inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t *outputTensors;
    uint32_t numOutputTensors;
  } GraphInfo_t;
  typedef GraphInfo_t *GraphInfoPtr_t;

  /**
   * @brief Model graph config struct.
   */
  typedef struct GraphConfigInfo {
    char *graphName;
    const QnnGraph_Config_t **graphConfigs;
  } GraphConfigInfo_t;

  using ComposeGraphsFnHandleType_t = Result (*)(
      Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE, Qnn_ContextHandle_t,
      const GraphConfigInfo_t **, const uint32_t, GraphInfo_t ***, uint32_t *,
      bool, QnnLog_Callback_t, QnnLog_Level_t);

  using FreeGraphInfoFnHandleType_t = Result (*)(GraphInfo_t ***, uint32_t);

private:
  /* Handles */
  void *m_libModelHandle{nullptr};

  Qnn_ContextHandle_t m_context_handle{nullptr};
  /* From backend */
  Qnn_BackendHandle_t *m_backend_handle{nullptr};
  Qnn_DeviceHandle_t *m_device_handle{nullptr};
  Qnn_ProfileHandle_t *m_profile_handle{nullptr};

  /* Function Pointers */
  QNN_INTERFACE_VER_TYPE *m_qnn_interface;
  FreeGraphInfoFnHandleType_t m_freeGraphsInfoFn{nullptr};

  /* Graph data */
  GraphInfo_t **m_graphs_info;
  uint32_t m_graphs_count;

  Qnn_Tensor_t **m_tensor_inputs{nullptr}, **m_tensor_outputs{nullptr};

public:
  Model(Qnn_BackendHandle_t *backend_handle, Qnn_DeviceHandle_t *device_handle,
        Qnn_ProfileHandle_t *profile_handle,
        QNN_INTERFACE_VER_TYPE *qnn_interface);
  ~Model() { close(); }

  /**
   * @brief Load static model library.
   *
   * The model static (.so) library has to be generated using Qualcomm's QAIRT
   * SDK tools. For QNN inferences on Hexagon NPUs the model must be compiled
   * with 8-bit quantization ([u]int8).
   *
   * @param lib_path Path to static lib file
   * @return Model action result
   */
  Result load_model_lib(const char *lib_path);

  /**
   * @brief Execute model graph on device.
   * @return Model action result
   */
  Result execute();

private:
  /**
   * @brief Initialize model tensor buffers.
   * @return Model action result
   */
  Result init_tensors();

  /**
   * @brief Release model tensors.
   * @return Model action result
   */
  Result close();
};

} // namespace model

#endif