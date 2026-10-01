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

#include <vector>

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
   * @brief Expected dense tensor shape, type and per-tensor quantization.
   * @note Scale/offset are checked only for fixed-point tensors.
   */
  struct TensorSpec {
    std::vector<uint32_t> dimensions;
    Qnn_DataType_t data_type;
    float scale;
    int32_t offset;
  };

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
  QNN_INTERFACE_VER_TYPE *m_qnn_interface{nullptr};
  FreeGraphInfoFnHandleType_t m_freeGraphsInfoFn{nullptr};

  /* Graph data */
  GraphInfo_t **m_graphs_info{nullptr};
  uint32_t m_graphs_count{0};
  bool m_initialized{false};
  bool m_inputs_ready{false};
  bool m_output_valid{false};

  Qnn_Tensor_t **m_tensor_inputs{nullptr}, **m_tensor_outputs{nullptr};

public:
  Model(Qnn_BackendHandle_t *backend_handle, Qnn_DeviceHandle_t *device_handle,
        Qnn_ProfileHandle_t *profile_handle,
        QNN_INTERFACE_VER_TYPE *qnn_interface);
  ~Model() { close(); }
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

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
   * @brief Validate the single-graph input/output contract at startup.
   * @note Requires one static, dense input and output tensor.
   * @return OK if shape, type and quantization match both specifications
   */
  Result validate_io(const TensorSpec &input, const TensorSpec &output) const;

  /**
   * @brief Execute all model graphs on device using the filled inputs.
   * @return Model action result
   */
  Result execute();

  /**
   * @brief Fill a single input tensor from buffer.
   *
   * Requires a single graph with one input tensor. Bytes must already match
   * the model's layout and quantization; this function performs a raw copy.
   *
   * @param i_buffer Input vector buffer
   * @return Model action result
   */
  Result fill_input(std::vector<uint8_t> &i_buffer);

  /**
   * @brief Fill multiple input tensors from 3D buffer.
   *
   * This function copies the passed 3D buffer to multiple graph input
   * tensors. Make sure the element count of the top-level vector equals the
   * number of model graphs, and the second level matches the input tensor count
   * of the graph. Use when the model consists of multiple graphs or batches.
   * Bytes must already match each tensor's layout and quantization.
   *
   * @param i_buffer 3D Input vector buffer
   * @return Model action result
   */
  Result fill_inputs(std::vector<std::vector<std::vector<uint8_t>>> &i_buffer);

  /**
   * @brief Retrieve the single output tensor of the model.
   *
   * This function retrieves the output tensor of the model and copies it to the
   * passed vector buffer.
   *
   * @note Requires successful execution since the last input fill.
   * @param[out] o_buffer Vector to hold the output.
   * @return Model action result
   */
  Result output(std::vector<uint8_t> &o_buffer);

  /**
   * @brief Retrieve the single output float tensor of the model.
   *
   * This function retrieves the output tensor of the model and copies it to the
   * passed float vector buffer. If the output tensor type is quantized 8-bits,
   * dequantization is automatically applied.
   *
   * @note Requires successful execution since the last input fill.
   * @param[out] o_buffer Vector to hold the output.
   * @return Model action result
   */
  Result output(std::vector<float> &o_buffer);

  /**
   * @brief Retrieve multi-graph output tensors of the model.
   *
   * This function retrieves all output tensors of the model and copies them to
   * the passed 3D vector buffer. The top-level size of the output buffer will
   * change according to the actual output size automatically.
   *
   * Top dim: graphs
   * Second dim: output tensors per graph
   * Third dim: output data
   *
   * @note Requires successful execution of all graphs since the last input
   * fill.
   * @param[out] o_buffer 3D vector to hold the output.
   * @return Model action result
   */
  Result outputs(std::vector<std::vector<std::vector<uint8_t>>> &o_buffer);

  /**
   * @brief Retrieve multi-graph output float tensors of the model.
   *
   * This function retrieves all output tensors of the model and copies them to
   * the passed 3D vector buffer. The top-level size of the output buffer will
   * change according to the actual output size automatically. If the output
   * tensor type is quantized 8-bits, dequantization is automatically applied.
   *
   * Top dim: graphs
   * Second dim: output tensors per graph
   * Third dim: output data
   *
   * @note Requires successful execution of all graphs since the last input
   * fill.
   * @param[out] o_buffer 3D vector to hold the output.
   * @return Model action result
   */
  Result outputs(std::vector<std::vector<std::vector<float>>> &o_buffer);

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
