/**
 * @file    backend.h
 * @brief   DeviceBackend class header
 *
 * This header contains enums, structs and forward declarations for the
 * QNN DeviceBackend wrapper class, under the backend namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#ifndef BACKEND_H
#define BACKEND_H

#include "QnnBackend.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace backend {

/**
 * @brief Backend class for device initialization.
 */
class DeviceBackend {
public:
  /**
   * @brief Device backend result enum class.
   */
  enum class Result {
    OK,
    DL_ERROR,
    BACKEND_ERROR,
    PROFILER_ERROR,
    INTERFACE_ERROR,
    SYSTEM_LIB_ERROR,
    DEVICE_ERROR
  };

  /**
   * @brief Device type enum class.
   */
  enum class DeviceType {
    CPU,
    GPU,
    HTP,
    LPAI
  };

private:
  const DeviceType m_dev_type{DeviceType::CPU};
  const bool m_profiling{false};

  /* Handles */
  void *m_libBackendHandle{nullptr};

  Qnn_BackendHandle_t m_backend_handle{nullptr};
  Qnn_DeviceHandle_t m_device_handle{nullptr};
  // Optional
  Qnn_ProfileHandle_t m_profile_handle{nullptr};

  /* Function Pointers */
  QNN_INTERFACE_VER_TYPE m_qnn_interface;
  QNN_SYSTEM_INTERFACE_VER_TYPE m_qnn_system_interface;

public:
  /* Class constructors */
  DeviceBackend(const DeviceType dev_type, const bool profiler);
  DeviceBackend() = default;
  ~DeviceBackend() { close(); };

  /* Getters */
  Qnn_BackendHandle_t *backend() { return &m_backend_handle; }
  Qnn_BackendHandle_t *device() { return &m_device_handle; }
  Qnn_ProfileHandle_t *profile() { return &m_profile_handle; }
  QNN_INTERFACE_VER_TYPE *interface() { return &m_qnn_interface; }

  /* Actions */
  /**
   * @brief Initialize the QAIRT backend for inference.
   * @return result enum value
   */
  Result init();

private:
  /**
   * @brief Release the QAIRT backend.
   * @return result enum value
   */
  Result close();
};

} // namespace backend

#endif