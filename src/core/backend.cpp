/**
 * @file    backend.cpp
 * @brief   DeviceBackend class function definitions
 *
 * This file contains static and class function definitions for the QNN
 * DeviceBackend wrapper class, under the backend namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include "backend.h"

#include "logger.hpp"
#include "dynload.h"

namespace backend {

/* -- Static helper functions -- */

static DeviceBackend::Result
get_interface_provider(void *libBackendHandle,
                       QNN_INTERFACE_VER_TYPE &interface) {
  /* Resolve interface providers symbol from dl backend */
  using QnnInterfaceGetProvidersFn_t =
      Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);

  QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};
  getInterfaceProviders = dl::resolve_symbol<QnnInterfaceGetProvidersFn_t>(
      libBackendHandle, "QnnInterface_getProviders");
  if (getInterfaceProviders == nullptr) {
    return DeviceBackend::Result::DL_ERROR;
  }

  /* Find all providers */
  QnnInterface_t **interfaceProviders{nullptr};
  uint32_t numProviders{0};
  if (QNN_SUCCESS !=
      getInterfaceProviders((const QnnInterface_t ***)&interfaceProviders,
                            &numProviders)) {
    QNN_ERROR("Could not get interface providers");
    return DeviceBackend::Result::INTERFACE_ERROR;
  }

  /* Filter providers */
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    const Qnn_Version_t &apiVer =
        interfaceProviders[pIdx]->apiVersion.coreApiVersion;
    QNN_INFO("Detected interface provider [v%d.%d]", apiVer.major,
             apiVer.minor);

    if (QNN_API_VERSION_MAJOR == apiVer.major /*&&
        QNN_API_VERSION_MINOR <= apiVer.minor*/) {
      interface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
      return DeviceBackend::Result::OK;
    }
  }

  QNN_ERROR("Unable to find a valid interface");
  return DeviceBackend::Result::INTERFACE_ERROR;
}

/* Class Function definitions */

DeviceBackend::DeviceBackend(const DeviceType dev_type, const bool profiler)
    : m_dev_type(dev_type), m_profiling(profiler) {}

DeviceBackend::Result DeviceBackend::init() {

  /* 1. Load dynamic backend library */
  const char *libFile = "";
  switch (m_dev_type) {
    case DeviceType::CPU: libFile = "libQnnCpu.so"; break;
    case DeviceType::GPU: libFile = "libQnnGpu.so"; break;
    case DeviceType::HTP: libFile = "libQnnHtp.so"; break;
    case DeviceType::LPAI: libFile = "libQnnLpai.so"; break;
    default: libFile = "libQnnCpu.so";
  }

  m_libBackendHandle = dl::dl_open(libFile, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == m_libBackendHandle) {
    return DeviceBackend::Result::DL_ERROR;
  }

  /* 2. Get interface providers */
  DeviceBackend::Result status =
      get_interface_provider(m_libBackendHandle, m_qnn_interface);
  if (status != DeviceBackend::Result::OK) {
    return status;
  }

  /* 3. Create backend */
  // const QnnBackend_Config_t *backendConfigs;
  if (QNN_BACKEND_NO_ERROR !=
      m_qnn_interface.backendCreate(nullptr, nullptr, &m_backend_handle)) {
    return DeviceBackend::Result::BACKEND_ERROR;
  }

  /* 4. Add profiling */
  if (m_profiling) {
    if (QNN_PROFILE_NO_ERROR !=
        m_qnn_interface.profileCreate(m_backend_handle, QNN_PROFILE_LEVEL_BASIC,
                                      &m_profile_handle)) {
      return DeviceBackend::Result::PROFILER_ERROR;
    }
  }

  /* 5. Create device */
  if (m_dev_type != DeviceType::CPU) {
    if (QNN_SUCCESS !=
        m_qnn_interface.deviceCreate(nullptr, nullptr, &m_device_handle)) {
      return DeviceBackend::Result::DEVICE_ERROR;
    }
  }

  return DeviceBackend::Result::OK;
}

DeviceBackend::Result DeviceBackend::close() {
  if (m_libBackendHandle != nullptr) {
    dl::dl_close(m_libBackendHandle);
    m_libBackendHandle = nullptr;
  }

  return DeviceBackend::Result::OK;
}

} // namespace backend
