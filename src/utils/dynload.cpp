/**
 * @file    dynload.cpp
 * @brief   Dynamic loading function definitions
 *
 * This file contains function definitions for the dynamic loading of libraries,
 * under the dl namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#include "dynload.h"

#include "logger.hpp"

void *dl::dl_open(const char *file, int mode) {
  void *libHandle{nullptr}; /* Return value */

  if (file == nullptr) {
    QNN_ERROR("dlOpen parameters null");
  } else {
    libHandle = dlopen(file, mode);
    if (libHandle == nullptr) {
      QNN_ERROR("Could not load backend");
    }
  }

  return libHandle;
}

bool dl::dl_close(void *libHandle) {
  bool status{false}; /* Return value */

  /* Verify parameters */
  if (libHandle == nullptr) {
    QNN_ERROR("dlClose parameters null");

  } else {
    status = dlclose(libHandle) != 0;
    if (status) {
      QNN_WARN("Could not close dl lib");
    }
  }

  return status;
}

void *dl::dl_sym(void *__restrict libHandle, const char *__restrict symbol) {
  void *ptr{nullptr}; /* Return value */

  /* Verify parameters */
  if (libHandle == nullptr || symbol == nullptr) {
    QNN_ERROR("dlSym parameters null");

  } else {
    /* Handle dynamic symbol resolving */
    ptr = dlsym(libHandle, symbol);
    if (ptr == nullptr) {
      QNN_ERROR("Unable to access symbol [%s]: %s", symbol, dlerror());
    }
  }

  return ptr;
}