/**
 * @file    dynload.h
 * @brief   Dynamic loading functions header
 *
 * This header contains templates and forward declarations for the dynamic
 * loading of libraries, under the dl namespace.
 *
 * @author  BillisC (Vasileios Ch.)
 */

#ifndef DYNLOAD_H
#define DYNLOAD_H

#include <dlfcn.h>

namespace dl {

// Address to distinguIsh from NULL pointer
#define DL_DEFAULT (void *)(0x4)

void *dl_open(const char *file, int mode);

bool dl_close(void *libHandle);

void *dl_sym(void *__restrict libHandle, const char *__restrict name);

template <class T>
inline T resolve_symbol(void *libHandle, const char *symbol) {
  return (T)dl::dl_sym(libHandle, symbol);
}

} // namespace dl

#endif