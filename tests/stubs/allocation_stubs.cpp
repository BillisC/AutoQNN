#include "allocation_stubs.h"

#include <cstdlib>

/* Inject allocation failures into the tensor utilities */
namespace allocation_stub {

int calloc_budget = -1;
bool fail_name = false;

} // namespace allocation_stub
extern "C" void *__real_calloc(size_t, size_t);
extern "C" char *__real_strdup(const char *);
extern "C" void *__wrap_calloc(size_t count, size_t size) {
  if (allocation_stub::calloc_budget == 0)
    return nullptr;
  if (allocation_stub::calloc_budget > 0)
    allocation_stub::calloc_budget--;
  return __real_calloc(count, size);
}
extern "C" char *__wrap_strdup(const char *name) {
  return allocation_stub::fail_name ? nullptr : __real_strdup(name);
}
