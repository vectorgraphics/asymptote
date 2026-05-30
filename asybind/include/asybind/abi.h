/*****
 * asybind/abi.h
 * Asymptote C++ plugin ABI — Phase 1.
 *
 * This header defines the binary contract between an Asymptote host (the
 * `asy` executable) and a C++ plugin (a shared library loaded at runtime).
 *
 * The plugin exports a single C-linkage symbol, `asybind_init_v1`, which
 * returns a pointer to a static `module_descriptor`. The host calls the
 * descriptor's `populate` function with an opaque module handle and a
 * pointer to a `host_api_v1` table; the plugin uses the table to register
 * functions, classes, and transfer values across the boundary.
 *
 * Keep this header free of C++ standard-library and Asymptote-internal
 * includes so that plugin authors are not forced to pull in heavy
 * dependencies.
 *****/

#ifndef ASYBIND_ABI_H
#define ASYBIND_ABI_H

#include <stddef.h>

/* Bump on any incompatible change to host_api_v1 or module_descriptor. */
#define ASYBIND_ABI_VERSION 2u

#if defined(_WIN32) || defined(__CYGWIN__)
#  define ASY_EXPORT __declspec(dllexport)
#else
#  define ASY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct asybind_stack_opaque*  asybind_stack_ptr;
typedef struct asybind_module_opaque* asybind_module_ptr;
typedef struct asybind_class_opaque*  asybind_class_ptr;

/* Type tags. ASYBIND_USERPTR denotes an opaque pointer to a C++-backed
 * class instance; the accompanying `cls` field of `asybind_type_spec`
 * identifies which class. */
enum asybind_type_tag {
  ASYBIND_VOID    = 0,
  ASYBIND_INT     = 1,
  ASYBIND_REAL    = 2,
  ASYBIND_BOOL    = 3,
  ASYBIND_STRING  = 4,
  ASYBIND_USERPTR = 5
};

/* A full type description used in function signatures. For primitive
 * tags `cls` is ignored and should be null. For ASYBIND_USERPTR the
 * `cls` field is the asybind_class_ptr returned by create_class. */
typedef struct asybind_type_spec {
  int tag;
  asybind_class_ptr cls;
} asybind_type_spec;

struct asybind_host_api_v1;

/* Thunk signature: invoked by the host's bltin wrapper. The thunk pops
 * its arguments from `stack` using `api`, calls the C++ body, and pushes
 * its result (if any).
 *
 * For free functions, the stack at entry has args in source order with
 * the last argument on top; pop in reverse to recover positional args.
 *
 * For methods (registered via add_method), the receiver (USERPTR) is on
 * top of the stack, followed by args (last argument on top of the args).
 * Pop the receiver FIRST, then args in reverse order.
 *
 * For readonly-field getters (registered via add_readonly_field), the
 * receiver is the only thing on top; pop it and push the field value. */
typedef void (*asybind_thunk_t)(asybind_stack_ptr stack,
                                const struct asybind_host_api_v1* api);

struct asybind_host_api_v1 {
  /* === Primitive marshalling ======================================= */

  void (*push_int)   (asybind_stack_ptr, long long);
  void (*push_real)  (asybind_stack_ptr, double);
  void (*push_bool)  (asybind_stack_ptr, int);
  /* The host copies `len` bytes into a freshly GC-allocated string and
   * pushes the resulting string-pointer item. */
  void (*push_string)(asybind_stack_ptr, const char* data, size_t len);

  long long (*pop_int)   (asybind_stack_ptr);
  double    (*pop_real)  (asybind_stack_ptr);
  int       (*pop_bool)  (asybind_stack_ptr);
  /* `*out_data` points into a GC-managed buffer; the plugin must copy if
   * it intends to retain the bytes past the current thunk invocation. */
  void      (*pop_string)(asybind_stack_ptr,
                          const char** out_data, size_t* out_len);

  /* === Error reporting ============================================= */

  /* Report an asy-side error and abort the current invocation. Never
   * returns. */
  void (*raise)(const char* msg);

  /* === Free-function registration ================================== */

  /* Register a function in `module`. `argtypes` is an array of `nargs`
   * type specs (positional). `restype` is also a spec; use
   * `{ASYBIND_VOID, NULL}` for procedures. */
  void (*add_func)(asybind_module_ptr module, const char* name,
                   asybind_thunk_t thunk,
                   asybind_type_spec restype,
                   int nargs, const asybind_type_spec* argtypes);

  /* === Class support (Phase 1) ===================================== */

  /* Create a new C++-backed class in `module` with the given name. The
   * returned handle is used to register methods and fields and to
   * identify instances in type specs (tag = ASYBIND_USERPTR, cls =
   * handle). */
  asybind_class_ptr (*create_class)(asybind_module_ptr module,
                                    const char* name);

  /* Register a method on `cls`. The method's asy signature is the
   * UNBOUND signature (e.g. `int()` for `Stack::size`, NOT
   * `int(Stack)`); the receiver is implicit. At call time, the receiver
   * is on top of the stack followed by `nargs` user-visible arguments. */
  void (*add_method)(asybind_class_ptr cls, const char* name,
                     asybind_thunk_t thunk,
                     asybind_type_spec restype,
                     int nargs, const asybind_type_spec* argtypes);

  /* Register a readonly virtual field on `cls`. The getter thunk pops
   * the receiver and pushes the field value (of type `type`). */
  void (*add_readonly_field)(asybind_class_ptr cls, const char* name,
                             asybind_thunk_t getter,
                             asybind_type_spec type);

  /* Allocate `size` bytes of GC-managed memory. The pointer is
   * conservatively scanned and never freed unless unreachable. */
  void* (*alloc_obj)(size_t size);

  /* Push/pop a raw GC pointer (used for class instances). Both sides
   * tag the item identically so that an item pushed with push_obj can
   * always be retrieved by pop_obj regardless of the underlying C++
   * type. */
  void  (*push_obj)(asybind_stack_ptr, void* obj);
  void* (*pop_obj) (asybind_stack_ptr);
};

struct asybind_module_descriptor {
  unsigned int abi_version;        /* ASYBIND_ABI_VERSION */
  const char*  asy_version_string; /* informational; mismatch is a warning */
  const char*  module_name;
  void (*populate)(asybind_module_ptr module,
                   const struct asybind_host_api_v1* api);
};

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ASYBIND_ABI_H */
