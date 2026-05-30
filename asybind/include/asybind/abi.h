/*****
 * asybind/abi.h
 * Asymptote C++ plugin ABI — Phase 2.
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
#define ASYBIND_ABI_VERSION 3u

#if defined(_WIN32) || defined(__CYGWIN__)
#  define ASY_EXPORT __declspec(dllexport)
#else
#  define ASY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct asybind_stack_opaque*    asybind_stack_ptr;
typedef struct asybind_module_opaque*   asybind_module_ptr;
typedef struct asybind_class_opaque*    asybind_class_ptr;
typedef struct asybind_callable_opaque* asybind_callable_ptr;
typedef struct asybind_funty_opaque*    asybind_funty_ptr;

/* Type tags. ASYBIND_USERPTR denotes an opaque pointer to a C++-backed
 * class instance; the accompanying `cls` field of `asybind_type_spec`
 * identifies which class. ASYBIND_FUNCTION denotes an asy function
 * value; the accompanying `fnty` field identifies the asy types::function*
 * (produced by `make_function_type`). */
enum asybind_type_tag {
  ASYBIND_VOID     = 0,
  ASYBIND_INT      = 1,
  ASYBIND_REAL     = 2,
  ASYBIND_BOOL     = 3,
  ASYBIND_STRING   = 4,
  ASYBIND_USERPTR  = 5,
  ASYBIND_FUNCTION = 6
};

/* A full type description used in function signatures.
 *  - For primitive tags `cls` and `fnty` are ignored (pass null).
 *  - For ASYBIND_USERPTR  `cls`  is the asybind_class_ptr returned by
 *                                create_class or result_class.
 *  - For ASYBIND_FUNCTION `fnty` is the asybind_funty_ptr returned by
 *                                make_function_type. */
typedef struct asybind_type_spec {
  int                  tag;
  asybind_class_ptr    cls;
  asybind_funty_ptr    fnty;
} asybind_type_spec;

struct asybind_host_api_v1;

/* Thunk signature: invoked by the host's bltin wrapper. */
typedef void (*asybind_thunk_t)(asybind_stack_ptr stack,
                                const struct asybind_host_api_v1* api);

struct asybind_host_api_v1 {
  /* === Primitive marshalling ======================================= */
  void (*push_int)   (asybind_stack_ptr, long long);
  void (*push_real)  (asybind_stack_ptr, double);
  void (*push_bool)  (asybind_stack_ptr, int);
  void (*push_string)(asybind_stack_ptr, const char* data, size_t len);

  long long (*pop_int)   (asybind_stack_ptr);
  double    (*pop_real)  (asybind_stack_ptr);
  int       (*pop_bool)  (asybind_stack_ptr);
  void      (*pop_string)(asybind_stack_ptr,
                          const char** out_data, size_t* out_len);

  /* === Error reporting ============================================= */
  void (*raise)(const char* msg);

  /* === Free-function registration ================================== */
  void (*add_func)(asybind_module_ptr module, const char* name,
                   asybind_thunk_t thunk,
                   asybind_type_spec restype,
                   int nargs, const asybind_type_spec* argtypes);

  /* === Class support (Phase 1) ===================================== */
  asybind_class_ptr (*create_class)(asybind_module_ptr module,
                                    const char* name);
  void (*add_method)(asybind_class_ptr cls, const char* name,
                     asybind_thunk_t thunk,
                     asybind_type_spec restype,
                     int nargs, const asybind_type_spec* argtypes);
  void (*add_readonly_field)(asybind_class_ptr cls, const char* name,
                             asybind_thunk_t getter,
                             asybind_type_spec type);
  void* (*alloc_obj)(size_t size);
  void  (*push_obj)(asybind_stack_ptr, void* obj);
  void* (*pop_obj) (asybind_stack_ptr);

  /* === Callable support (Phase 2) ================================== */

  /* Synthesize an asy function type matching the supplied signature. */
  asybind_funty_ptr (*make_function_type)(asybind_type_spec restype,
                                          int nargs,
                                          const asybind_type_spec* argtypes);

  /* Marshal asy callables (vm::callable*) as opaque pointers. */
  asybind_callable_ptr (*pop_callable) (asybind_stack_ptr);
  void                 (*push_callable)(asybind_stack_ptr,
                                        asybind_callable_ptr);

  /* Invoke an asy callable: caller must push arguments in source order
   * (last arg on top); on return the callable's result (if any) is on
   * top of stack. */
  void (*invoke_callable)(asybind_stack_ptr stack,
                          asybind_callable_ptr callable);

  /* === Result<T> support (Phase 2) ================================= */

  /* Synthesize (and cache) a host-side record `result_T` with two
   * readonly fields: `bool found` and `T value` (T = `elem`). */
  asybind_class_ptr (*result_class)(asybind_type_spec elem);

  /* Construct and push a result instance. If `found` is non-zero, the
   * value is first popped from the top of the stack. */
  void (*push_result)(asybind_stack_ptr stack,
                      asybind_class_ptr result_cls,
                      int found);
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
