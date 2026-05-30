/*****
 * asybind/abi.h
 * Asymptote C++ plugin ABI — Phase 0.
 *
 * This header defines the binary contract between an Asymptote host (the
 * `asy` executable) and a C++ plugin (a shared library loaded at runtime).
 *
 * The plugin exports a single C-linkage symbol, `asybind_init_v1`, which
 * returns a pointer to a static `module_descriptor`. The host calls the
 * descriptor's `populate` function with an opaque module handle and a
 * pointer to a `host_api_v1` table; the plugin uses the table to register
 * functions and to transfer values across the boundary.
 *
 * Keep this header free of C++ standard-library and Asymptote-internal
 * includes so that plugin authors are not forced to pull in heavy
 * dependencies.
 *****/

#ifndef ASYBIND_ABI_H
#define ASYBIND_ABI_H

#include <stddef.h>

/* Bump on any incompatible change to host_api_v1 or module_descriptor. */
#define ASYBIND_ABI_VERSION 1u

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

/* Type tags for primitives (Phase 0). */
enum asybind_type_tag {
  ASYBIND_VOID   = 0,
  ASYBIND_INT    = 1,
  ASYBIND_REAL   = 2,
  ASYBIND_BOOL   = 3,
  ASYBIND_STRING = 4
};

struct asybind_host_api_v1;

/* Thunk signature: invoked by the host's bltin wrapper. The thunk pops
 * its arguments from `stack` using `api`, calls the C++ body, and pushes
 * its result (if any). */
typedef void (*asybind_thunk_t)(asybind_stack_ptr stack,
                                const struct asybind_host_api_v1* api);

struct asybind_host_api_v1 {
  /* Push primitives onto the asy stack. */
  void (*push_int)   (asybind_stack_ptr, long long);
  void (*push_real)  (asybind_stack_ptr, double);
  void (*push_bool)  (asybind_stack_ptr, int);
  /* The host copies `len` bytes into a freshly GC-allocated string and
   * pushes the resulting string-pointer item. */
  void (*push_string)(asybind_stack_ptr, const char* data, size_t len);

  /* Pop primitives from the asy stack (last argument first). */
  long long (*pop_int)   (asybind_stack_ptr);
  double    (*pop_real)  (asybind_stack_ptr);
  int       (*pop_bool)  (asybind_stack_ptr);
  /* `*out_data` points into a GC-managed buffer; the plugin must copy if
   * it intends to retain the bytes past the current thunk invocation. */
  void      (*pop_string)(asybind_stack_ptr,
                          const char** out_data, size_t* out_len);

  /* Report an asy-side error and abort the current invocation. Never
   * returns. */
  void (*raise)(const char* msg);

  /* Register a function in `module`. `argtypes` is an array of `nargs`
   * `asybind_type_tag` values (positional). `restype` is also a tag;
   * use `ASYBIND_VOID` for procedures. */
  void (*add_func)(asybind_module_ptr module, const char* name,
                   asybind_thunk_t thunk,
                   int restype, int nargs, const int* argtypes);
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
