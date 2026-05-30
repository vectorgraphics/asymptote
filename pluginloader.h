/*****
 * pluginloader.h
 * Loader for C++ plugin modules (asybind) — Phase 0.
 *
 * Given a base module name (e.g. "hello"), `tryLoadPlugin` searches the
 * asy file-path for a corresponding shared library (`libhello.so` on
 * Linux, `libhello.dylib` on macOS, `hello.dll` on Windows), dlopens it,
 * verifies the asybind ABI handshake, and runs the plugin's populate
 * function against a freshly-built `record`. On success the populated
 * `record*` is returned; on any failure it returns nullptr (without
 * emitting a diagnostic — the caller decides whether the absence of a
 * plugin is itself an error).
 *****/

#ifndef PLUGINLOADER_H
#define PLUGINLOADER_H

#include "common.h"
#include "symbol.h"

namespace types { class record; }

namespace asybind {

/* Attempts to locate and load a C++ plugin matching `filename`. Returns
 * nullptr if no such plugin exists or if loading failed. On a real
 * loading failure (ABI mismatch, missing entry point, dlopen error for
 * a file that *was* found) a diagnostic is emitted via `em`. */
types::record* tryLoadPlugin(sym::symbol id, mem::string filename);

}  // namespace asybind

#endif /* PLUGINLOADER_H */
