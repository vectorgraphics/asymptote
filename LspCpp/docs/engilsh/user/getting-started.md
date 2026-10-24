# Getting started

This guide walks you through building LspCpp and running a minimal language server over stdio—the transport used by most editors (VS Code, Neovim, Emacs, etc.). If you are not building an LSP server and only want a typed JSON-RPC message framework, see [Custom protocol framework](custom-protocol.md).

## Prerequisites

- **CMake** 3.16 or newer
- A **C++17** compiler (GCC, Clang, or MSVC)
- No Boost required for the library itself or the minimal example

Optional (for full examples and tests):

- **Boost** (`filesystem`, `program_options`, `system`)

## Build the library

```shell
mkdir _build && cd _build
cmake ..
cmake --build . -j
```

This produces `liblspcpp.a` (Linux/macOS) or `lspcpp.lib` (Windows) under the build directory.

## Build the minimal example

The smallest runnable server lives in `examples/MinimalStdIOServerExample.cpp`. It does not require Boost:

```shell
cmake -S . -B _build_minimal -DLSPCPP_BUILD_MINIMAL_EXAMPLE=ON
cmake --build _build_minimal --target MinimalStdIOServerExample
```

The binary is written to `_build_minimal/MinimalStdIOServerExample` (or `.exe` on Windows).

## Minimal server code

```cpp
#include "LibLsp/LspCpp.h"

int main() {
    lsp::LanguageSession server;
    Condition<bool> exit_requested;

    server.on([](td_initialize::request const& req) {
        td_initialize::response rsp;
        rsp.id = req.id;
        return rsp;
    });

    server.on([&](Notify_Exit::notify&) {
        exit_requested.notify(std::make_unique<bool>(true));
    });

    server.startStdio();
    exit_requested.wait();
    server.stop();
}
```

Key points:

1. **`LanguageSession`** wraps JSON-RPC I/O, message parsing, and handler dispatch.
2. **`server.on(...)`** registers handlers for LSP requests and notifications. The handler type (e.g. `td_initialize::request`) determines which method is handled.
3. **`startStdio()`** reads LSP frames from stdin and writes responses to stdout.
4. Wait for the **`exit`** notification before calling **`stop()`**, following the LSP shutdown sequence.

## Link against LspCpp in your project

After installing (see [Build and install](build-and-install.md)):

```cmake
find_package(lspcpp CONFIG REQUIRED)

add_executable(my_language_server main.cpp)
target_link_libraries(my_language_server PRIVATE lspcpp::lspcpp)
```

Or add LspCpp as a subdirectory in your CMake project and link to the `lspcpp` target directly.

## Test with an editor

Most editors launch a language server as a subprocess and communicate over stdin/stdout. Configure your editor to run the path to your built binary. For example, in VS Code `settings.json`:

```json
{
  "languageServerExample.trace.server": "verbose",
  "languageServerExample.serverPath": "/path/to/MinimalStdIOServerExample"
}
```

Exact configuration depends on your editor and LSP client extension. See [Build and install — Editor integration](build-and-install.md#editor-integration) for general guidance.

## Next steps

- [Writing a language server](writing-a-language-server.md) — implement capabilities, handle document events, send diagnostics
- [Advanced customization](advanced-customization.md) — add custom messages and override built-in parsers
- [Custom protocol framework](custom-protocol.md) — use LspCpp without LSP-specific message types
- [Transport options](transport.md) — TCP and WebSocket servers
- [Build and install](build-and-install.md) — all CMake options and vcpkg usage
