# LspCpp

A C++ library for building [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/) servers and custom JSON-RPC protocols. It provides JSON-RPC transport, LSP message types, typed custom messages, and helpers for stdio, TCP, and WebSocket communication.

[中文 README](README_zh.md)

## Documentation

Full documentation lives in the [`docs/`](docs/) directory:

- **Users** — [getting started](docs/engilsh/user/getting-started.md), [writing a language server](docs/engilsh/user/writing-a-language-server.md), [advanced customization](docs/engilsh/user/advanced-customization.md), [custom protocol framework](docs/engilsh/user/custom-protocol.md), [transport](docs/engilsh/user/transport.md), [build and install](docs/engilsh/user/build-and-install.md)
- **Contributors** — [architecture](docs/engilsh/developer/architecture.md), [build and test](docs/engilsh/developer/build-and-test.md), [contributing](docs/engilsh/developer/contributing.md), [protocol types](docs/engilsh/developer/protocol-types.md)
- **English index** — [docs/engilsh/README.md](docs/engilsh/README.md)
- **中文** — [文档索引](docs/zh/README.md)

## Dependencies

### Core (library only)

These are required to build the `lspcpp` static library:

| Dependency | Source |
| ---------- | ------ |
| [Asio](https://think-async.com/Asio/) (standalone) | Bundled in `third_party/asio`, or via vcpkg / system package |
| [RapidJSON](https://github.com/Tencent/rapidjson) | Bundled in `third_party/rapidjson`, or system package |
| [utfcpp](https://github.com/nemtrif/utfcpp) | Bundled in `third_party/utfcpp` |
| [IXWebSocket](https://github.com/machinezone/IXWebSocket) | Bundled in `third_party/ixwebsocket`, or via vcpkg |
| [ZLIB](https://zlib.net/) | System package used by bundled IXWebSocket compression support |

Boost is **not** required for the library when `LSPCPP_STANDALONE_ASIO` is enabled (the default).

### Optional

| Dependency | When needed |
| ---------- | ----------- |
| Boost (`filesystem`, `program_options`, `system`) | Building examples or tests |
| [Boehm GC](https://www.hboehm.info/gc/) | Optional GC support (`LSPCPP_SUPPORT_BOEHM_GC=ON`) |

## Build

Requires CMake 3.16+ and a C++17-capable compiler (`LSPCPP_USE_CPP17=ON` by default).

### 1. Build the library (Linux / macOS)

```shell
mkdir _build && cd _build
cmake ..
cmake --build . -j
```

This produces `liblspcpp.a` (or `lspcpp.lib` on Windows) without installing Boost.

### 2. Quickstart: minimal stdio server

`LibLsp/LspCpp.h` provides a small convenience entry point for new code while the
existing `RemoteEndPoint` API remains fully supported:

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

The same code is available as `examples/MinimalStdIOServerExample.cpp` and does
not require Boost:

```shell
cmake -S . -B _build_minimal -DLSPCPP_BUILD_MINIMAL_EXAMPLE=ON
cmake --build _build_minimal --target MinimalStdIOServerExample
```

### 3. Use an installed package

```cmake
find_package(lspcpp CONFIG REQUIRED)

add_executable(my_language_server main.cpp)
target_link_libraries(my_language_server PRIVATE lspcpp::lspcpp)
```

Install from a local checkout with:

```shell
cmake -S . -B _build -DLSPCPP_INSTALL=ON
cmake --build _build -j
cmake --install _build --prefix /path/to/install
```

### 4. Build with examples and tests

Examples and CTest smoke tests require Boost. Enable them with:

```shell
cmake -DLSPCPP_BUILD_TESTS=ON ..
cmake --build . -j
ctest --output-on-failure
```

`LSPCPP_BUILD_TESTS=ON` automatically enables `LSPCPP_BUILD_EXAMPLES`. With the default `LSPCPP_BUILD_WEBSOCKETS=ON`, this registers 19 CTest cases.

Optional performance smoke tests (`lspcpp.perf_smoke`, adds one more CTest case) can be enabled with:

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_BUILD_PERF_SMOKE=ON ..
cmake --build . -j
ctest --output-on-failure
```

To fail the perf smoke test when it emits warnings:

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_BUILD_PERF_SMOKE=ON -DLSPCPP_PERF_WARNINGS_AS_ERRORS=ON ..
```

On Linux:

```shell
sudo apt-get install libboost-filesystem-dev libboost-program-options-dev libboost-system-dev
```

On macOS:

```shell
brew install boost
```

### 5. Build with vcpkg (CI configuration)

A `vcpkg.json` manifest is provided for convenience. It is not required for a local build, but matches the CI setup.

Set `VCPKG_ROOT` to your vcpkg installation, then:

```shell
export LSPCPP_CI_VCPKG_FEATURES=tests   # optional: pull in Boost for tests/examples
cmake --preset ci/default
cmake --build --preset ci/default -j
ctest --preset ci/default
```

An overlay vcpkg port is available under `ports/lspcpp` for local validation:

```shell
vcpkg install lspcpp --overlay-ports=ports
```

This overlay port is structured like an upstream vcpkg port, but it uses this
checkout as the source tree. Before submitting it to the official vcpkg registry,
switch the source acquisition to the tagged GitHub release and fill in the
archive SHA512 expected by `vcpkg_from_github`.

### Windows

Generate a Visual Studio solution and build:

```shell
mkdir _build
cd _build
cmake ..
```

Open the generated solution in Visual Studio, or build from the command line with `cmake --build .`.

To request the static CRT (`/MT`, `/MTd`), pass `-DLSPCPP_USE_STATIC_CRT=ON`.
This keeps bundled dependencies such as `ixwebsocket` on the same
runtime so you do not need to override `/MD` manually. If a toolchain already sets
`CMAKE_MSVC_RUNTIME_LIBRARY`, LspCpp follows that runtime for bundled dependencies.
When using vcpkg for external dependencies, build with a matching CRT triplet (for
example `x64-windows-static`).

To build tests on Windows, use vcpkg with the `tests` feature (see section 5 above).

### Useful CMake options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `LSPCPP_STANDALONE_ASIO` | `ON` | Use standalone Asio instead of Boost.Asio |
| `LSPCPP_BUILD_MINIMAL_EXAMPLE` | `OFF` | Build the Boost-free minimal stdio example |
| `LSPCPP_BUILD_WEBSOCKETS` | `ON` | Build WebSocket server support |
| `LSPCPP_BUILD_EXAMPLES` | `OFF` | Build example applications |
| `LSPCPP_BUILD_TESTS` | `OFF` | Build and register CTest smoke tests |
| `LSPCPP_USE_CPP17` | `ON` | Compile as C++17 |
| `USE_SYSTEM_RAPIDJSON` | `OFF` | Use system RapidJSON instead of the submodule |
| `USE_EXTERNAL_ASIO` | `OFF` | Use vcpkg/system Asio instead of the submodule |
| `USE_EXTERNAL_IXWEBSOCKET` | `OFF` | Use vcpkg/system IXWebSocket instead of the submodule |
| `LSPCPP_USE_STATIC_CRT` | `OFF` | Request static MSVC runtime linking (`/MT`, `/MTd`; Windows only) |

## Examples

Example applications live in the [examples](https://github.com/kuafuwang/LspCpp/tree/master/examples) directory:

- `StdIOClientExample` / `StdIOServerExample` — stdio JSON-RPC
- `TcpServerExample` — TCP server
- `WebsocketExample` — WebSocket server

## Versioning

LspCpp uses semantic versioning for project and package metadata. The current
project version is `1.0.3`.

When preparing a release:

- Update `LSPCPP_VERSION` in `CMakeLists.txt`.
- Update `version-semver` in `vcpkg.json` to the same value.
- Tag the release as `vX.Y.Z`, for example `v1.0.3`.
- Use the same version in the GitHub release title.

## Projects using LspCpp

- [JCIDE](https://www.javacardos.com/javacardforum/viewtopic.php?f=5&t=3569&sid=e01238adf55cd08696fbf495dfa6c8e5)
- [LPG-language-server](https://github.com/kuafuwang/LPG-language-server)
- [Asymptote](https://github.com/vectorgraphics)
- [chemical](https://github.com/chemicallang/chemical)

## Reference

Some code from [cquery][1].

## License

MIT

## Development guide

For any merges into the `master` branch, ensure the C++ code complies with the clang-format standard. As of currently, the latest clang-format version offered in Ubuntu 24.04 (18) is used, but this may change in the future as newer versions of clang-format become available for Ubuntu.

To check the current version of clang-format used, see the `check-format-cpp` workflow. It prints out the version used. Ensure the C++ code is compliant with that version of clang-format.

[1]: https://github.com/cquery-project/cquery "cquery"
