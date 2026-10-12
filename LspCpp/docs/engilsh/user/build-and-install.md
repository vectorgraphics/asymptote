# Build and install

Complete reference for building, installing, and consuming LspCpp.

## Dependencies

### Core (library only)

| Dependency | Source |
|------------|--------|
| [Asio](https://think-async.com/Asio/) (standalone) | `third_party/asio`, or vcpkg / system |
| [RapidJSON](https://github.com/Tencent/rapidjson) | `third_party/rapidjson`, or system |
| [utfcpp](https://github.com/nemtrif/utfcpp) | `third_party/utfcpp` |
| [IXWebSocket](https://github.com/machinezone/IXWebSocket) | `third_party/ixwebsocket`, or vcpkg |
| ZLIB | System (used by IXWebSocket compression) |

Boost is **not** required when `LSPCPP_STANDALONE_ASIO=ON` (default).

### Optional

| Dependency | When needed |
|------------|-------------|
| Boost (`filesystem`, `program_options`, `system`) | Examples and tests |
| [Boehm GC](https://www.hboehm.info/gc/) | `LSPCPP_SUPPORT_BOEHM_GC=ON` |

## Basic build

```shell
mkdir _build && cd _build
cmake ..
cmake --build . -j
```

Default build type is `RelWithDebInfo`. Override with:

```shell
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Install locally

```shell
cmake -S . -B _build -DLSPCPP_INSTALL=ON
cmake --build _build -j
cmake --install _build --prefix /path/to/install
```

Then in your project:

```cmake
list(APPEND CMAKE_PREFIX_PATH "/path/to/install")
find_package(lspcpp CONFIG REQUIRED)
target_link_libraries(my_server PRIVATE lspcpp::lspcpp)
```

## CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `LSPCPP_STANDALONE_ASIO` | `ON` | Use standalone Asio instead of Boost.Asio |
| `LSPCPP_BUILD_MINIMAL_EXAMPLE` | `OFF` | Build the Boost-free minimal stdio example |
| `LSPCPP_BUILD_WEBSOCKETS` | `ON` | Build WebSocket server support |
| `LSPCPP_BUILD_EXAMPLES` | `OFF` | Build example applications |
| `LSPCPP_BUILD_TESTS` | `OFF` | Build and register CTest smoke tests |
| `LSPCPP_BUILD_PERF_SMOKE` | `OFF` | Optional performance smoke tests |
| `LSPCPP_PERF_WARNINGS_AS_ERRORS` | `OFF` | Fail perf smoke on warnings |
| `LSPCPP_USE_CPP17` | `ON` | Compile as C++17 |
| `LSPCPP_INSTALL` | `OFF` | Create install target |
| `LSPCPP_USE_STATIC_CRT` | `OFF` | Static MSVC runtime (`/MT`, Windows only) |
| `LSPCPP_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings as errors |
| `LSPCPP_SUPPORT_BOEHM_GC` | `OFF` | Enable Boehm GC integration |
| `USE_SYSTEM_RAPIDJSON` | `OFF` | Use system RapidJSON |
| `USE_EXTERNAL_ASIO` | `OFF` | Use vcpkg/system Asio |
| `USE_EXTERNAL_IXWEBSOCKET` | `OFF` | Use vcpkg/system IXWebSocket |

Enabling `LSPCPP_BUILD_TESTS` automatically turns on `LSPCPP_BUILD_EXAMPLES`.

## Examples and tests

```shell
# Linux
sudo apt-get install libboost-filesystem-dev libboost-program-options-dev libboost-system-dev

# macOS
brew install boost

cmake -DLSPCPP_BUILD_TESTS=ON ..
cmake --build . -j
ctest --output-on-failure
```

With the default `LSPCPP_BUILD_WEBSOCKETS=ON`, this registers 19 CTest cases. With `-DLSPCPP_BUILD_PERF_SMOKE=ON`, one additional perf test is added.

## vcpkg

A `vcpkg.json` manifest ships with the repository:

```shell
export VCPKG_ROOT=/path/to/vcpkg
export LSPCPP_CI_VCPKG_FEATURES=tests   # optional: Boost for tests/examples
cmake --preset ci/default
cmake --build --preset ci/default -j
ctest --preset ci/default
```

Install via overlay port:

```shell
vcpkg install lspcpp --overlay-ports=ports
```

## Windows notes

Generate a Visual Studio solution:

```shell
mkdir _build && cd _build
cmake ..
cmake --build .
```

For static CRT (`/MT`, `/MTd`):

```shell
cmake -DLSPCPP_USE_STATIC_CRT=ON ..
```

When using vcpkg on Windows, match the CRT triplet (e.g. `x64-windows-static`).

## Versioning

LspCpp uses semantic versioning. Current version: **1.0.3** (see `LSPCPP_VERSION` in `CMakeLists.txt` and `version-semver` in `vcpkg.json`).

## Editor integration

Language servers built with LspCpp are ordinary executables. Your editor's LSP client extension needs:

1. **Command** — path to the server binary
2. **Arguments** — optional CLI flags (many servers use Boost `program_options`, see `StdIOServerExample`)
3. **Root directory** — usually the workspace folder

The server must:

- Read LSP messages from **stdin**
- Write LSP messages to **stdout**
- Use **stderr** only for logging (not protocol traffic)

Configure tracing in your editor while developing; most LSP clients can log raw JSON-RPC messages to help debug handler registration and capability mismatches.

## Projects using LspCpp

- [JCIDE](https://www.javacardos.com/javacardforum/viewtopic.php?f=5&t=3569)
- [LPG-language-server](https://github.com/kuafuwang/LPG-language-server)
- [Asymptote](https://github.com/vectorgraphics)
- [chemical](https://github.com/chemicallang/chemical)
