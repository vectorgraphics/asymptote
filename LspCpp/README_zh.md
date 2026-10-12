# LspCpp

用于构建 [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/) 服务器和自定义 JSON-RPC 协议的 C++ 库。提供 JSON-RPC 传输、LSP 消息类型、类型化自定义消息，以及 stdio、TCP、WebSocket 通信辅助功能。

[English README](README.md)

## 文档

完整文档位于 [`docs/`](docs/) 目录：

- **用户** — [快速入门](docs/zh/user/getting-started.md)、[编写语言服务器](docs/zh/user/writing-a-language-server.md)、[高级定制](docs/zh/user/advanced-customization.md)、[自定义协议框架](docs/zh/user/custom-protocol.md)、[传输方式](docs/zh/user/transport.md)、[构建与安装](docs/zh/user/build-and-install.md)
- **开发者** — [架构](docs/zh/developer/architecture.md)、[构建与测试](docs/zh/developer/build-and-test.md)、[贡献指南](docs/zh/developer/contributing.md)、[协议类型](docs/zh/developer/protocol-types.md)
- **文档索引** — [docs/zh/README.md](docs/zh/README.md)
- **English** — [docs/engilsh/README.md](docs/engilsh/README.md)

## 依赖

### 核心（仅库）

构建 `lspcpp` 静态库需要以下依赖：

| 依赖 | 来源 |
| ---- | ---- |
| [Asio](https://think-async.com/Asio/)（standalone） | 捆绑于 `third_party/asio`，或通过 vcpkg / 系统包 |
| [RapidJSON](https://github.com/Tencent/rapidjson) | 捆绑于 `third_party/rapidjson`，或系统包 |
| [utfcpp](https://github.com/nemtrif/utfcpp) | 捆绑于 `third_party/utfcpp` |
| [IXWebSocket](https://github.com/machinezone/IXWebSocket) | 捆绑于 `third_party/ixwebsocket`，或通过 vcpkg |
| [ZLIB](https://zlib.net/) | 系统包（捆绑 IXWebSocket 压缩支持使用） |

启用 `LSPCPP_STANDALONE_ASIO`（默认）时，库本身**不需要** Boost。

### 可选

| 依赖 | 何时需要 |
| ---- | -------- |
| Boost（`filesystem`、`program_options`、`system`） | 构建示例或测试 |
| [Boehm GC](https://www.hboehm.info/gc/) | 可选 GC 支持（`LSPCPP_SUPPORT_BOEHM_GC=ON`） |

## 构建

需要 CMake 3.16+ 和支持 C++17 的编译器（默认 `LSPCPP_USE_CPP17=ON`）。

### 1. 构建库（Linux / macOS）

```shell
mkdir _build && cd _build
cmake ..
cmake --build . -j
```

这会生成 `liblspcpp.a`（Windows 上为 `lspcpp.lib`），无需安装 Boost。

### 2. 快速入门：最小 stdio 服务器

`LibLsp/LspCpp.h` 为新代码提供便捷入口，同时完全保留现有 `RemoteEndPoint` API：

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

相同代码见 `examples/MinimalStdIOServerExample.cpp`，**不需要** Boost：

```shell
cmake -S . -B _build_minimal -DLSPCPP_BUILD_MINIMAL_EXAMPLE=ON
cmake --build _build_minimal --target MinimalStdIOServerExample
```

### 3. 使用已安装的包

```cmake
find_package(lspcpp CONFIG REQUIRED)

add_executable(my_language_server main.cpp)
target_link_libraries(my_language_server PRIVATE lspcpp::lspcpp)
```

从本地 checkout 安装：

```shell
cmake -S . -B _build -DLSPCPP_INSTALL=ON
cmake --build _build -j
cmake --install _build --prefix /path/to/install
```

### 4. 构建示例与测试

示例和 CTest 冒烟测试需要 Boost。启用方式：

```shell
cmake -DLSPCPP_BUILD_TESTS=ON ..
cmake --build . -j
ctest --output-on-failure
```

`LSPCPP_BUILD_TESTS=ON` 会自动启用 `LSPCPP_BUILD_EXAMPLES`。默认 `LSPCPP_BUILD_WEBSOCKETS=ON` 时注册 **19** 个 CTest 用例。

可选性能冒烟测试（`lspcpp.perf_smoke`，额外增加 1 个 CTest 用例）：

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_BUILD_PERF_SMOKE=ON ..
cmake --build . -j
ctest --output-on-failure
```

性能冒烟出现警告时失败：

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_BUILD_PERF_SMOKE=ON -DLSPCPP_PERF_WARNINGS_AS_ERRORS=ON ..
```

Linux：

```shell
sudo apt-get install libboost-filesystem-dev libboost-program-options-dev libboost-system-dev
```

macOS：

```shell
brew install boost
```

### 5. 使用 vcpkg 构建（CI 配置）

仓库提供 `vcpkg.json` manifest 以便使用，本地构建不强制要求，但与 CI 配置一致。

将 `VCPKG_ROOT` 设为 vcpkg 安装路径，然后：

```shell
export LSPCPP_CI_VCPKG_FEATURES=tests   # 可选：拉取测试/示例所需的 Boost
cmake --preset ci/default
cmake --build --preset ci/default -j
ctest --preset ci/default
```

`ports/lspcpp` 下提供 overlay vcpkg port，用于本地验证：

```shell
vcpkg install lspcpp --overlay-ports=ports
```

该 overlay port 结构类似上游 vcpkg port，但使用本 checkout 作为源码树。提交到官方 vcpkg registry 前，应将源码获取切换为带 tag 的 GitHub release，并填写 `vcpkg_from_github` 所需的 archive SHA512。

### Windows

生成 Visual Studio 解决方案并构建：

```shell
mkdir _build
cd _build
cmake ..
```

在 Visual Studio 中打开生成的解决方案，或使用 `cmake --build .` 从命令行构建。

请求静态 CRT（`/MT`、`/MTd`）时传入 `-DLSPCPP_USE_STATIC_CRT=ON`。
这样捆绑依赖（如 `ixwebsocket`）会使用相同运行时，无需手动覆盖 `/MD`。
若工具链已设置 `CMAKE_MSVC_RUNTIME_LIBRARY`，LspCpp 会让捆绑依赖遵循该运行时。
使用 vcpkg 获取外部依赖时，请使用匹配的 CRT triplet（例如 `x64-windows-static`）。

在 Windows 上构建测试，请使用带 `tests` feature 的 vcpkg（见上文第 5 节）。

### 常用 CMake 选项

| 选项 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `LSPCPP_STANDALONE_ASIO` | `ON` | 使用 standalone Asio 而非 Boost.Asio |
| `LSPCPP_BUILD_MINIMAL_EXAMPLE` | `OFF` | 构建不依赖 Boost 的最小 stdio 示例 |
| `LSPCPP_BUILD_WEBSOCKETS` | `ON` | 构建 WebSocket 服务器支持 |
| `LSPCPP_BUILD_EXAMPLES` | `OFF` | 构建示例应用 |
| `LSPCPP_BUILD_TESTS` | `OFF` | 构建并注册 CTest 冒烟测试 |
| `LSPCPP_USE_CPP17` | `ON` | 以 C++17 编译 |
| `USE_SYSTEM_RAPIDJSON` | `OFF` | 使用系统 RapidJSON 而非 submodule |
| `USE_EXTERNAL_ASIO` | `OFF` | 使用 vcpkg/系统 Asio 而非 submodule |
| `USE_EXTERNAL_IXWEBSOCKET` | `OFF` | 使用 vcpkg/系统 IXWebSocket 而非 submodule |
| `LSPCPP_USE_STATIC_CRT` | `OFF` | 请求静态 MSVC 运行时链接（`/MT`、`/MTd`；仅 Windows） |

## 示例

示例应用位于 [examples](https://github.com/kuafuwang/LspCpp/tree/master/examples) 目录：

- `StdIOClientExample` / `StdIOServerExample` — stdio JSON-RPC
- `TcpServerExample` — TCP 服务器
- `WebsocketExample` — WebSocket 服务器

## 版本

LspCpp 对项目与包元数据使用语义化版本。当前项目版本为 **1.0.3**。

准备发布时：

- 更新 `CMakeLists.txt` 中的 `LSPCPP_VERSION`。
- 将 `vcpkg.json` 中的 `version-semver` 更新为相同值。
- 打 tag `vX.Y.Z`，例如 `v1.0.3`。
- GitHub release 标题使用相同版本号。

## 使用 LspCpp 的项目

- [JCIDE](https://www.javacardos.com/javacardforum/viewtopic.php?f=5&t=3569&sid=e01238adf55cd08696fbf495dfa6c8e5)
- [LPG-language-server](https://github.com/kuafuwang/LPG-language-server)
- [Asymptote](https://github.com/vectorgraphics)
- [chemical](https://github.com/chemicallang/chemical)

## 参考

部分代码来自 [cquery][1]。

## 许可证

MIT

## 开发指南

合并到 `master` 分支的 C++ 代码必须符合 clang-format 标准。当前使用 Ubuntu 24.04 提供的 clang-format 18，未来 Ubuntu 提供更新版本时可能会变更。

查看当前使用的 clang-format 版本，请参阅 `check-format-cpp` 工作流，其中会打印所用版本。请确保 C++ 代码符合该版本的 clang-format。

[1]: https://github.com/cquery-project/cquery "cquery"
