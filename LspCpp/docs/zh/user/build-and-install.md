# 构建与安装

构建、安装与消费 LspCpp 的完整参考。

## 依赖

### 核心（仅库）

| 依赖 | 来源 |
|------|------|
| [Asio](https://think-async.com/Asio/)（standalone） | `third_party/asio`，或 vcpkg / 系统包 |
| [RapidJSON](https://github.com/Tencent/rapidjson) | `third_party/rapidjson`，或系统包 |
| [utfcpp](https://github.com/nemtrif/utfcpp) | `third_party/utfcpp` |
| [IXWebSocket](https://github.com/machinezone/IXWebSocket) | `third_party/ixwebsocket`，或 vcpkg |
| ZLIB | 系统（IXWebSocket 压缩使用） |

当 `LSPCPP_STANDALONE_ASIO=ON`（默认）时，**不需要** Boost。

### 可选

| 依赖 | 何时需要 |
|------|----------|
| Boost（`filesystem`、`program_options`、`system`） | 示例与测试 |
| [Boehm GC](https://www.hboehm.info/gc/) | `LSPCPP_SUPPORT_BOEHM_GC=ON` |

## 基本构建

```shell
mkdir _build && cd _build
cmake ..
cmake --build . -j
```

默认构建类型为 `RelWithDebInfo`。可覆盖为：

```shell
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## 本地安装

```shell
cmake -S . -B _build -DLSPCPP_INSTALL=ON
cmake --build _build -j
cmake --install _build --prefix /path/to/install
```

然后在你的项目中：

```cmake
list(APPEND CMAKE_PREFIX_PATH "/path/to/install")
find_package(lspcpp CONFIG REQUIRED)
target_link_libraries(my_server PRIVATE lspcpp::lspcpp)
```

## CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `LSPCPP_STANDALONE_ASIO` | `ON` | 使用 standalone Asio 而非 Boost.Asio |
| `LSPCPP_BUILD_MINIMAL_EXAMPLE` | `OFF` | 构建不依赖 Boost 的最小 stdio 示例 |
| `LSPCPP_BUILD_WEBSOCKETS` | `ON` | 构建 WebSocket 服务器支持 |
| `LSPCPP_BUILD_EXAMPLES` | `OFF` | 构建示例应用 |
| `LSPCPP_BUILD_TESTS` | `OFF` | 构建并注册 CTest 冒烟测试 |
| `LSPCPP_BUILD_PERF_SMOKE` | `OFF` | 可选性能冒烟测试 |
| `LSPCPP_PERF_WARNINGS_AS_ERRORS` | `OFF` | 性能冒烟出现警告时失败 |
| `LSPCPP_USE_CPP17` | `ON` | 以 C++17 编译 |
| `LSPCPP_INSTALL` | `OFF` | 创建 install target |
| `LSPCPP_USE_STATIC_CRT` | `OFF` | 静态 MSVC 运行时（`/MT`，仅 Windows） |
| `LSPCPP_WARNINGS_AS_ERRORS` | `OFF` | 将编译器警告视为错误 |
| `LSPCPP_SUPPORT_BOEHM_GC` | `OFF` | 启用 Boehm GC 集成 |
| `USE_SYSTEM_RAPIDJSON` | `OFF` | 使用系统 RapidJSON |
| `USE_EXTERNAL_ASIO` | `OFF` | 使用 vcpkg/系统 Asio |
| `USE_EXTERNAL_IXWEBSOCKET` | `OFF` | 使用 vcpkg/系统 IXWebSocket |

启用 `LSPCPP_BUILD_TESTS` 会自动打开 `LSPCPP_BUILD_EXAMPLES`。

## 示例与测试

```shell
# Linux
sudo apt-get install libboost-filesystem-dev libboost-program-options-dev libboost-system-dev

# macOS
brew install boost

cmake -DLSPCPP_BUILD_TESTS=ON ..
cmake --build . -j
ctest --output-on-failure
```

默认 `LSPCPP_BUILD_WEBSOCKETS=ON` 时注册 **19** 个 CTest 用例。加上 `-DLSPCPP_BUILD_PERF_SMOKE=ON` 会多一个性能测试。

## vcpkg

仓库附带 `vcpkg.json` manifest：

```shell
export VCPKG_ROOT=/path/to/vcpkg
export LSPCPP_CI_VCPKG_FEATURES=tests   # 可选：测试/示例所需的 Boost
cmake --preset ci/default
cmake --build --preset ci/default -j
ctest --preset ci/default
```

通过 overlay port 安装：

```shell
vcpkg install lspcpp --overlay-ports=ports
```

## Windows 说明

生成 Visual Studio 解决方案：

```shell
mkdir _build && cd _build
cmake ..
cmake --build .
```

静态 CRT（`/MT`、`/MTd`）：

```shell
cmake -DLSPCPP_USE_STATIC_CRT=ON ..
```

Windows 上使用 vcpkg 时，需匹配 CRT triplet（如 `x64-windows-static`）。

## 版本

LspCpp 使用语义化版本。当前版本：**1.0.3**（见 `CMakeLists.txt` 中的 `LSPCPP_VERSION` 和 `vcpkg.json` 中的 `version-semver`）。

## 编辑器集成

用 LspCpp 构建的语言服务器就是普通可执行文件。编辑器的 LSP 客户端扩展需要：

1. **命令** — 服务器二进制路径
2. **参数** — 可选 CLI 标志（许多服务器使用 Boost `program_options`，见 `StdIOServerExample`）
3. **根目录** — 通常是工作区文件夹

服务器必须：

- 从 **stdin** 读取 LSP 消息
- 向 **stdout** 写入 LSP 消息
- **stderr** 仅用于日志（不能输出协议流量）

开发时在编辑器中开启 tracing；大多数 LSP 客户端可以记录原始 JSON-RPC 消息，便于调试 handler 注册和能力不匹配问题。

## 使用 LspCpp 的项目

- [JCIDE](https://www.javacardos.com/javacardforum/viewtopic.php?f=5&t=3569)
- [LPG-language-server](https://github.com/kuafuwang/LPG-language-server)
- [Asymptote](https://github.com/vectorgraphics)
- [chemical](https://github.com/chemicallang/chemical)
