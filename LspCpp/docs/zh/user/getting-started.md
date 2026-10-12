# 快速入门

本指南介绍如何构建 LspCpp，并运行一个基于 stdio 的最小语言服务器——这也是大多数编辑器（VS Code、Neovim、Emacs 等）使用的传输方式。如果你不是在构建 LSP 服务器，而只是想要一个类型化的 JSON-RPC 消息框架，请参阅[自定义协议框架](custom-protocol.md)。

## 前置条件

- **CMake** 3.16 或更高版本
- 支持 **C++17** 的编译器（GCC、Clang 或 MSVC）
- 库本身和最小示例**不需要** Boost

可选（用于完整示例和测试）：

- **Boost**（`filesystem`、`program_options`、`system`）

## 构建库

```shell
mkdir _build && cd _build
cmake ..
cmake --build . -j
```

构建产物为 `liblspcpp.a`（Linux/macOS）或 `lspcpp.lib`（Windows），位于构建目录中。

## 构建最小示例

最小可运行服务器位于 `examples/MinimalStdIOServerExample.cpp`，不需要 Boost：

```shell
cmake -S . -B _build_minimal -DLSPCPP_BUILD_MINIMAL_EXAMPLE=ON
cmake --build _build_minimal --target MinimalStdIOServerExample
```

可执行文件输出到 `_build_minimal/MinimalStdIOServerExample`（Windows 上为 `.exe`）。

## 最小服务器代码

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

要点：

1. **`LanguageSession`** 封装 JSON-RPC I/O、消息解析与 handler 分发。
2. **`server.on(...)`** 注册 LSP 请求和通知 handler。handler 参数类型（如 `td_initialize::request`）决定处理哪个方法。
3. **`startStdio()`** 从 stdin 读取 LSP 帧，向 stdout 写入响应。
4. 按 LSP 关闭流程，收到 **`exit`** 通知后再调用 **`stop()`**。

## 在项目中链接 LspCpp

安装完成后（见[构建与安装](build-and-install.md)）：

```cmake
find_package(lspcpp CONFIG REQUIRED)

add_executable(my_language_server main.cpp)
target_link_libraries(my_language_server PRIVATE lspcpp::lspcpp)
```

也可以将 LspCpp 作为 CMake 子目录加入项目，直接链接 `lspcpp` target。

## 在编辑器中测试

大多数编辑器以子进程方式启动语言服务器，并通过 stdin/stdout 通信。在编辑器配置中指定构建出的二进制路径。例如在 VS Code 的 `settings.json` 中：

```json
{
  "languageServerExample.trace.server": "verbose",
  "languageServerExample.serverPath": "/path/to/MinimalStdIOServerExample"
}
```

具体配置取决于编辑器和 LSP 客户端扩展。通用说明见[构建与安装 — 编辑器集成](build-and-install.md#编辑器集成)。

## 下一步

- [编写语言服务器](writing-a-language-server.md) — 实现能力、处理文档事件、发送诊断
- [高级定制](advanced-customization.md) — 添加自定义消息、覆盖内置解析器
- [自定义协议框架](custom-protocol.md) — 不使用 LSP 专用消息类型
- [传输方式](transport.md) — TCP 与 WebSocket 服务器
- [构建与安装](build-and-install.md) — 全部 CMake 选项与 vcpkg 用法
