# LspCpp 文档（中文）

LspCpp 是一个用于构建 [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/) 服务器和自定义 JSON-RPC 协议的 C++ 库。它提供 JSON-RPC 传输、类型化消息定义，以及 stdio、TCP、WebSocket 通信辅助功能。

[English documentation](../engilsh/README.md)

## 文档目录

### 用户文档（构建语言服务器或自定义协议）

| 文档 | 说明 |
| ---- | ---- |
| [快速入门](user/getting-started.md) | 安装、构建并运行第一个 stdio 服务器 |
| [编写语言服务器](user/writing-a-language-server.md) | Handler、生命周期、能力声明与常见模式 |
| [高级定制](user/advanced-customization.md) | 自定义消息、覆盖解析器与厂商扩展 |
| [自定义协议框架](user/custom-protocol.md) | 不依赖 LSP 专用类型的 JSON-RPC 层用法 |
| [传输方式](user/transport.md) | stdio、TCP、WebSocket 与自定义流 |
| [构建与安装](user/build-and-install.md) | CMake 选项、vcpkg、打包与编辑器集成 |

### 开发者文档（开发 LspCpp 本身）

| 文档 | 说明 |
| ---- | ---- |
| [架构](developer/architecture.md) | 代码布局、分层与消息流 |
| [构建与测试](developer/build-and-test.md) | 本地开发、CTest、CI 与 Sanitizer 构建 |
| [贡献指南](developer/contributing.md) | 代码风格、格式化与 PR 检查清单 |
| [协议类型](developer/protocol-types.md) | LSP 请求/通知类型的定义与扩展方式 |

### 其他

| 文档 | 说明 |
| ---- | ---- |
| [LSP 3.18 覆盖矩阵](LSP_3_18_COVERAGE.md) | 协议实现覆盖情况 |

## 快速链接

- 项目 README：[../../README_zh.md](../../README_zh.md)（[English](../../README.md)）
- 最小示例：[../../examples/MinimalStdIOServerExample.cpp](../../examples/MinimalStdIOServerExample.cpp)
- LSP 规范：https://microsoft.github.io/language-server-protocol/
