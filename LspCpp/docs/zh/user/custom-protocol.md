# 自定义协议框架

LspCpp 不仅限于 Language Server Protocol 服务器。LSP 层构建在可复用的 JSON-RPC 消息框架之上，因此你可以定义自己的协议 method，并使用相同的传输、帧格式、分发、取消与类型化序列化基础设施。

在以下场景使用此模式：

- 两个 C++ 进程之间的私有 JSON-RPC 协议。
- 非标准 LSP 的类编辑器插件协议。
- 带类型化请求/响应消息的测试 harness、守护进程或工具服务器。
- 在 stdio、TCP、WebSocket 或自定义流上使用 LSP 兼容帧格式，但不使用 LSP method 名。

## 可复用的层

项目有两个实用层：

| 层 | 主要类型 | 适用场景 |
|----|----------|----------|
| LSP 便利层 | `lsp::LanguageSession`、`lsp::ProtocolJsonHandler` | 构建常规 LSP 服务器或扩展 LSP。 |
| 通用 JSON-RPC 层 | `MessageJsonHandler`、`GenericEndpoint`、`RemoteEndPoint` | 定义自己的协议 method 和消息类型。 |

`LanguageSession` 会创建 `ProtocolJsonHandler`，预注册大量 LSP method。若要干净的自定义协议，请自行用普通 `MessageJsonHandler` 构造 `RemoteEndPoint`。

## 最小自定义协议服务器

本示例定义名为 `tool/echo` 的请求和名为 `tool/exit` 的通知。不依赖任何 LSP 请求或 capability 类型。

```cpp
#include "LibLsp/JsonRpc/Condition.h"
#include "LibLsp/JsonRpc/Endpoint.h"
#include "LibLsp/JsonRpc/NotificationInMessage.h"
#include "LibLsp/JsonRpc/RemoteEndPoint.h"
#include "LibLsp/JsonRpc/RequestInMessage.h"
#include "LibLsp/JsonRpc/stream.h"

#include <memory>
#include <string>

struct EchoParams
{
    std::string text;
    MAKE_SWAP_METHOD(EchoParams, text);
};
MAKE_REFLECT_STRUCT(EchoParams, text);

struct EchoResult
{
    std::string text;
    MAKE_SWAP_METHOD(EchoResult, text);
};
MAKE_REFLECT_STRUCT(EchoResult, text);

DEFINE_REQUEST_RESPONSE_TYPE(Echo, EchoParams, EchoResult, "tool/echo");

struct ExitParams
{
    bool requested = true;
    MAKE_SWAP_METHOD(ExitParams, requested);
};
MAKE_REFLECT_STRUCT(ExitParams, requested);

DEFINE_NOTIFICATION_TYPE(Notify_ToolExit, ExitParams, "tool/exit");

int main()
{
    lsp::NullLog log;
    auto json_handler = std::make_shared<MessageJsonHandler>();
    auto local_endpoint = std::make_shared<GenericEndpoint>(log);
    RemoteEndPoint endpoint(json_handler, local_endpoint, log);
    Condition<bool> exit_requested;

    endpoint.registerHandler(
        [](Echo::request const& req)
        {
            Echo::response rsp;
            rsp.id = req.id;
            rsp.result.text = req.params.text;
            return rsp;
        });

    endpoint.registerHandler(
        [&](Notify_ToolExit::notify const&)
        {
            exit_requested.notify(std::make_unique<bool>(true));
        });

    endpoint.startProcessingMessages(lsp::make_stdin_stream(), lsp::make_stdout_stream());
    exit_requested.wait();
    endpoint.stop();
}
```

`registerHandler()` 会同时安装消息 handler 和未知自定义 method 的 JSON 解析器。简单自定义请求和通知类型无需手动填充 `MessageJsonHandler`。

## 线格式

默认 `JSONStreamStyle::Standard` 传输使用 LSP 风格的 `Content-Length` 帧格式，但 JSON-RPC 载荷可以是你的协议：

```text
Content-Length: 72

{"jsonrpc":"2.0","id":1,"method":"tool/echo","params":{"text":"hello"}}
```

响应使用 `DEFINE_REQUEST_RESPONSE_TYPE` 中声明的 result 类型：

```text
Content-Length: 52

{"jsonrpc":"2.0","id":1,"result":{"text":"hello"}}
```

## 发送自定义消息

出站消息同样通过 `RemoteEndPoint` 发送：

```cpp
auto req = endpoint.createRequest<Echo::request>();
req.params.text = "ping";

auto future = endpoint.send(req);
future.wait();

auto result = future.get();
if (!result.IsError())
{
    auto echoed = result.response.result.text;
}
```

通知不需要 request id：

```cpp
Notify_ToolExit::notify notify;
endpoint.send(notify);
```

## 使用不同传输

通用层可使用与 LSP 层相同的传输：

- stdio：`endpoint.startProcessingMessages(lsp::make_stdin_stream(), lsp::make_stdout_stream())`
- 自定义流：传入自己的 `std::shared_ptr<lsp::istream>` 和 `std::shared_ptr<lsp::ostream>`
- TCP：基于 `lsp::TcpServer` 构建，在 `server.point` 上注册 handler
- WebSocket：当 `LSPCPP_BUILD_WEBSOCKETS=ON` 时使用现有 WebSocket 服务器支持

消息类型不必是 LSP 类型。只需 JSON-RPC 信封和帧格式与双方预期一致。

## 何时改用 `LanguageSession`

当你的协议仍以 LSP 为主时使用 `lsp::LanguageSession`：

- 需要标准 LSP method，如 `initialize`、`shutdown`、`textDocument/hover` 或 diagnostics。
- 需要内置 LSP 请求和通知解析器。
- 为编辑器编写语言服务器。

当你想要与 LSP 无关的协议时，使用普通 `MessageJsonHandler` 的 `RemoteEndPoint`。

## 设计指南

- 选择稳定的 method 名，例如 `tool/echo`、`daemon/build` 或 `$/progress`。
- 保持请求 params 和响应 result 小而明确。
- 若客户端与服务器可能独立升级，在初始化请求中对自定义协议做版本化。
- 将 `Content-Length` 帧格式视为传输契约的一部分；双方必须发送完整 JSON-RPC 帧。
- 若后续添加标准 LSP 支持，切换到 `lsp::ProtocolJsonHandler` 或 `lsp::LanguageSession`，并将自定义 method 与 LSP method 并存。

## 相关文档

- [高级定制](advanced-customization.md)
- [传输方式](transport.md)
- [协议类型](../developer/protocol-types.md)
