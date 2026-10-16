# 高级定制

本指南涵盖非标准 LSP 扩展，以及需要改变 LspCpp 对已存在方法的解析方式的情况。若协议本身不是 LSP，请使用[自定义协议框架](custom-protocol.md)中描述的更低层 JSON-RPC 层。

在以下场景使用这些 API：

- 添加自定义请求或通知，如 `workspace/index` 或 `$/myServer/progress`。
- 支持特定编辑器或语言服务器客户端使用的厂商扩展。
- 以不同于 LspCpp 内置结构体的方式解析标准 LSP 方法。
- 从服务器向客户端发送自定义请求或通知。

## 解析与 handler 如何配合

LspCpp 对每条入站消息有两块独立逻辑：

1. `MessageJsonHandler` 中的 JSON 解析器，将 method 名和 JSON body 转为类型化的 `LspMessage`。
2. `RemoteEndPoint` 中的 handler，接收该类型化 C++ 消息。

对于自定义 method，`server.on(...)` 通常两者都做：注册 handler，且若 method 未知则自动注册匹配的 JSON 解析器。

对于 `ProtocolJsonHandler` 已知的 method，仅注册自定义 handler **不足以**改变解析。你还必须在消息到达前覆盖解析器。

## 定义自定义请求

定义 params 和 result 结构体，做 reflection，然后使用 `DEFINE_REQUEST_RESPONSE_TYPE`。

```cpp
#include "LibLsp/LspCpp.h"
#include "LibLsp/JsonRpc/RequestInMessage.h"

#include <string>
#include <vector>

struct WorkspaceIndexParams
{
    std::string rootUri;
    bool rebuild = false;

    MAKE_SWAP_METHOD(WorkspaceIndexParams, rootUri, rebuild);
};
MAKE_REFLECT_STRUCT(WorkspaceIndexParams, rootUri, rebuild);

struct WorkspaceIndexResult
{
    bool accepted = false;
    std::vector<std::string> indexedFiles;

    MAKE_SWAP_METHOD(WorkspaceIndexResult, accepted, indexedFiles);
};
MAKE_REFLECT_STRUCT(WorkspaceIndexResult, accepted, indexedFiles);

DEFINE_REQUEST_RESPONSE_TYPE(
    WorkspaceIndexRequest, WorkspaceIndexParams, WorkspaceIndexResult, "workspace/index"
);
```

像内置请求一样处理：

```cpp
server.on(
    [](WorkspaceIndexRequest::request const& req)
        -> lsp::ResponseOrError<WorkspaceIndexRequest::response>
    {
        WorkspaceIndexRequest::response rsp;
        rsp.id = req.id;
        rsp.result.accepted = true;
        rsp.result.indexedFiles = { req.params.rootUri };
        return rsp;
    });
```

注册该 handler 时，`RemoteEndPoint` 发现 `workspace/index` 尚无解析器，会自动安装 `WorkspaceIndexRequest::request::ReflectReader`。

## 定义自定义通知

通知使用 `DEFINE_NOTIFICATION_TYPE`。

```cpp
#include "LibLsp/LspCpp.h"
#include "LibLsp/JsonRpc/NotificationInMessage.h"

#include <string>

struct IndexProgressParams
{
    std::string rootUri;
    int percent = 0;

    MAKE_SWAP_METHOD(IndexProgressParams, rootUri, percent);
};
MAKE_REFLECT_STRUCT(IndexProgressParams, rootUri, percent);

DEFINE_NOTIFICATION_TYPE(Notify_IndexProgress, IndexProgressParams, "$/indexProgress");
```

注册入站通知 handler：

```cpp
server.on(
    [](Notify_IndexProgress::notify const& n)
    {
        // n.params.rootUri 和 n.params.percent 可用。
    });
```

向客户端发送同一通知：

```cpp
Notify_IndexProgress::notify n;
n.params.rootUri = "file:///workspace";
n.params.percent = 50;
server.endpoint().send(n);
```

## 向客户端发送自定义请求

服务器到客户端的请求使用同一请求类型。用 `createRequest<T>()` 创建请求 id，填充 params，然后调用 `send()`。

```cpp
auto req = server.endpoint().createRequest<WorkspaceIndexRequest::request>();
req.params.rootUri = "file:///workspace";
req.params.rebuild = true;

auto future = server.endpoint().send(req);
future.wait();

auto result = future.get();
if (result.IsError())
{
    // result.error 包含 JSON-RPC 错误。
}
else
{
    // result.response.result 包含 WorkspaceIndexResult。
}
```

当某 method 尚无响应解析器时，`send()` 会自动为自定义请求类型注册响应解析器。

## 覆盖内置请求解析器

当 method 名已存在于 LspCpp 但你需要不同的参数类型时，使用 `overrideRequestParser()`。

例如，将 `initialize` 解析为自定义请求类型：

```cpp
struct CustomInitializeParams
{
    optional<lsDocumentUri> rootUri;
    optional<lsp::Any> initializationOptions;
    optional<std::string> clientName;

    MAKE_SWAP_METHOD(CustomInitializeParams, rootUri, initializationOptions, clientName);
};
MAKE_REFLECT_STRUCT(CustomInitializeParams, rootUri, initializationOptions, clientName);

DEFINE_REQUEST_RESPONSE_TYPE(
    CustomInitialize, CustomInitializeParams, InitializeResult, "initialize"
);

server.overrideRequestParser<CustomInitialize::request>();

server.on(
    [](CustomInitialize::request const& req)
    {
        CustomInitialize::response rsp;
        rsp.id = req.id;
        rsp.result.capabilities.hoverProvider = true;
        return rsp;
    });
```

解析器与 handler 必须就具体 C++ 类型达成一致。若解析器返回 `CustomInitialize::request`，该 method 的 handler 也应接受 `CustomInitialize::request const&`。

## 用自定义逻辑覆盖解析器

若 reflection 不够，可安装手动解析器：

```cpp
server.overrideRequestParser(
    CustomInitialize::request::kMethodInfo,
    [](Reader& visitor) -> std::unique_ptr<LspMessage>
    {
        return CustomInitialize::request::ReflectReader(visitor);
    });
```

当你需要规范化旧客户端、容忍厂商特定 JSON，或将多种 wire 形态路由到同一 C++ 请求类型时，这种形式很有用。

## 覆盖通知或响应解析器

`LanguageSession` 对请求暴露 `overrideRequestParser()`。对通知和响应，在启动 session 前直接使用底层 `ProtocolJsonHandler`。

```cpp
server.protocolJsonHandler()->SetNotificationJsonHandler(
    Notify_IndexProgress::notify::kMethodInfo,
    [](Reader& visitor)
    {
        return Notify_IndexProgress::notify::ReflectReader(visitor);
    });
```

响应：

```cpp
server.protocolJsonHandler()->SetResponseJsonHandler(
    WorkspaceIndexRequest::request::kMethodInfo,
    [](Reader& visitor)
    {
        if (visitor.HasMember("error"))
        {
            return Rsp_Error::ReflectReader(visitor);
        }
        return WorkspaceIndexRequest::response::ReflectReader(visitor);
    });
```

在 `start()`、`startStdio()` 或发送对应请求之前完成上述设置。

## 不改变解析，仅替换处理逻辑

若只想对内置 method 使用不同行为，保留内置解析器并为内置类型注册 handler：

```cpp
server.on(
    [](td_hover::request const& req)
    {
        td_hover::response rsp;
        rsp.id = req.id;
        // 用服务器要返回的 hover 内容填充 rsp.result。
        return rsp;
    });
```

此场景无需覆盖解析器。

## 顺序规则

- 在注册 handler 之前定义自定义消息类型。
- 在 `start()` 之前，或在需要自定义响应解析器的请求发送之前，注册解析器覆盖。
- 每个 method 名保持一种解析器和一种 handler 类型。用不兼容的 C++ 类型复用同一 method 字符串可能导致 dispatch 时无效转换。
- 扩展优先使用新 method 名。仅在必须支持非标准客户端 JSON 时才覆盖标准 LSP method。

## 常见模式

### 厂商扩展请求

使用新 method 名如 `myServer/analyzeProject` 和自定义请求类型。注册 handler 即可。

### 标准 method 带额外 params

使用相同 method 名，定义替换请求类型，调用 `overrideRequestParser<YourRequest>()`，再为 `YourRequest::request` 注册 handler。

### 自定义服务器进度

使用自定义通知如 `$/indexProgress`，在索引线程中调用 `server.endpoint().send(notification)`。若 handler 与后台工作并发运行，需保护共享状态。

## 相关文档

- [编写语言服务器](writing-a-language-server.md)
- [协议类型](../developer/protocol-types.md)
