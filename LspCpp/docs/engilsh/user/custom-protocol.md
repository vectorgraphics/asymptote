# Custom protocol framework

LspCpp is not limited to Language Server Protocol servers. The LSP layer is built on top of a reusable JSON-RPC message framework, so you can define your own protocol methods and use the same transport, framing, dispatch, cancellation, and typed serialization infrastructure.

Use this mode when you want:

- A private JSON-RPC protocol between two C++ processes.
- An editor-like plugin protocol that is not the standard LSP.
- A test harness, daemon, or tool server with typed request/response messages.
- LSP-compatible framing over stdio, TCP, WebSocket, or custom streams without using LSP method names.

## Layers you can reuse

The project has two useful layers:

| Layer | Main types | Use when |
|-------|------------|----------|
| LSP convenience layer | `lsp::LanguageSession`, `lsp::ProtocolJsonHandler` | You are building a normal LSP server or extending LSP. |
| Generic JSON-RPC layer | `MessageJsonHandler`, `GenericEndpoint`, `RemoteEndPoint` | You want to define your own protocol methods and message types. |

`LanguageSession` creates a `ProtocolJsonHandler`, which pre-registers many LSP methods. For a clean custom protocol, construct `RemoteEndPoint` yourself with a plain `MessageJsonHandler`.

## Minimal custom protocol server

This example defines a request named `tool/echo` and a notification named `tool/exit`. It does not depend on any LSP request or capability types.

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

`registerHandler()` installs both the message handler and the JSON parser for unknown custom methods. You do not need to manually populate `MessageJsonHandler` for simple custom request and notification types.

## Wire format

The default `JSONStreamStyle::Standard` transport uses LSP-style `Content-Length` framing, but the JSON-RPC payload can be your own protocol:

```text
Content-Length: 72

{"jsonrpc":"2.0","id":1,"method":"tool/echo","params":{"text":"hello"}}
```

The response uses the result type declared in `DEFINE_REQUEST_RESPONSE_TYPE`:

```text
Content-Length: 52

{"jsonrpc":"2.0","id":1,"result":{"text":"hello"}}
```

## Send custom messages

Use `RemoteEndPoint` the same way for outbound messages:

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

Notifications do not need request ids:

```cpp
Notify_ToolExit::notify notify;
endpoint.send(notify);
```

## Use different transports

The generic layer can use the same transports as the LSP layer:

- stdio: `endpoint.startProcessingMessages(lsp::make_stdin_stream(), lsp::make_stdout_stream())`
- custom streams: pass your own `std::shared_ptr<lsp::istream>` and `std::shared_ptr<lsp::ostream>`
- TCP: build around `lsp::TcpServer` and register handlers on `server.point`
- WebSocket: use the existing WebSocket server support when `LSPCPP_BUILD_WEBSOCKETS=ON`

The message types do not need to be LSP types. Only the JSON-RPC envelope and framing need to match what both sides expect.

## When to use `LanguageSession` instead

Use `lsp::LanguageSession` when your protocol is still mostly LSP:

- You need standard LSP methods such as `initialize`, `shutdown`, `textDocument/hover`, or diagnostics.
- You want built-in LSP request and notification parsers.
- You are writing a language server for an editor.

Use `RemoteEndPoint` with a plain `MessageJsonHandler` when you want a protocol that is independent from LSP.

## Design guidelines

- Pick stable method names, for example `tool/echo`, `daemon/build`, or `$/progress`.
- Keep request params and response results small and explicit.
- Version your custom protocol in an initialization request if clients and servers may be upgraded independently.
- Treat `Content-Length` framing as part of the transport contract; both peers must send complete JSON-RPC frames.
- If you later add standard LSP support, switch to `lsp::ProtocolJsonHandler` or `lsp::LanguageSession` and keep your custom methods alongside the LSP ones.

## Related docs

- [Advanced customization](advanced-customization.md)
- [Transport options](transport.md)
- [Protocol types](../developer/protocol-types.md)
