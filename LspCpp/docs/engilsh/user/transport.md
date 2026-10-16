# Transport options

LspCpp supports several ways to move JSON-RPC messages between client and server.

## stdio (recommended for editors)

The standard LSP transport: the editor launches your server as a subprocess and communicates over stdin/stdout. stderr is free for logging.

```cpp
lsp::LanguageSession server;
server.startStdio();  // equivalent to start(make_stdin_stream(), make_stdout_stream())
```

Or with explicit streams:

```cpp
server.start(lsp::make_stdin_stream(), lsp::make_stdout_stream());
```

Messages use the LSP content-length header framing defined by the protocol. LspCpp handles framing, parsing, and serialization automatically.

## Custom in-memory streams (testing)

For unit tests, use feedable input and capturing output streams. See `tests/test_helpers.h` for `FeedableIStream` and `StringOStream`:

```cpp
auto input = std::make_shared<test::FeedableIStream>();
auto output = std::make_shared<test::StringOStream>();
server.start(input, output);

input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})"));
// inspect output->snapshot()
```

## TCP

For remote or multi-client scenarios, use `TcpServer` from `LibLsp/JsonRpc/TcpServer.h`. See `examples/TcpServerExample.cpp`:

```cpp
#include "LibLsp/JsonRpc/TcpServer.h"

lsp::TcpServer server("127.0.0.1", "9333", protocol_json_handler, endpoint, log);
server.point.registerHandler(/* ... */);
server.run();  // blocks until stop() is called
```

The example listens on `127.0.0.1:9333` by default and runs `server.run()` on a background thread so the test client can connect in the same process.

Build with the default configuration; TCP support is always included.

## WebSocket

WebSocket transport is built when `LSPCPP_BUILD_WEBSOCKETS=ON` (the default). See `examples/WebsocketExample.cpp`.

WebSocket support depends on the bundled [IXWebSocket](https://github.com/machinezone/IXWebSocket) library (or a system/vcpkg copy when `USE_EXTERNAL_IXWEBSOCKET=ON`).

Disable WebSocket support to reduce dependencies:

```shell
cmake -DLSPCPP_BUILD_WEBSOCKETS=OFF ..
```

## JSON stream styles

`LanguageSession` and `RemoteEndPoint` accept a `JSONStreamStyle` parameter:

```cpp
lsp::LanguageSession server(lsp::JSONStreamStyle::Standard);
```

Use `Standard` for normal LSP content-length framing. Other styles exist for legacy or test scenarios; most users should keep the default.

## Worker threads

Both `LanguageSession` and `RemoteEndPoint` accept a `max_workers` parameter (default `2`) controlling how many threads process incoming messages concurrently. Increase this if handlers are CPU-bound and independent; keep it low if handlers share mutable state without synchronization.

## Choosing a transport

| Transport | Best for |
|-----------|----------|
| **stdio** | Editor integration (VS Code, Neovim, etc.) |
| **TCP** | Custom clients, debugging with netcat, remote development |
| **WebSocket** | Browser-based editors, web IDEs |
| **Custom streams** | Automated tests, embedding in another process |

For production editor plugins, stdio is almost always the right choice.

## Optional Transport facade

`LibLsp/JsonRpc/Transport.h` provides an optional thin facade over an existing `RemoteEndPoint`. It does **not** replace `LanguageSession`, `TcpServer`, or `WebSocketServer`; those types still own network lifecycle and stream wiring.

Use it when you already hold a `RemoteEndPoint&` (for example `server.point` on `TcpServer`) and want clangd-style helpers:

```cpp
#include "LibLsp/JsonRpc/Transport.h"

lsp::Transport transport(server.point);
transport.notify(exit_notify);
transport.reply(error_response);
auto future = transport.call(client_request);
transport.run(input, output);  // or transport.loop(...) — same async-start semantics
transport.stop();
```

`run()` and `loop()` both call `RemoteEndPoint::startProcessingMessages()` and return immediately; they do not block until shutdown. Prefer `LanguageSession::start()` for stdio servers unless you explicitly want the lower-level endpoint API.
