# 传输方式

LspCpp 支持多种在客户端与服务器之间传输 JSON-RPC 消息的方式。

## stdio（推荐用于编辑器）

标准 LSP 传输方式：编辑器以子进程方式启动服务器，通过 stdin/stdout 通信。stderr 可用于日志输出。

```cpp
lsp::LanguageSession server;
server.startStdio();  // 等价于 start(make_stdin_stream(), make_stdout_stream())
```

也可以显式指定流：

```cpp
server.start(lsp::make_stdin_stream(), lsp::make_stdout_stream());
```

消息使用 LSP 协议定义的 Content-Length 帧格式。LspCpp 自动处理帧封装、解析与序列化。

## 自定义内存流（测试）

单元测试中可使用可注入输入流和可捕获输出流。参见 `tests/test_helpers.h` 中的 `FeedableIStream` 和 `StringOStream`：

```cpp
auto input = std::make_shared<test::FeedableIStream>();
auto output = std::make_shared<test::StringOStream>();
server.start(input, output);

input->append(test::MakeLspFrame(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})"));
// 检查 output->snapshot()
```

## TCP

远程或多客户端场景可使用 `LibLsp/JsonRpc/TcpServer.h` 中的 `TcpServer`。参见 `examples/TcpServerExample.cpp`：

```cpp
#include "LibLsp/JsonRpc/TcpServer.h"

lsp::TcpServer server("127.0.0.1", "9333", protocol_json_handler, endpoint, log);
server.point.registerHandler(/* ... */);
server.run();  // 阻塞直到调用 stop()
```

示例默认监听 `127.0.0.1:9333`，并在后台线程运行 `server.run()`，以便同一进程内的测试客户端连接。

默认配置下即包含 TCP 支持。

## WebSocket

当 `LSPCPP_BUILD_WEBSOCKETS=ON`（默认）时启用 WebSocket 传输。参见 `examples/WebsocketExample.cpp`。

WebSocket 依赖内置的 [IXWebSocket](https://github.com/machinezone/IXWebSocket)（或 `USE_EXTERNAL_IXWEBSOCKET=ON` 时使用系统/vcpkg 版本）。

如需减少依赖，可关闭 WebSocket：

```shell
cmake -DLSPCPP_BUILD_WEBSOCKETS=OFF ..
```

## JSON 流风格

`LanguageSession` 和 `RemoteEndPoint` 接受 `JSONStreamStyle` 参数：

```cpp
lsp::LanguageSession server(lsp::JSONStreamStyle::Standard);
```

`Standard` 用于常规 LSP Content-Length 帧格式。其他风格用于遗留或测试场景；大多数用户保持默认即可。

## 工作线程

`LanguageSession` 和 `RemoteEndPoint` 的 `max_workers` 参数（默认 `2`）控制并发处理消息的线程数。若 handler 为 CPU 密集型且相互独立，可适当增大；若 handler 共享可变状态且未同步，应保持较小值。

## 如何选择传输方式

| 传输方式 | 适用场景 |
| -------- | -------- |
| **stdio** | 编辑器集成（VS Code、Neovim 等） |
| **TCP** | 自定义客户端、netcat 调试、远程开发 |
| **WebSocket** | 浏览器编辑器、Web IDE |
| **自定义流** | 自动化测试、嵌入其他进程 |

生产环境的编辑器插件几乎总是选择 stdio。

## 可选 Transport 门面

`LibLsp/JsonRpc/Transport.h` 提供可选的薄门面，包装已有的 `RemoteEndPoint`。**它不能**替代 `LanguageSession`、`TcpServer` 或 `WebSocketServer`；网络生命周期与流连接仍由这些类型负责。

当你已持有 `RemoteEndPoint&`（例如 `TcpServer` 上的 `server.point`）并需要 clangd 风格的辅助 API 时，可以使用：

```cpp
#include "LibLsp/JsonRpc/Transport.h"

lsp::Transport transport(server.point);
transport.notify(exit_notify);
transport.reply(error_response);
auto future = transport.call(client_request);
transport.run(input, output);  // 或 transport.loop(...) — 与 run 相同的异步启动语义
transport.stop();
```

`run()` 与 `loop()` 均调用 `RemoteEndPoint::startProcessingMessages()` 并立即返回，不会在关闭前阻塞。除非明确需要底层 endpoint API，stdio 服务器仍应优先使用 `LanguageSession::start()`。
