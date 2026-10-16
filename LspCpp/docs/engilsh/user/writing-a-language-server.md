# Writing a language server

This guide covers the main API patterns for implementing an LSP server with LspCpp.

## Two API levels

LspCpp offers two ways to wire up a server:

| API | Header | When to use |
|-----|--------|-------------|
| **`LanguageSession`** | `LibLsp/LspCpp.h` | Recommended for new code. Thin wrapper around `RemoteEndPoint`. |
| **`RemoteEndPoint`** | `LibLsp/JsonRpc/RemoteEndPoint.h` | Lower-level control; used in older examples. |

Both use the same handler registration model. `LanguageSession` is an alias-friendly entry point:

```cpp
lsp::LanguageServer server;  // alias for LanguageSession
```

## Registering handlers

Handlers are registered with `on()` (LanguageSession) or `registerHandler()` (RemoteEndPoint). The compiler deduces the LSP method from the handler's parameter type.

### Request handlers

```cpp
server.on([](td_initialize::request const& req) -> lsp::ResponseOrError<td_initialize::response> {
    td_initialize::response rsp;
    rsp.id = req.id;
    rsp.result.capabilities.hoverProvider = true;
    return rsp;
});
```

Return `lsp::ResponseOrError<T>` to send either a success response or a JSON-RPC error. You can also return the response type directly; the library wraps it automatically in many cases.

Set `rsp.id = req.id` so the client can correlate the response.

### Request handlers with cancellation

Long-running requests can accept a `CancelMonitor`:

```cpp
server.on([](td_definition::request const& req, CancelMonitor const& monitor)
    -> lsp::ResponseOrError<td_definition::response>
{
    if (monitor && monitor()) {
        Rsp_Error err;
        err.error.code = lsErrorCodes::RequestCancelled;
        err.error.message = "cancelled";
        return err;
    }
    td_definition::response rsp;
    rsp.id = req.id;
    // ...
    return rsp;
});
```

### Async request handlers

Long-running requests can return `lsp::future<T>` or `lsp::future<lsp::ResponseOrError<T>>`. The library waits for the future on a background path and sends the response without blocking the dispatch worker:

```cpp
server.on([](td_hover::request const& req) -> lsp::future<td_hover::response> {
    auto promise = std::make_shared<lsp::promise<td_hover::response>>();
    auto future = promise->get_future();
    std::thread([promise, id = req.id]() {
        td_hover::response rsp;
        rsp.id = id;
        // ...
        promise->set_value(std::move(rsp));
    }).detach();
    return future;
});
```

Cancellable async handlers use the same second `CancelMonitor const&` parameter.

### Returning errors with RequestError

In addition to returning `lsp::ResponseOrError<T>` or `Rsp_Error` explicitly, you can throw `lsp::RequestError` and the library converts it to a JSON-RPC error response:

```cpp
#include "LibLsp/JsonRpc/RequestError.h"

server.on([](td_initialize::request const&) -> td_initialize::response {
    throw lsp::RequestError(lsErrorCodes::InvalidParams, "invalid initialize params");
});
```

`ResponseOrError<T>` remains the explicit, non-exception alternative.

### Notification handlers

Notifications have no response. Register them the same way:

```cpp
server.on([](Notify_Exit::notify const&) {
    // handle exit
});

server.on([](Notify_TextDocumentDidOpen::notify const& n) {
    // textDocument/didOpen
});
```

## LSP lifecycle

A well-behaved server follows this sequence:

```
Client                          Server
  |---- initialize ----------->  |
  |<--- InitializeResult ------|
  |---- initialized --------->  |  (notification, no response)
  |---- ... requests ... ---->  |
  |---- shutdown ------------>  |
  |<--- null ------------------|
  |---- exit ----------------->  |
  |                             | stop()
```

Implement at minimum:

```cpp
server.on([](td_initialize::request const& req) { /* return capabilities */ });
server.on([](td_shutdown::request const& req) {
    td_shutdown::response rsp;
    rsp.id = req.id;
    return rsp;
});
server.on([&](Notify_Exit::notify const&) {
    exit_requested.notify(std::make_unique<bool>(true));
});
```

Until `initialize` completes, the server should reject other requests (LspCpp enforces protocol parsing; your handlers should not assume documents are open before initialization).

## Declaring capabilities

Capabilities are returned in the `initialize` response:

```cpp
server.on([](td_initialize::request const& req) {
    td_initialize::response rsp;
    rsp.id = req.id;

    lsTextDocumentSyncOptions sync;
    sync.openClose = true;
    sync.change = lsTextDocumentSyncKind::Full;
    rsp.result.capabilities.textDocumentSync =
        std::make_pair(optional<lsTextDocumentSyncKind>(), optional<lsTextDocumentSyncOptions>(sync));

    lsCompletionOptions completion;
    completion.triggerCharacters = std::vector<std::string> { "." };
    rsp.result.capabilities.completionProvider = completion;

    rsp.result.capabilities.hoverProvider = true;

    WorkDoneProgressOptions definition_options;
    rsp.result.capabilities.definitionProvider =
        std::make_pair(optional<bool>(true), optional<WorkDoneProgressOptions>(definition_options));

    return rsp;
});
```

Only advertise features you actually implement. Clients use this list to decide which requests to send.

Several server capability fields mirror LSP union types. In this codebase those fields are represented as `std::pair<optional<...>, optional<...>>`; the first slot is usually the simple boolean/kind form and the second slot is the detailed options object.

## Sending outbound messages

Use `server.endpoint()` to send notifications or requests to the client:

```cpp
// Log to the client's output panel
Notify_LogMessage::notify log;
log.params.type = lsMessageType::Log;
log.params.message = "indexing complete";
session.endpoint().send(log);

// Publish diagnostics
Notify_TextDocumentPublishDiagnostics::notify diag;
diag.params.uri.raw_uri_ = "file:///path/to/file.lang";
diag.params.diagnostics = { /* lsDiagnostic entries */ };
session.endpoint().send(diag);
```

## Tracking open documents

LspCpp provides `WorkingFiles` to maintain in-memory buffers synchronized with LSP document events:

```cpp
#include "LibLsp/lsp/working_files.h"

WorkingFiles files;

server.on([&](Notify_TextDocumentDidOpen::notify const& n) {
    files.OnOpen(n.params.textDocument);
});

server.on([&](Notify_TextDocumentDidChange::notify const& n) {
    files.OnChange(n.params);
});

server.on([&](Notify_TextDocumentDidClose::notify const& n) {
    files.OnClose(n.params.textDocument);
});
```

`WorkingFile` stores buffer content and precomputed line offsets for converting LSP positions to byte offsets.

## Logging

Pass a custom `Log` implementation to `LanguageSession` for diagnostics during development:

```cpp
lsp::StderrLog log;
lsp::LanguageSession server(log);
```

Use `NullLog` (the default) in production if you do not need stderr output.

## Optional clangd-style helpers (opt-in)

These headers are additive; existing `LanguageSession`, `RemoteEndPoint`, `WorkingFiles`, and `ProtocolJsonHandler` defaults are unchanged.

### Request cancellation (`LibLsp/JsonRpc/RequestCancellation.h`)

Install a cancelable task scope around long-running work and poll cancellation from ambient `Context`:

```cpp
#include "LibLsp/JsonRpc/RequestCancellation.h"

server.on([](td_definition::request const& req) -> lsp::ResponseOrError<td_definition::response> {
    if (auto monitor = lsp::getCancelledMonitor(req.id)) {
        if (lsp::isCancellationRequested(*monitor)) {
            throw lsp::makeRequestCancelledError();
        }
    }
    // ...
});
```

`BindLspHandlerWithMonitor` and `BindLspAsyncHandlerWithMonitor` in `LibLsp/lsp/BindLspHandler.h` pass a `CancelMonitor` directly. Async binders preserve the handler's exact `lsp::future<T>` return type, including `lsp::future<lsp::ResponseOrError<T>>`.

### Error helpers (`LibLsp/JsonRpc/Error.h`)

Helpers for LSP-specific error codes such as `RequestCancelled (-32800)` and `ContentModified (-32801)`:

```cpp
#include "LibLsp/JsonRpc/Error.h"

throw lsp::makeRequestCancelledError();
throw lsp::makeContentModifiedError("buffer changed");
return lsp::makeRequestCancelledResponse(req.id);
```

**Malformed parameters** must still use `RequestError(lsErrorCodes::InvalidParams, ...)`. Cancellation and content-modified helpers do not replace `InvalidParams` handling.

### Request context (`LibLsp/lsp/request_context.h`)

Opt-in ambient offset encoding and path mapping for handler code:

```cpp
#include "LibLsp/lsp/request_context.h"

lsp::WithContext scope(lsp::withOffsetEncodingContext(lsp::OffsetEncoding::Utf8));
int offset = lsp::GetOffsetForPositionInContext(position, buffer);
```

UTF-16 remains the default when no context is installed.

## Custom request parsing

Override the built-in JSON parser for a method when you need non-standard parameter handling. The parser must return the same concrete request type accepted by the handler for that method:

```cpp
server.overrideRequestParser(
    td_initialize::request::kMethodInfo,
    [](Reader& visitor) {
        return td_initialize::request::ReflectReader(visitor);
    });
```

Or use the template shortcut:

```cpp
server.overrideRequestParser<td_initialize::request>();
```

For custom request/notification types, vendor extensions, and replacing built-in message shapes, see [Advanced customization](advanced-customization.md).

## Finding LSP type headers

Request and notification types follow a naming convention:

| Pattern | Example | LSP method |
|---------|---------|------------|
| `td_<feature>::request` | `td_definition::request` | `textDocument/definition` |
| `td_<feature>::response` | `td_definition::response` | (response struct) |
| `Notify_<Name>::notify` | `Notify_Exit::notify` | `exit` |

Headers are organized under `include/LibLsp/lsp/` by domain:

- `general/` — initialize, shutdown, exit
- `textDocument/` — completion, hover, rename, etc.
- `workspace/` — symbols, configuration, applyEdit

See [Protocol types](../developer/protocol-types.md) for how these types are generated and extended.

## Examples in the repository

| Example | Description |
|---------|-------------|
| `MinimalStdIOServerExample.cpp` | Smallest stdio server (no Boost) |
| `StdIOServerExample.cpp` | stdio with capabilities, definition handler, cancellation |
| `ProcessClientExample.cpp` | Lightweight client that launches a server with `lsp::Process` over stdio |
| `TcpServerExample.cpp` | TCP server on a configurable port |
| `WebsocketExample.cpp` | WebSocket server |
| `StdIOClientExample.cpp` | Boost.Process-based client JSON-RPC over stdio |

Build all examples with `-DLSPCPP_BUILD_EXAMPLES=ON` (requires Boost; `ProcessClientExample` only needs the in-tree `lsp::Process` helper).

## Launching a server process from a client

LSP clients usually spawn the language server and talk over stdio. Use `lsp::Process` (`LibLsp/JsonRpc/Process.h`):

```cpp
#include "LibLsp/JsonRpc/Process.h"

lsp::Process process = lsp::Process::start("/path/to/language-server", {"--stdio"});
RemoteEndPoint point(json_handler, endpoint, log);
point.startProcessingMessages(process.input(), process.output());
// send initialize, handle responses...
Notify_Exit::notify exit_notify;
point.send(exit_notify);
point.stop();
process.wait();
```

See `examples/ProcessClientExample.cpp` for a full walkthrough. With examples enabled, CTest runs:

```bash
ProcessClientExample --execPath StdIOServerExample
```
