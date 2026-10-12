# Advanced customization

This guide covers non-standard LSP extensions and cases where you need to change how LspCpp parses a method that already exists in the library. If you want a protocol that is not LSP at all, use the lower-level JSON-RPC layer described in [Custom protocol framework](custom-protocol.md).

Use these APIs when you need to:

- Add custom requests or notifications such as `workspace/index` or `$/myServer/progress`.
- Support vendor extensions used by a specific editor or language server client.
- Parse a standard LSP method differently from LspCpp's built-in struct.
- Send a custom request or notification from the server to the client.

## How parsing and handlers fit together

LspCpp has two separate pieces for every incoming message:

1. A JSON parser in `MessageJsonHandler` that turns a method name and JSON body into a typed `LspMessage`.
2. A handler in `RemoteEndPoint` that receives that typed C++ message.

For custom methods, `server.on(...)` usually does both: it registers the handler and, if the method is unknown, registers the matching JSON parser automatically.

For methods already known to `ProtocolJsonHandler`, registering a custom handler is not enough to change parsing. You must also override the parser before messages arrive.

## Define a custom request

Define params and result structs, reflect them, then use `DEFINE_REQUEST_RESPONSE_TYPE`.

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

Handle it like any built-in request:

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

When this handler is registered, `RemoteEndPoint` sees that `workspace/index` has no parser yet and installs `WorkspaceIndexRequest::request::ReflectReader` automatically.

## Define a custom notification

Notifications use `DEFINE_NOTIFICATION_TYPE`.

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

Register an incoming notification handler:

```cpp
server.on(
    [](Notify_IndexProgress::notify const& n)
    {
        // n.params.rootUri and n.params.percent are available here.
    });
```

Send the same notification to the client:

```cpp
Notify_IndexProgress::notify n;
n.params.rootUri = "file:///workspace";
n.params.percent = 50;
server.endpoint().send(n);
```

## Send a custom request to the client

Server-to-client requests use the same request type. Create a request id with `createRequest<T>()`, fill params, then call `send()`.

```cpp
auto req = server.endpoint().createRequest<WorkspaceIndexRequest::request>();
req.params.rootUri = "file:///workspace";
req.params.rebuild = true;

auto future = server.endpoint().send(req);
future.wait();

auto result = future.get();
if (result.IsError())
{
    // result.error contains the JSON-RPC error.
}
else
{
    // result.response.result contains WorkspaceIndexResult.
}
```

`send()` automatically registers the response parser for custom request types when no parser exists for the method.

## Override a built-in request parser

Use `overrideRequestParser()` when the method name already exists in LspCpp but you need a different parameter type.

For example, to parse `initialize` into a custom request type:

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

The parser and handler must agree on the concrete C++ type. If the parser returns `CustomInitialize::request`, the handler for that method should also accept `CustomInitialize::request const&`.

## Override a parser with custom logic

If reflection is not enough, install a manual parser:

```cpp
server.overrideRequestParser(
    CustomInitialize::request::kMethodInfo,
    [](Reader& visitor) -> std::unique_ptr<LspMessage>
    {
        return CustomInitialize::request::ReflectReader(visitor);
    });
```

This form is useful when you need to normalize old clients, tolerate vendor-specific JSON, or route multiple wire shapes into one C++ request type.

## Override notification or response parsers

`LanguageSession` exposes `overrideRequestParser()` for requests. For notifications and responses, use the underlying `ProtocolJsonHandler` directly before starting the session.

```cpp
server.protocolJsonHandler()->SetNotificationJsonHandler(
    Notify_IndexProgress::notify::kMethodInfo,
    [](Reader& visitor)
    {
        return Notify_IndexProgress::notify::ReflectReader(visitor);
    });
```

For responses:

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

Do this before `start()`, `startStdio()`, or sending the corresponding request.

## Replace handling without changing parsing

If you only want different behavior for a built-in method, keep the built-in parser and register a handler for the built-in type:

```cpp
server.on(
    [](td_hover::request const& req)
    {
        td_hover::response rsp;
        rsp.id = req.id;
        // Fill rsp.result with the hover content your server wants to return.
        return rsp;
    });
```

No parser override is needed in this case.

## Ordering rules

- Define custom message types before registering handlers for them.
- Register parser overrides before `start()` or before sending a request that needs the custom response parser.
- Keep one parser and one handler type per method name. Reusing the same method string with incompatible C++ types can lead to invalid casts during dispatch.
- Prefer new method names for extensions. Override standard LSP methods only when you must support non-standard client JSON.

## Common patterns

### Vendor extension request

Use a new method name such as `myServer/analyzeProject` and a custom request type. Registering the handler is enough.

### Standard method with extra params

Use the same method name, define a replacement request type, call `overrideRequestParser<YourRequest>()`, then register a handler for `YourRequest::request`.

### Custom server progress

Use a custom notification like `$/indexProgress` and call `server.endpoint().send(notification)` from the indexing thread. Protect shared state if handlers and background work run concurrently.

## Related docs

- [Writing a language server](writing-a-language-server.md)
- [Protocol types](../developer/protocol-types.md)
