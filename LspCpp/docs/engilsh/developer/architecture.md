# Architecture

This document describes how LspCpp is organized and how messages flow through the stack.

## Overview

LspCpp has two major layers:

```
┌─────────────────────────────────────────────────────────┐
│  LSP layer (include/LibLsp/lsp/)                        │
│  Typed requests/notifications, capabilities, helpers    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  JSON-RPC layer (include/LibLsp/JsonRpc/)               │
│  Framing, parsing, dispatch, endpoints, transports      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Third-party / bundled deps                             │
│  Asio, RapidJSON, utfcpp, IXWebSocket                   │
└─────────────────────────────────────────────────────────┘
```

## Directory layout

```
LspCpp/
├── include/LibLsp/
│   ├── LspCpp.h              # Convenience umbrella header
│   ├── JsonRpc/              # JSON-RPC infrastructure
│   └── lsp/                  # LSP protocol types and helpers
├── src/
│   ├── jsonrpc/              # JSON-RPC implementation (.cpp)
│   └── lsp/                  # LSP helpers (.cpp)
├── examples/                 # Sample servers and clients
├── tests/                    # CTest smoke tests
├── third_party/              # Bundled dependencies
├── ports/lspcpp/             # vcpkg overlay port
└── docs/                     # Documentation
```

## JSON-RPC layer

### Core components

| Component | Role |
|-----------|------|
| **`StreamMessageProducer`** | Reads content-length–framed message bodies from an `istream` |
| **`MessageJsonHandler`** | Maps method names to JSON parse/serialize functions |
| **`ProtocolJsonHandler`** | LSP-specific `MessageJsonHandler` with all standard methods registered |
| **`RemoteEndPoint`** | Incoming dispatch, outgoing requests, handler registration, worker threads |
| **`GenericEndpoint` / `Endpoint`** | Sends messages on the wire |
| **`stream.h`** | Abstract `istream`/`ostream`, stdio adapters, LSP framing |

### Message flow (server)

#### Inbound overview

```
stdin / TCP / WebSocket
        │
        ▼
  StreamMessageProducer  ──►  parse pool: parse JSON
                                      │
                                      ▼
                             sequence reorder buffer
        │                                              │
        │                                              ▼
        │                                    typed request/notification
        │                                              │
        ▼                                              ▼
 Ordered dispatcher                      RemoteEndPoint::registerHandler
        │                                              │
        ├── notifications: FIFO serial thread ─────────┤
        ├── opt-out notifications: handler pool ───────┤
        ├── requests: wait for prior notifications ────┘
        └── responses: exact pending completion
                                                       │
                                                       ▼
                                              user handler (lambda)
                                                       │
                                                       ▼
                                              serialize response
                                                       │
                                                       ▼
                                                   stdout / socket
```

#### Inbound pipeline (detailed)

Same visual style as the overview above. Parsing may run in parallel; **routing** and **ordered notifications** after `submitParsedMessage` advance serially by `sequence`.

**Threads and execution units**

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  reader thread ×1                                        │
                    │  message_producer.listen → seq++ → postParseTask         │
                    └──────────────────────────┬──────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────────────────────┐
                    │  parse pool ×N  (N = max_workers, 0→1)                     │
                    │  parseAndClassify ──► submitParsedMessage ──► route...    │
                    │  (routing runs synchronously inside parse tasks)           │
                    └───────┬──────────────────┬──────────────────┬────────────┘
                            │                  │                  │
              ┌─────────────▼───┐  ┌───────────▼──────────┐  ┌───▼──────────────┐
              │ notification ×1  │  │ handler pool ×N      │  │ async completion │
              │ notificationLoop │  │ requests / opt-out   │  │ ×1               │
              │ FIFO serial      │  │ / outbound callbacks │  │ outbound cb      │
              └─────────────────┘  └──────────────────────┘  └──────────────────┘
```

**Seven-stage main chain**

```
① frame read    ② backpressure   ③ parse          ④ reorder route
StreamMessage   postParseTask    parseAndClassify submitParsedMessage
Producer        acquireSlot      ParsedMessage    reorder buffer
   │                │                │                │
   ▼                ▼                ▼                ▼
 frame size      parse queue      JSON→typed       sync by seq
 assign seq      limit post       register cancel  routeParsedIncoming
   │                │                │                │
   └────────────────┴────────────────┴────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
⑤ type dispatch            ⑥ ordered dispatch         ⑦ handler execution
routeParsedIncoming        gate / parked_requests    postToWorker
   │                           │                           │
   ├── cancel → onCancel       ├── ordered N → notif thread └── mainLoopCatching
   ├── ordered N → enqueueN    ├── request → may park            → user handler
   ├── opt-out → handler pool  └── N done → release parked       → write response
   ├── request → enqueueR
   └── response → completePending
```

**② postParseTask / parse-queue backpressure**

```
postParseTask
    │
    ├─ quit or no parse_pool ───────────────────────────► drop
    │
    └─ acquireParseQueueSlot
            │
            ├─ queue not full ──► asio::post(parse pool)
            │                        │
            │                        ├─ parseAndClassify
            │                        └─ submitParsedMessage  ◄── normal path, once per message
            │
            └─ queue full
                    │
                    ├─ DropNewest ──► submitParsedMessage(empty placeholder)  no JSON parse
                    │
                    └─ StopProcessing (default) ──► stop session
```

**④ reorder buffer**

```
submitParsedMessage(seq, parsed)     [route_dispatch_mutex]
    │
    ├─ seq == next_route_sequence ──► ready[]  next++
    │
    └─ seq != next ──► reorder_buffer[seq]  wait for gap
                              │
                              ▼
                    drain next, next+1, … → ready[]
                              │
                              ▼
                    for item in ready:  routeParsedIncoming  (sync serial)
```

**⑤ routeParsedIncoming dispatch**

```
routeParsedIncoming
    │
    ├─ $/cancelRequest ──────────────► onCancel (side channel, no gate)
    │
    ├─ notification + ordered ────────► enqueueNotification ──► notification thread FIFO
    │
    ├─ notification + opt-out ────────► postToWorker ──────────► handler pool
    │
    ├─ request ───────────────────────► enqueueRequest
    │                                       │
    │                                       ├─ prior ordered N incomplete ──► parked_requests
    │                                       └─ no pending N ────────────────► handler pool
    │
    ├─ response (matches pending) ────► completePendingResponse
    │
    └─ other ─────────────────────────► postToWorker
```

**⑥ notification gate (wire: N1 → R1)**

```
wire:  N1 ──────► R1

route: enqueueNotification(N1)     last_seq = N1
              │
notif thread: handler(N1) ... done
              │                  completed_seq = N1
              │                  releaseParkedRequests
              │
route: enqueueRequest(R1)  ──if N1 incomplete──► parked (gate=N1)
              │
              └─ after N1 completes ─────────────► postToWorker(R1)

one-way gate:  N→R  request waits;  R→N  routing order (N not enqueued when R is routed)
```

**End-to-end timeline**

```
[reader]       seq++ ──► postParseTask
                              │
[parse pool]                  ▼
              parseAndClassify ──► submitParsedMessage ──► routeParsedIncoming
                              │
              ┌───────────────┼───────────────┬──────────────┐
              ▼               ▼               ▼              ▼
         onCancel      notification thread  handler pool  completePending
         (side)        + parked_requests   (request/     (outbound response)
                                          opt-out)
```

#### Ordering and concurrency (summary)

`RemoteEndPoint` keeps message reading separate from parsing and handler execution. Incoming message bodies are assigned a sequence number in wire order, parsed in a dedicated parse pool, then released by a reorder buffer so routing still observes wire order:

- ordinary notifications run on a dedicated FIFO notification thread, so order-sensitive notifications such as `textDocument/didOpen` and incremental `textDocument/didChange` are applied in wire order;
- methods explicitly marked with `allowConcurrentNotification` bypass the FIFO notification thread and run on the handler pool; these notifications are not ordered and do not gate following requests, so only order-insensitive handlers should opt in;
- requests are posted to the handler pool only after all earlier notifications have completed, so request handlers observe document state established by prior notifications;
- requests may still run concurrently with each other; the `max_workers` constructor parameter controls handler-pool concurrency;
- outbound responses are matched against the exact pending request captured during parsing. Future and blocking wait APIs complete their promise/condition directly in the routing path, without running user code, so a synchronous handler waiting for a response does not need a free handler worker. Callback-based send APIs preserve their callback semantics by deferring the user callback to the handler pool after the pending entry has been atomically removed;
- `$/cancelRequest` bypasses the notification queue. Cancellation is delayed only by parsing and sequence reordering of earlier frames, not by slow notification handlers. Parsing uses a separate pool so blocked synchronous handlers cannot starve cancellation parsing;
- each `startProcessingMessages` run owns an active session output state. Handlers capture that state, so responses from handlers that outlive `stop()` are dropped instead of being written to a later restarted session;
- embedders may configure `RemoteEndPointLimits` for maximum frame size, parse backlog, reorder buffer, FIFO notification queue, parked request queue, pending cancel table, and seen request-id table. By default all limits are disabled; when enabled, the default overload policy stops the current session rather than silently corrupting an ordered JSON-RPC stream;
- explicit handler registration, parser overrides, custom response parser registration, and `allowConcurrentNotification` are a pre-start contract. Runtime parsing and handler dispatch do not lock registration maps; `ProtocolJsonHandler` pre-registers standard LSP parsers during construction, and custom extensions must do the same before `startProcessingMessages()`. Registration APIs report misuse by returning `false`, logging a warning, and asserting in debug builds;
- if a real runtime registration requirement appears later, do not add mutexes to the parse/handler hot path. Prefer a snapshot-swap design: keep registration tables behind `std::shared_ptr<const map>` snapshots, update them with copy-on-write and an atomic pointer swap, and let readers pay only one atomic snapshot load.

### Transports

| File | Transport |
|------|-----------|
| `stream.h` / `StreamMessageProducer.cpp` | stdio and generic streams |
| `TcpServer.cpp` / `TcpServer.h` | TCP accept loop, one endpoint per connection |
| `WebSocketServer.cpp` | WebSocket (IXWebSocket) |

## LSP layer

### Protocol types

Each LSP method has C++ structs generated via macros in `LibLsp/JsonRpc/RequestInMessage.h` and `LibLsp/JsonRpc/NotificationInMessage.h`:

- **`DEFINE_REQUEST_RESPONSE_TYPE`** — request + response pair (e.g. `td_initialize`)
- **`DEFINE_NOTIFICATION_TYPE`** — client/server notifications

Types use RapidJSON reflection (`MAKE_REFLECT_STRUCT`, `ReflectReader`) for serialization.

Headers mirror the LSP spec structure:

- `general/` — lifecycle (`initialize`, `shutdown`, `exit`)
- `textDocument/` — document features
- `workspace/` — workspace features
- `client/` — client-side registration
- `extention/jdtls/` — Eclipse JDT.LS extensions (legacy)

### Helpers

| Module | Purpose |
|--------|---------|
| **`WorkingFiles`** | Track open buffers, incremental edits, line offsets |
| **`utils.cpp`** | URI handling, path normalization, UTF-8/UTF-16 conversion |
| **`lsp_diagnostic`** | Diagnostic construction helpers |
| **`Markup.cpp`** | MarkupContent for hover/documentation |
| **`ParentProcessWatcher`** | Detect when the parent editor process exits |

## LanguageSession

Added as a thin convenience wrapper (`include/LibLsp/lsp/LanguageSession.h`):

```cpp
LanguageSession
  ├── ProtocolJsonHandler   (shared, all LSP methods known)
  ├── GenericEndpoint       (outbound send)
  └── RemoteEndPoint        (inbound dispatch)
```

It does not add new protocol behavior; it reduces boilerplate for the common stdio-server case.

## Threading and cancellation

See **Inbound pipeline (detailed)** above: “Threads and execution units” and “End-to-end timeline”.

- **Frame read**: dedicated `message_producer` thread (not the handler pool).
- **Parse**: `asio::thread_pool` (`max_workers`).
- **Ordered notifications**: single-threaded FIFO `notificationLoop`.
- **Handlers**: `asio::thread_pool` (`max_workers`); requests may be gated in `parked_requests`.
- **Cancellation**: `$/cancelRequest` uses `onCancel()` side channel; not delayed by slow notifications.
- `CancelMonitor` is a callable passed to handlers for long requests; it reflects LSP `$/cancelRequest`.
- `Condition<T>` provides a simple wait/notify primitive used in examples for shutdown signaling.

## Optional: Boehm GC

When `LSPCPP_SUPPORT_BOEHM_GC=ON`, `GCThreadContext` integrates Boehm GC with the worker thread model. Most deployments leave this off.

## Design lineage

Parts of the codebase derive from [cquery](https://github.com/cquery-project/cquery). The JSON reflection and handler registration patterns follow that project's conventions.

## Related reading

- [Protocol types](protocol-types.md) — extending LSP message definitions
- [Build and test](build-and-test.md) — running tests that exercise these layers
