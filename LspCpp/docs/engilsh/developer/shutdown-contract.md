# LanguageSession Shutdown Contract

This document describes the current LspCpp shutdown behavior. It records existing
semantics so embedders can rely on characterization tests rather than assuming
clangd-specific exit-code rules.

## Expected client sequence

1. `initialize` request/response
2. optional `initialized` notification
3. normal LSP traffic
4. `shutdown` request with `result: null`
5. `exit` notification

## Current LspCpp behavior

| Scenario | Current behavior |
| --- | --- |
| Clean `shutdown` then `exit` | `exit` notification handler runs; session stops normally. |
| `exit` without prior `shutdown` | `exit` notification handler still runs; LspCpp does not enforce a process exit code. |
| stdin EOF before any frame | message producer stops; no `exit` notification is dispatched. |
| unsupported request with id | JSON-RPC error `-32601` is returned when no handler/parser is registered. |
| endpoint `stop()` | pending outgoing requests are failed; processing threads are joined. |

## Opt-in limits

`RemoteEndPointLimits::max_pending_outgoing_requests` defaults to `0` (unlimited).
When set, new server-initiated client requests are rejected once the pending reply
map reaches the configured cap.

## Compatibility note

These semantics are covered by characterization tests adapted from clangd scenarios.
Future behavior changes must update both this document and the corresponding tests.
