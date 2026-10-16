# LSP 3.18 Coverage Matrix

This document tracks LspCpp protocol coverage against the official LSP 3.18
specification. Status values:

- `implemented`: type definitions are present and the method is registered by
  `ProtocolJsonHandler`.
- `type-only`: type definitions are present, but default protocol registration is
  missing.
- `partial`: some model or capability fields exist, but the feature is not
  complete.
- `missing`: no first-class model or default handler support is present.

## Method Coverage

| LSP Area | Method / Feature | Status | Notes |
| --- | --- | --- | --- |
| Lifecycle | `initialize`, `initialized`, `shutdown`, `exit` | implemented | Core lifecycle is registered. |
| Text sync | `didOpen`, `didChange`, `didSave`, `didClose`, `willSave`, `willSaveWaitUntil` | implemented | Classic text document sync is registered. |
| Completion | `textDocument/completion`, `completionItem/resolve` | implemented | 3.18 `CompletionList.applyKind` is modeled. |
| Hover/signature | `textDocument/hover`, `textDocument/signatureHelp` | implemented | `SignatureInformation.activeParameter` is modeled for 3.18. |
| Navigation | declaration, definition, typeDefinition, implementation, references | implemented | Registered by default. |
| Symbols | `textDocument/documentSymbol`, `workspace/symbol`, `workspaceSymbol/resolve` | implemented | Workspace symbol resolve is modeled and registered. |
| Code actions | `textDocument/codeAction`, `codeAction/resolve` | partial | Resolve request and kind documentation are modeled; main response remains the existing command-shaped API. |
| Code lens | `textDocument/codeLens`, `codeLens/resolve` | implemented | 3.18 resolve property enumeration is modeled. |
| Formatting | formatting, rangeFormatting, rangesFormatting, onTypeFormatting | implemented | 3.18 multi-range formatting is modeled and registered. |
| Document links | `textDocument/documentLink`, `documentLink/resolve` | implemented | Registered by default. |
| Folding range | `textDocument/foldingRange`, `workspace/foldingRange/refresh` | implemented | Refresh request is modeled and registered. |
| Semantic tokens | `textDocument/semanticTokens/full`, `full/delta` | implemented | Default handler registration is present. |
| Inlay hints | `textDocument/inlayHint`, `inlayHint/resolve` | implemented | Default handler registration is present. |
| Diagnostics | `textDocument/publishDiagnostics` | implemented | Pull diagnostics are newly added. |
| Diagnostics | `textDocument/diagnostic`, `workspace/diagnostic` | implemented | Added by this upgrade. |
| Inline values | `textDocument/inlineValue`, `workspace/inlineValue/refresh` | implemented | Added by this upgrade. |
| Inline completions | `textDocument/inlineCompletion` | implemented | Added by this upgrade. |
| Dynamic document content | `workspace/textDocumentContent`, `workspace/textDocumentContent/refresh` | implemented | Added by this upgrade. |
| Notebook documents | notebook document sync/filter models | partial | Filter and identifier models are present; sync notifications remain future work. |
| File operations | create/delete/rename capabilities | partial | Client capabilities exist; not every request/notification is modeled. |

## Capability / Type Coverage

| Type / Capability | Status | Notes |
| --- | --- | --- |
| Workspace edit annotations | implemented | 3.16 annotation support exists. |
| Workspace edit metadata | implemented | Added by this upgrade. |
| Snippet text edits | implemented | Added by this upgrade. |
| Relative pattern | implemented | Added by this upgrade. |
| Command tooltip | implemented | Added by this upgrade. |
| Debug message kind | implemented | Added by this upgrade. |
| Code lens resolve support | implemented | Added by this upgrade. |
| Inline completion capabilities | implemented | Added by this upgrade. |
| Pull diagnostic capabilities/providers | implemented | Added by this upgrade. |
| Text document content capability/provider | implemented | Added by this upgrade. |

## Automated verification

`tools/check_lsp_metamodel_coverage.py` compares `DEFINE_*` declarations and `ProtocolJsonHandler` registrations against a vendored LSP 3.18 metaModel snapshot. Known intentional gaps are tracked in `tools/lsp-metamodel-allowlist.json`.

The checker also surfaces a doc/source mismatch: `window/showMessageRequest` is a distinct request in the metaModel, but the current `WindowShowMessage` type still uses the `"window/showMessage"` method string (colliding with the `window/showMessage` notification). The request therefore remains classified as missing until that method string is corrected.

Opt-in parser groups are available via `ProtocolJsonHandlerOptions`:

- `enableExperimentalStandardRequests`: registers omitted standard request/response parsers such as `prepareRename`, `selectionRange`, `typeDefinition`, and `workspace/applyEdit`.
- `enableServerRefreshRequests`: registers server-initiated refresh request/response parsers such as `workspace/semanticTokens/refresh`, `workspace/inlayHint/refresh`, and `workspace/diagnostic/refresh`.
