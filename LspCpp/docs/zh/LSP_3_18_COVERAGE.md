# LSP 3.18 覆盖矩阵

本文档跟踪 LspCpp 相对官方 LSP 3.18 规范的协议覆盖情况。状态含义如下：

- `implemented`：类型定义已存在，且 `ProtocolJsonHandler` 已注册该方法。
- `type-only`：类型定义已存在，但缺少默认协议注册。
- `partial`：部分模型或能力字段已存在，但功能不完整。
- `missing`：缺少一等模型或默认 handler 支持。

## 方法覆盖

| LSP 领域 | 方法 / 功能 | 状态 | 说明 |
| --- | --- | --- | --- |
| 生命周期 | `initialize`、`initialized`、`shutdown`、`exit` | implemented | 核心生命周期已注册。 |
| 文本同步 | `didOpen`、`didChange`、`didSave`、`didClose`、`willSave`、`willSaveWaitUntil` | implemented | 经典文本文档同步已注册。 |
| 补全 | `textDocument/completion`、`completionItem/resolve` | implemented | 3.18 的 `CompletionList.applyKind` 已建模。 |
| Hover/签名 | `textDocument/hover`、`textDocument/signatureHelp` | implemented | 3.18 的 `SignatureInformation.activeParameter` 已建模。 |
| 导航 | declaration、definition、typeDefinition、implementation、references | implemented | 默认已注册。 |
| 符号 | `textDocument/documentSymbol`、`workspace/symbol`、`workspaceSymbol/resolve` | implemented | Workspace symbol resolve 已建模并注册。 |
| Code action | `textDocument/codeAction`、`codeAction/resolve` | partial | Resolve 请求与 kind 文档已建模；主响应仍沿用现有 command 形态 API。 |
| Code lens | `textDocument/codeLens`、`codeLens/resolve` | implemented | 3.18 resolve 属性枚举已建模。 |
| 格式化 | formatting、rangeFormatting、rangesFormatting、onTypeFormatting | implemented | 3.18 多范围格式化已建模并注册。 |
| 文档链接 | `textDocument/documentLink`、`documentLink/resolve` | implemented | 默认已注册。 |
| 折叠范围 | `textDocument/foldingRange`、`workspace/foldingRange/refresh` | implemented | Refresh 请求已建模并注册。 |
| 语义 token | `textDocument/semanticTokens/full`、`full/delta` | implemented | 默认 handler 注册已存在。 |
| Inlay hint | `textDocument/inlayHint`、`inlayHint/resolve` | implemented | 默认 handler 注册已存在。 |
| 诊断 | `textDocument/publishDiagnostics` | implemented | Pull diagnostics 已新增。 |
| 诊断 | `textDocument/diagnostic`、`workspace/diagnostic` | implemented | 本次升级新增。 |
| 内联值 | `textDocument/inlineValue`、`workspace/inlineValue/refresh` | implemented | 本次升级新增。 |
| 内联补全 | `textDocument/inlineCompletion` | implemented | 本次升级新增。 |
| 动态文档内容 | `workspace/textDocumentContent`、`workspace/textDocumentContent/refresh` | implemented | 本次升级新增。 |
| Notebook 文档 | notebook document sync/filter 模型 | partial | Filter 与 identifier 模型已存在；同步通知仍为后续工作。 |
| 文件操作 | create/delete/rename capabilities | partial | Client capabilities 已存在；并非所有 request/notification 都已建模。 |

## 能力 / 类型覆盖

| 类型 / 能力 | 状态 | 说明 |
| --- | --- | --- |
| Workspace edit annotations | implemented | 3.16 annotation 支持已存在。 |
| Workspace edit metadata | implemented | 本次升级新增。 |
| Snippet text edits | implemented | 本次升级新增。 |
| Relative pattern | implemented | 本次升级新增。 |
| Command tooltip | implemented | 本次升级新增。 |
| Debug message kind | implemented | 本次升级新增。 |
| Code lens resolve support | implemented | 本次升级新增。 |
| Inline completion capabilities | implemented | 本次升级新增。 |
| Pull diagnostic capabilities/providers | implemented | 本次升级新增。 |
| Text document content capability/provider | implemented | 本次升级新增。 |

## 自动化校验

`tools/check_lsp_metamodel_coverage.py` 会将 `DEFINE_*` 声明、`ProtocolJsonHandler` 注册与 vendored 的 LSP 3.18 metaModel 快照对比。已知缺口记录在 `tools/lsp-metamodel-allowlist.json`。

校验器还暴露了一处文档/源码不一致：`window/showMessageRequest` 在 metaModel 中是独立 request，但当前 `WindowShowMessage` 类型仍使用 `"window/showMessage"` 作为 method 字符串（与 `window/showMessage` notification 冲突）。该 request 因此仍归类为 `missing`，直到 method 字符串修正为止。
