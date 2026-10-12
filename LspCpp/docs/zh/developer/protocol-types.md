# 协议类型

LspCpp 如何在 C++ 中表示 LSP 消息以及如何扩展它们。

## 类型生成宏

请求和通知辅助宏位于 `include/LibLsp/JsonRpc/RequestInMessage.h` 和 `include/LibLsp/JsonRpc/NotificationInMessage.h`。Reflection 辅助位于 `include/LibLsp/JsonRpc/serializer.h`。

### 请求/响应对

```cpp
DEFINE_REQUEST_RESPONSE_TYPE(td_initialize, lsInitializeParams, InitializeResult, "initialize");
```

展开为：

- `td_initialize::request` — 入站请求结构体
- `td_initialize::response` — 出站响应结构体
- `td_initialize::request::kMethodInfo` — method 名常量（`"initialize"`）
- 通过 `ReflectReader` / `ToJson()` 的 JSON 解析/序列化钩子

### 通知

```cpp
DEFINE_NOTIFICATION_TYPE(Notify_TextDocumentDidOpen, TextDocumentDidOpen::Params, "textDocument/didOpen");
```

展开为 `Notify_TextDocumentDidOpen::notify`，无响应类型。

### 结构体 reflection

数据字段使用 RapidJSON reflection 宏：

```cpp
struct InitializeResult {
    lsServerCapabilities capabilities;
    MAKE_SWAP_METHOD(InitializeResult, capabilities);
};
MAKE_REFLECT_STRUCT(InitializeResult, capabilities);
```

`MAKE_REFLECT_STRUCT` 生成用于反序列化的 `ReflectReader` 和用于序列化的 `ToJson()`。

## 请求类型结构

生成的 request 继承自 `lsRequest<Params, request>`，后者继承自 `RequestInMessage`。概念上暴露：

```cpp
struct request : lsRequest<ParamsType, request> {
    static constexpr MethodType kMethodInfo = "textDocument/definition";
    using Response = response;
    // params 继承自 lsRequest<ParamsType, request>
    // id 和 method 继承自 RequestInMessage
    // ReflectReader、ToJson 等
};
```

Response 继承自 `ResponseMessage<ResultType, response>`，暴露 `result` 及继承的 JSON-RPC 响应字段：

```cpp
struct response : ResponseMessage<InitializeResult, response> {
    // result 继承自 ResponseMessage<InitializeResult, response>
    // id 继承自 ResponseInMessage
};
```

## Handler 注册

`RemoteEndPoint::registerHandler` 从 handler 签名推断 method：

```cpp
// Request: 第一个参数是 td_initialize::request const&
remote_end_point.registerHandler([](td_initialize::request const& req) {
    td_initialize::response rsp;
    rsp.id = req.id;
    return rsp;
});

// Notification: 第一个参数是 Notify_Exit::notify const&
remote_end_point.registerHandler([](Notify_Exit::notify const&) { });
```

返回类型：

- 请求：`T`、`lsp::ResponseOrError<T>`、`lsp::future<T>` 或 `lsp::future<lsp::ResponseOrError<T>>`
- 通知：`void`

可取消请求的第二个参数是 `CancelMonitor const&`，同步与异步 handler 均适用。

抛出 `lsp::RequestError` 时，handler 包装层会将其转换为 `Rsp_Error` 并发送。

## ProtocolJsonHandler

`src/lsp/ProtocolJsonHandler.cpp` 为每个已知 LSP method 注册 JSON 解析器。添加新 method 时：

1. 用 `DEFINE_*` 宏创建头文件。
2. 在 `src/lsp/ProtocolJsonHandler.cpp` 中包含该头文件。
3. 在相应辅助函数中注册解析器（`AddStandardRequestJsonRpcMethod`、`AddNotifyJsonRpcMethod` 或某响应辅助函数）。

运行时，`StreamMessageProducer` 在此 handler 中查找 method 字符串，在 dispatch 前反序列化 params。

## 命名约定

| 前缀 | 含义 | 示例 |
|------|------|------|
| `td_` | textDocument 请求 | `td_hover`、`td_completion` |
| `Notify_` | 通知 | `Notify_TextDocumentDidOpen` |
| 结构体 `ls` 前缀 | LSP 数据类型 | `lsPosition`、`lsRange`、`lsDiagnostic` |

Method 结构体名缩写 LSP 路径：

- `textDocument/definition` → `td_definition`
- `textDocument/didOpen` → `Notify_TextDocumentDidOpen`

发明新名前，先在 `include/LibLsp/lsp/` 下搜索现有头文件。

## 协议版本

LspCpp 跟踪多个 LSP 版本：

| 测试文件 | 覆盖 |
|----------|------|
| `lsp_types_roundtrip_tests.cpp` | 核心类型、通用往返 |
| `lsp_3_16_17_tests.cpp` | LSP 3.16 和 3.17 新增 |
| `lsp_3_18_tests.cpp` | LSP 3.18 新增 |

版本专用头文件包括 `protocol_3_18.h` 和功能门控结构体。为新规范版本添加类型时，放在适当命名的头文件中，并扩展对应测试文件。

## 可选与 variant 字段

LSP 可选字段使用代码库中的辅助类型：

- `optional<T>` 包装（见 `optionalVersion.h`；C++17 构建中别名 `std::optional`）
- `lsp::Any` 用于弱类型 JSON 值

同一头文件中现有结构体决定 nullable 与 omitted 字段语义。

## 自定义 JSON 解析

默认 `ReflectReader` 不足时：

```cpp
session.overrideRequestParser(
    "myExtension/doThing",
    [](Reader& visitor) -> std::unique_ptr<LspMessage> {
        // custom parse logic
    });
```

或覆盖类型化解析器：

```cpp
session.overrideRequestParser<td_initialize::request>();
```

谨慎使用 — 当 JSON 形态与规范一致时，优先修正结构体 reflection。

## 错误响应

用 `Rsp_Error` 返回错误：

```cpp
Rsp_Error err;
err.id = req.id;
err.error.code = lsErrorCodes::InternalError;
err.error.message = "something went wrong";
return err;
```

或使用 `lsp::ResponseOrError<T>`：

```cpp
return lsp::ResponseOrError<td_foo::response>(err);
```

也可以抛出 `lsp::RequestError`（`include/LibLsp/JsonRpc/RequestError.h`）：

```cpp
throw lsp::RequestError(lsErrorCodes::InvalidParams, "invalid parameters");
```

标准错误码在 `lsErrorCodes`（见 LSP 规范附录）。

## `optional<T>` 与 `Nullable<T>`

LSP 里很多字段需要区分 **字段缺省** 与 **字段存在且值为 JSON null**：

| 类型 | 对象成员缺省 | 对象成员显式 null | 顶层/数组元素 null |
|------|--------------|-------------------|--------------------|
| `optional<T>` | 省略 key | 不支持（缺省即 omit） | 写出 `null` |
| `Nullable<T>` | 不适用（成员总会写出） | 写出 `"key": null` | 写出 `null` |

- 规范里的 `field?: T` 用 `optional<T>`。
- 规范里的 `field: T \| null` 或必须区分 null 与缺省时，用 `Nullable<T>`（定义在 `LibLsp/JsonRpc/serializer.h`）。

例如 `workspace/workspaceFolders` 响应使用 `Nullable<std::vector<WorkspaceFolder>>`，可分别表示 `null`、空数组 `[ ]` 与文件夹列表。

## JDT.LS 扩展

`include/LibLsp/lsp/extention/jdtls/` 目录包含 Eclipse JDT Language Server 扩展，为基于 LspCpp 的 Java 工具保留兼容性。这些不属于标准 LSP 规范。
`ProtocolJsonHandler` 默认不注册这些方法。构建 Java 工具时需要显式开启：

```cpp
lsp::ProtocolJsonHandlerOptions options;
options.enableJdtlsExtensions = true;
lsp::ProtocolJsonHandler handler(options);
```

如果使用 `LanguageSession`，设置 `LanguageSessionOptions::protocol.enableJdtlsExtensions`。

## 测试新类型

添加将结构体序列化为 JSON 再解析回来的往返测试：

```cpp
// lsp_types_roundtrip_tests.cpp 中的模式
auto json = value.ToJson();
Reader reader(json);
auto parsed = SomeType::ReflectReader(reader);
Expect(parsed.field == value.field, "field must round-trip");
```

可尽早发现 reflection 宏错误。

## MetaModel 覆盖校验

LspCpp 在 `tools/check_lsp_metamodel_coverage.py` 中维护一个只读校验器，将手工维护的方法面与 vendored 的 LSP 3.18 metaModel 快照（`tools/lsp-metaModel-3.18.json`）对比。它会检查：

- 标准 LSP 方法是否缺少 `DEFINE_*` 类型声明
- 已声明的方法是否在 `ProtocolJsonHandler.cpp` 中注册了 request/response/notification 解析器
- 是否只有 response 解析器、缺少 request 解析器
- `ProtocolJsonHandler.cpp` 中是否存在重复注册

本地运行：

```shell
python3 tools/check_lsp_metamodel_coverage.py
```

探索性报告（不因已知缺口失败）：

```shell
python3 tools/check_lsp_metamodel_coverage.py --warn-only
```

开启测试构建后，CTest 也会运行：

```shell
ctest -R lspcpp.lsp_metamodel_coverage --output-on-failure
```

已知有意缺口记录在 `tools/lsp-metamodel-allowlist.json`。当你**有意**保留某个标准 LSP 缺口时，把对应 method 加入 allowlist；当你**新实现**某个标准 LSP 方法时，应同时更新类型声明、`ProtocolJsonHandler` 注册，并从 allowlist 中移除该项。若 allowlist 中的条目已不再是实际缺口（过期条目），校验器会在 `Stale allowlist entries` 一节报告并**失败**，以强制保持 allowlist 与实现同步。厂商扩展（`java/*`、`sonarlint/*` 等）单独分组，仅作信息报告，不会导致 CTest 失败。

## LSP 协议生成器（lspgen）

`tools/lspgen.py` 在 coverage 校验结果基础上，为**未在 allowlist 中**的标准 LSP 缺口生成 C++ 类型、`DEFINE_*` 宏，以及 `ProtocolJsonHandler.cpp` 注册补丁。生成物写入：

- `include/LibLsp/lsp/generated/lsp_generated_protocol.h`
- `src/lsp/ProtocolJsonHandler.cpp` 中带 `// BEGIN LSPGEN` 标记的注册块

默认是 dry-run，只打印计划与预览，不改文件：

```shell
python3 tools/lspgen.py
```

预览指定 method（忽略 allowlist，便于本地验证生成器）：

```shell
python3 tools/lspgen.py --method window/showDocument
```

确认无误后写入：

```shell
python3 tools/lspgen.py --method window/showDocument --write
```

写入后请：

1. 编译并跑相关测试，确认生成类型可编译、可 round-trip。
2. 从 `tools/lsp-metamodel-allowlist.json` 移除已实现 method。
3. 运行 `python3 tools/check_lsp_metamodel_coverage.py` 确认无 stale allowlist 条目。

复杂 union / literal 暂时会回退到 `lsp::Any` 并带 TODO 注释；生成后仍需人工 review。

## 延伸阅读

- [LSP 规范](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
- [架构](architecture.md) — 解析与分发在栈中的位置
- [编写语言服务器](../user/writing-a-language-server.md) — 在应用代码中使用类型
