# 构建与测试

开发并验证 LspCpp 本身变更的指南。

## 开发构建

```shell
mkdir _build && cd _build
cmake -DLSPCPP_BUILD_TESTS=ON ..
cmake --build . -j
```

`LSPCPP_BUILD_TESTS=ON` 会自动启用示例并注册全部 CTest target。
默认 `LSPCPP_BUILD_WEBSOCKETS=ON` 时当前注册 **25** 个 CTest 用例。`LSPCPP_BUILD_PERF_SMOKE=ON` 会添加 `lspcpp.perf_smoke`。

运行完整测试套件：

```shell
ctest --output-on-failure
```

运行单个测试：

```shell
ctest -R language_session --output-on-failure
```

或直接执行测试二进制（迭代更快）：

```shell
./lspcpp_language_session_tests
```

## 测试布局

测试是 `tests/` 中的独立可执行文件，不是单独的测试框架二进制。每个文件有 `main()`，运行命名测试函数，并通过 `tests/test_helpers.h` 报告失败。

| 测试二进制 | 重点 |
|------------|------|
| `lspcpp_language_session_tests` | `LanguageSession` API、生命周期、自定义解析器 |
| `lspcpp_remote_endpoint_tests` | `RemoteEndPoint` 分发与出站发送 |
| `lspcpp_jsonrpc_tests` | JSON-RPC 消息处理 |
| `lspcpp_stream_message_producer_tests` | 帧格式与流解析 |
| `lspcpp_endpoint_serializer_tests` | 序列化往返 |
| `lspcpp_infrastructure_tests` | 工具、Condition、流 |
| `lspcpp_working_files_tests` | 文档缓冲区管理 |
| `lspcpp_lsp_types_tests` | LSP 类型 JSON reflection |
| `lspcpp_protocol_json_handler_tests` | `ProtocolJsonHandler` 注册、allowlist 边界、golden LSP fixture |
| `lspcpp_clangd_fixture_tests` | clangd 衍生 golden fixture、session manifest、生命周期与错误码 |
| `lspcpp_lsp_3_16_17_tests` | 协议 3.16/3.17 类型 |
| `lspcpp_lsp_3_18_tests` | 协议 3.18 类型 |
| `lspcpp_tcp_write_queue_tests` | TCP 写队列 |
| `lspcpp_websocket_write_queue_tests` | WebSocket 写队列 |
| `lspcpp_api_compat_tests` | 可选取消、错误、Transport、请求上下文与 binder 兼容性 |
| `lspcpp_perf_smoke_tests` | 可选性能冒烟（默认关闭） |

共享辅助（`FeedableIStream`、`StringOStream`、`MakeLspFrame`）在 `tests/test_helpers.h`。
Protocol JSON 辅助（`ExpectParsesRequest`、`ExpectParsesResponse`、`ExpectParsesNotification`）在 `tests/protocol_test_helpers.h`。

## Protocol 覆盖策略

`ProtocolJsonHandler` 是 LSP method 的主要 wire-level 注册表。新增或修改已实现 LSP method 时：

1. 在 `src/lsp/ProtocolJsonHandler.cpp` 注册 request、response 或 notification parser。
2. 在 `tests/protocol_json_handler_tests.cpp`、`tests/lsp_types_roundtrip_tests.cpp` 或版本相关的 `tests/lsp_3_*_tests.cpp` 中添加/扩展解析断言。
3. 对用户可见或容易回归的 JSON 形状，在 `tests/fixtures/lsp/` 添加精简 golden 消息。
4. 不要把已实现 method 留在 `tools/lsp-metamodel-allowlist.json`。allowlist 表示“已知尚未实现”，不是“已实现但未测试”。

Protocol handler 测试会迭代当前已注册 parser map，并确认已注册 method 可用，且没有继续被 allowlist 标记为 missing。

以下为 consumer 项目与示例额外注册的 CTest 冒烟检查：

| CTest 名称 | 重点 |
|------------|------|
| `lspcpp.consumer_cmake_configure` | 配置示例 consumer 项目 |
| `lspcpp.consumer_cmake_build` | 从 consumer 构建构建 `consumer_smoke` 和 `MinimalStdIOServerExample` |
| `lspcpp.consumer_smoke_run` | 运行 consumer smoke 二进制 |
| `lspcpp.minimal_stdio_example_build` | 构建最小 stdio 示例 |
| `lspcpp.tcp_server_example` | 运行 TCP 示例冒烟测试 |
| `lspcpp.stdio_client_server_example` | 运行 stdio 客户端/服务器冒烟测试 |
| `lspcpp.websocket_example` | 启用 WebSocket 支持时运行 WebSocket 示例冒烟测试 |

## CMake consumer 测试

`tests/cmake/consumer_project/` 验证外部 CMake 项目能否通过 `add_subdirectory` 消费本源码树并链接 `lspcpp` target：

```shell
cmake -S tests/cmake/consumer_project -B _build_consumer \
  -DLSPCPP_SOURCE_DIR=/path/to/LspCpp
cmake --build _build_consumer
```

修改导出 target、target 名称或公共 `LibLsp/LspCpp.h` 入口时，请手动构建此测试。

## CI 配置

`.github/workflows/` 下的 GitHub Actions 工作流：

| 工作流 | 用途 |
|--------|------|
| `build-lsp-linux.yaml` | 使用 vcpkg preset 的 Linux 构建（C++17 含/不含 GC 矩阵，以及非 GC/非 WebSocket C++14 leg） |
| `build-lsp-windows.yaml` | Windows 构建 |
| `pull-req-precheck.yaml` | PR 验证 |
| `check-format-cpp.yaml` | clang-format 合规 |

本地 CI 等价构建：

```shell
export VCPKG_ROOT=/path/to/vcpkg
export LSPCPP_CI_VCPKG_FEATURES=tests
cmake --preset ci/default
cmake --build --preset ci/default -j
ctest --preset ci/default
```

C++14 兼容性验证（与 Linux 非 GC/非 WebSocket CI leg 一致）：

```shell
export VCPKG_ROOT=/path/to/vcpkg
export LSPCPP_CI_VCPKG_FEATURES=tests
cmake --preset ci/cpp14
cmake --build --preset ci/cpp14 -j
ctest --preset ci/cpp14 -R lspcpp --output-on-failure
```

## Sanitizer 构建

可选 sanitizer 标志，用于调试内存与线程问题：

| 选项 | Sanitizer |
|------|-----------|
| `LSPCPP_ASAN=ON` | AddressSanitizer |
| `LSPCPP_MSAN=ON` | MemorySanitizer |
| `LSPCPP_TSAN=ON` | ThreadSanitizer |

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_ASAN=ON ..
cmake --build . -j
```

不要在同一构建中同时启用 ASAN 和 TSAN。

## 性能冒烟测试

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_BUILD_PERF_SMOKE=ON ..
cmake --build . -j
ctest -R perf_smoke --output-on-failure
```

加上 `-DLSPCPP_PERF_WARNINGS_AS_ERRORS=ON` 可在性能警告时失败。

## Fuzzing

`LSPCPP_BUILD_FUZZER` 在 CMake 中声明为选项，但当前树尚未为其定义 fuzz target。在 CI 中依赖此选项前，需添加并文档化具体 target。

## 调试技巧

1. **使用 `StderrLog`** — 在示例或临时服务器中查看内部消息。
2. **运行单个测试二进制** — 按 `test_helpers.h` 风格添加临时 `Expect()` 调用。
3. **检查帧输出** — 测试通过 `StringOStream::snapshot()` 捕获 stdout；与预期 JSON 子串对比。
4. **编辑器 trace 日志** — 与真实编辑器集成时，开启 LSP trace，将 wire 格式与测试 fixture 对比。

## 添加新测试

1. 在 `tests/` 中创建或扩展文件。
2. 使用 `test::Expect(condition, "message")` 做断言；`test::Failures()` 返回失败计数。
3. 在 `CMakeLists.txt` 中注册可执行文件（参照现有 `add_executable` + `add_test` 块）。
4. 提交 PR 前运行 `ctest --output-on-failure`。
