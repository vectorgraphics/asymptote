# Build and test

Guide for developing and validating changes to LspCpp itself.

## Development build

```shell
mkdir _build && cd _build
cmake -DLSPCPP_BUILD_TESTS=ON ..
cmake --build . -j
```

`LSPCPP_BUILD_TESTS=ON` enables examples automatically and registers all CTest targets.
With the default `LSPCPP_BUILD_WEBSOCKETS=ON`, this currently registers 25 CTest cases. `LSPCPP_BUILD_PERF_SMOKE=ON` adds `lspcpp.perf_smoke`.

Run the full suite:

```shell
ctest --output-on-failure
```

Run a single test:

```shell
ctest -R language_session --output-on-failure
```

Or execute the test binary directly (faster iteration):

```shell
./lspcpp_language_session_tests
```

## Test layout

Tests are standalone executables in `tests/`, not a separate framework binary. Each file has a `main()` that runs named test functions and reports failures via `tests/test_helpers.h`.

| Test binary | Focus |
|-------------|-------|
| `lspcpp_language_session_tests` | `LanguageSession` API, lifecycle, custom parsers |
| `lspcpp_remote_endpoint_tests` | `RemoteEndPoint` dispatch and outbound send |
| `lspcpp_jsonrpc_tests` | JSON-RPC message handling |
| `lspcpp_stream_message_producer_tests` | Framing and stream parsing |
| `lspcpp_endpoint_serializer_tests` | Serialization round-trips |
| `lspcpp_infrastructure_tests` | Utilities, conditions, streams |
| `lspcpp_working_files_tests` | Document buffer management |
| `lspcpp_lsp_types_tests` | LSP type JSON reflection |
| `lspcpp_protocol_json_handler_tests` | `ProtocolJsonHandler` registrations, allowlist boundaries, golden LSP fixtures |
| `lspcpp_clangd_fixture_tests` | clangd-derived golden fixtures, session manifests, lifecycle and error codes |
| `lspcpp_lsp_3_16_17_tests` | Protocol 3.16/3.17 types |
| `lspcpp_lsp_3_18_tests` | Protocol 3.18 types |
| `lspcpp_tcp_write_queue_tests` | TCP write queue |
| `lspcpp_websocket_write_queue_tests` | WebSocket write queue |
| `lspcpp_api_compat_tests` | Opt-in cancellation, error, Transport, request context, and binder compatibility |
| `lspcpp_perf_smoke_tests` | Optional perf smoke (off by default) |

Shared helpers (`FeedableIStream`, `StringOStream`, `MakeLspFrame`) live in `tests/test_helpers.h`.
Protocol JSON helpers (`ExpectParsesRequest`, `ExpectParsesResponse`, `ExpectParsesNotification`) live in `tests/protocol_test_helpers.h`.

## Protocol coverage policy

`ProtocolJsonHandler` is the main wire-level registry for LSP methods. When adding or changing an implemented LSP method:

1. Register the request, response, or notification parser in `src/lsp/ProtocolJsonHandler.cpp`.
2. Add or extend a parser assertion in `tests/protocol_json_handler_tests.cpp`, `tests/lsp_types_roundtrip_tests.cpp`, or the version-specific `tests/lsp_3_*_tests.cpp` file.
3. Add a compact golden message to `tests/fixtures/lsp/` when the JSON shape is user-facing or historically easy to regress.
4. Do not leave implemented methods in `tools/lsp-metamodel-allowlist.json`. The allowlist means “known not implemented yet”, not “implemented but untested”.

The protocol handler test iterates the currently registered parser maps and checks that registered methods are usable and are not still marked as missing in the allowlist.

Additional CTest-only smoke checks are registered for the consumer project and examples:

| CTest name | Focus |
|------------|-------|
| `lspcpp.consumer_cmake_configure` | Configure the sample consumer project |
| `lspcpp.consumer_cmake_build` | Build `consumer_smoke` and `MinimalStdIOServerExample` from the consumer build |
| `lspcpp.consumer_smoke_run` | Run the consumer smoke binary |
| `lspcpp.minimal_stdio_example_build` | Build the minimal stdio example |
| `lspcpp.tcp_server_example` | Run the TCP example smoke test |
| `lspcpp.stdio_client_server_example` | Run the stdio client/server smoke test |
| `lspcpp.websocket_example` | Run the WebSocket example smoke test when WebSocket support is enabled |

## CMake consumer test

`tests/cmake/consumer_project/` verifies that an external CMake project can consume this source tree through `add_subdirectory` and link the `lspcpp` target:

```shell
cmake -S tests/cmake/consumer_project -B _build_consumer \
  -DLSPCPP_SOURCE_DIR=/path/to/LspCpp
cmake --build _build_consumer
```

Build this manually when changing exported targets, target names, or the public `LibLsp/LspCpp.h` entry point.

## CI configuration

GitHub Actions workflows under `.github/workflows/`:

| Workflow | Purpose |
|----------|---------|
| `build-lsp-linux.yaml` | Linux build with vcpkg preset (C++17 matrix with/without GC, plus non-GC/non-WebSocket C++14 leg) |
| `build-lsp-windows.yaml` | Windows build |
| `pull-req-precheck.yaml` | PR validation |
| `check-format-cpp.yaml` | clang-format compliance |

Local CI-equivalent build:

```shell
export VCPKG_ROOT=/path/to/vcpkg
export LSPCPP_CI_VCPKG_FEATURES=tests
cmake --preset ci/default
cmake --build --preset ci/default -j
ctest --preset ci/default
```

C++14 compatibility verification (matches the non-GC/non-WebSocket Linux CI leg):

```shell
export VCPKG_ROOT=/path/to/vcpkg
export LSPCPP_CI_VCPKG_FEATURES=tests
cmake --preset ci/cpp14
cmake --build --preset ci/cpp14 -j
ctest --preset ci/cpp14 -R lspcpp --output-on-failure
```

## Sanitizer builds

Optional sanitizer flags for debugging memory and threading issues:

| Option | Sanitizer |
|--------|-----------|
| `LSPCPP_ASAN=ON` | AddressSanitizer |
| `LSPCPP_MSAN=ON` | MemorySanitizer |
| `LSPCPP_TSAN=ON` | ThreadSanitizer |

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_ASAN=ON ..
cmake --build . -j
```

Do not combine ASAN and TSAN in the same build.

## Performance smoke tests

```shell
cmake -DLSPCPP_BUILD_TESTS=ON -DLSPCPP_BUILD_PERF_SMOKE=ON ..
cmake --build . -j
ctest -R perf_smoke --output-on-failure
```

Add `-DLSPCPP_PERF_WARNINGS_AS_ERRORS=ON` to fail on perf warnings.

## Fuzzing

`LSPCPP_BUILD_FUZZER` is declared as a CMake option, but this tree currently does not define a fuzz target for it. Add and document a concrete target before relying on this option in CI.

## Debugging tips

1. **Use `StderrLog`** in examples or a scratch server to see internal messages.
2. **Run tests with a single binary** and add temporary `Expect()` calls in `test_helpers.h` style.
3. **Inspect framed output** — tests capture stdout via `StringOStream::snapshot()`; compare against expected JSON substrings.
4. **Editor trace logs** — when integrating with a real editor, enable LSP trace to compare wire format against test fixtures.

## Adding a new test

1. Create or extend a file in `tests/`.
2. Use `test::Expect(condition, "message")` for assertions; `test::Failures()` returns the failure count.
3. Register the executable in `CMakeLists.txt` (follow an existing `add_executable` + `add_test` block).
4. Run `ctest --output-on-failure` before submitting a PR.
