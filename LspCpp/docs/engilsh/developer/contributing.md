# Contributing

Guidelines for submitting changes to LspCpp.

## Getting started

1. Fork and clone the repository.
2. Build with tests enabled (see [Build and test](build-and-test.md)).
3. Make your changes in a feature branch.
4. Run tests and format checks before opening a pull request.

## Code style

### C++ standard

Target **C++17** (`LSPCPP_USE_CPP17=ON` by default). Avoid Boost in library code when a standard-library or standalone-Asio alternative exists.

### Formatting

All C++ code merged to `master` must pass **clang-format**. CI runs the `check-format-cpp` workflow.

Check formatting locally:

```shell
# Discover the clang-format version used in CI from .github/workflows/check-format-cpp.yaml
clang-format --version

# Format a file in place
clang-format -i path/to/file.cpp

# Check without modifying
clang-format --dry-run -Werror path/to/file.cpp
```

The project currently aligns with the clang-format version available on Ubuntu 24.04 (clang-format 18). Re-run format if CI updates the required version.

### Conventions observed in the codebase

- **Namespaces**: `lsp` for library-facing APIs; some legacy types (`RemoteEndPoint`, `WorkingFiles`) live in the global namespace.
- **Headers**: `#pragma once`, includes grouped (project headers, then standard library).
- **Naming**: LSP types follow generated names (`td_initialize`, `Notify_Exit`). Free functions and helpers use `PascalCase` or `snake_case` depending on the surrounding module—match the file you edit.
- **Minimal scope**: Prefer focused changes. Do not reformat unrelated files.

## Pull request checklist

Before requesting review:

- [ ] Code builds with `-DLSPCPP_BUILD_TESTS=ON`
- [ ] `ctest --output-on-failure` passes
- [ ] clang-format applied to changed `.cpp` / `.h` files
- [ ] New LSP types have round-trip or handler tests when applicable
- [ ] Public API changes documented in `docs/engilsh/` or `docs/zh/`
- [ ] Version bumped in `CMakeLists.txt` and `vcpkg.json` only for releases (not every PR)

## Adding LSP protocol support

When adding or updating LSP methods:

1. Define types in the appropriate header under `include/LibLsp/lsp/`.
2. Register parse/serialize handlers in `ProtocolJsonHandler` (see [Protocol types](protocol-types.md)).
3. Add round-trip tests in `tests/lsp_types_roundtrip_tests.cpp` or a version-specific test file.
4. Mention the LSP spec version in the header comment if the method is version-gated.

## Dependency changes

Bundled dependencies live in `third_party/`. Prefer updating via git submodules rather than vendoring arbitrary snapshots.

To use system/vcpkg packages instead:

- `USE_SYSTEM_RAPIDJSON=ON`
- `USE_EXTERNAL_ASIO=ON`
- `USE_EXTERNAL_IXWEBSOCKET=ON`

Document new CMake options in `docs/engilsh/user/build-and-install.md`, `docs/zh/user/build-and-install.md`, and the root `README.md`.

## Releases

LspCpp uses [semantic versioning](https://semver.org/). Release steps:

1. Update `LSPCPP_VERSION` in `CMakeLists.txt`.
2. Update `version-semver` in `vcpkg.json`.
3. Tag `vX.Y.Z` on GitHub.
4. Publish release notes listing API or behavior changes.

## License

Contributions are accepted under the project **MIT** license. By submitting a pull request, you agree that your contributions will be licensed under MIT.

## Reference

Some code originates from [cquery](https://github.com/cquery-project/cquery). Preserve existing copyright notices when modifying those files.
