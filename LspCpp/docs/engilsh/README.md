# LspCpp Documentation

LspCpp is a C++ library for building [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/) servers and custom JSON-RPC protocols. It provides JSON-RPC transport, typed message definitions, and helpers for stdio, TCP, and WebSocket communication.

[中文文档](../zh/README.md)

## Documentation map

### For users (building language servers or custom protocols)

| Document | Description |
|----------|-------------|
| [Getting started](user/getting-started.md) | Install, build, and run your first stdio server |
| [Writing a language server](user/writing-a-language-server.md) | Handlers, lifecycle, capabilities, and common patterns |
| [Advanced customization](user/advanced-customization.md) | Custom messages, parser overrides, and vendor extensions |
| [Custom protocol framework](user/custom-protocol.md) | Use the JSON-RPC layer without LSP-specific types |
| [Transport options](user/transport.md) | stdio, TCP, WebSocket, and custom streams |
| [Build and install](user/build-and-install.md) | CMake options, vcpkg, packaging, and editor integration |

### For contributors (developing LspCpp itself)

| Document | Description |
|----------|-------------|
| [Architecture](developer/architecture.md) | Code layout, layers, and data flow |
| [Build and test](developer/build-and-test.md) | Local development, CTest, CI, and sanitizer builds |
| [Contributing](developer/contributing.md) | Code style, formatting, and pull request checklist |
| [Protocol types](developer/protocol-types.md) | How LSP request/notification types are defined and extended |

### Other

| Document | Description |
|----------|-------------|
| [LSP 3.18 coverage matrix](LSP_3_18_COVERAGE.md) | Protocol implementation coverage |

## Quick links

- Project README: [../../README.md](../../README.md)
- Minimal example: [../../examples/MinimalStdIOServerExample.cpp](../../examples/MinimalStdIOServerExample.cpp)
- LSP specification: https://microsoft.github.io/language-server-protocol/
