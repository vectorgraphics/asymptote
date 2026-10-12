# 贡献指南

向 LspCpp 提交变更的指南。

## 入门

1. Fork 并 clone 仓库。
2. 启用测试进行构建（见[构建与测试](build-and-test.md)）。
3. 在 feature 分支上修改。
4. 打开 pull request 前运行测试与格式检查。

## 代码风格

### C++ 标准

目标 **C++17**（默认 `LSPCPP_USE_CPP17=ON`）。库代码中若有标准库或 standalone-Asio 替代方案，避免使用 Boost。

### 格式化

合并到 `master` 的全部 C++ 代码必须通过 **clang-format**。CI 运行 `check-format-cpp` 工作流。

本地检查格式：

```shell
# 从 .github/workflows/check-format-cpp.yaml 查看 CI 使用的 clang-format 版本
clang-format --version

# 就地格式化文件
clang-format -i path/to/file.cpp

# 仅检查不修改
clang-format --dry-run -Werror path/to/file.cpp
```

项目当前与 Ubuntu 24.04 可用的 clang-format 版本对齐（clang-format 18）。若 CI 更新所需版本，请重新格式化。

### 代码库中的约定

- **命名空间**：面向库的 API 使用 `lsp`；部分遗留类型（`RemoteEndPoint`、`WorkingFiles`）在全局命名空间。
- **头文件**：`#pragma once`，include 分组（项目头文件，然后标准库）。
- **命名**：LSP 类型遵循生成名（`td_initialize`、`Notify_Exit`）。自由函数与辅助工具依模块使用 `PascalCase` 或 `snake_case` — 匹配你编辑的文件。
- **最小范围**：优先聚焦变更。不要格式化无关文件。

## Pull request 检查清单

请求 review 前：

- [ ] 代码在 `-DLSPCPP_BUILD_TESTS=ON` 下可构建
- [ ] `ctest --output-on-failure` 通过
- [ ] 对已修改的 `.cpp` / `.h` 文件应用 clang-format
- [ ] 新 LSP 类型在适用时有往返或 handler 测试
- [ ] 公共 API 变更已在 `docs/engilsh/` 或 `docs/zh/` 中文档化
- [ ] 仅在发布时 bump `CMakeLists.txt` 和 `vcpkg.json` 中的版本（不是每个 PR）

## 添加 LSP 协议支持

添加或更新 LSP method 时：

1. 在 `include/LibLsp/lsp/` 下适当头文件中定义类型。
2. 在 `ProtocolJsonHandler` 中注册解析/序列化 handler（见[协议类型](protocol-types.md)）。
3. 在 `tests/lsp_types_roundtrip_tests.cpp` 或版本专用测试文件中添加往返测试。
4. 若 method 受版本限制，在头文件注释中注明 LSP 规范版本。

## 依赖变更

捆绑依赖在 `third_party/`。优先通过 git submodule 更新，而非随意 vendoring 快照。

改用系统/vcpkg 包：

- `USE_SYSTEM_RAPIDJSON=ON`
- `USE_EXTERNAL_ASIO=ON`
- `USE_EXTERNAL_IXWEBSOCKET=ON`

在 `docs/engilsh/user/build-and-install.md`、`docs/zh/user/build-and-install.md` 和根 `README.md` 中文档化新 CMake 选项。

## 发布

LspCpp 使用[语义化版本](https://semver.org/)。发布步骤：

1. 更新 `CMakeLists.txt` 中的 `LSPCPP_VERSION`。
2. 更新 `vcpkg.json` 中的 `version-semver`。
3. 在 GitHub 上打 tag `vX.Y.Z`。
4. 发布说明中列出 API 或行为变更。

## 许可证

贡献在项目的 **MIT** 许可证下接受。提交 pull request 即表示你同意贡献以 MIT 许可。

## 参考

部分代码源自 [cquery](https://github.com/cquery-project/cquery)。修改这些文件时请保留现有版权声明。
