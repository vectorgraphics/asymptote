#!/bin/bash
# build-macos-native.sh
# 在 macOS (Apple Silicon / Intel) 上本地编译 Asymptote
#
# 用法：
#   chmod +x build-macos-native.sh
#   ./build-macos-native.sh
#
# 产物：当前目录下的 ./asy（macOS 原生可执行文件）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Asymptote macOS 本地构建脚本"
echo "========================================"

# ── 1. 检查 Xcode 命令行工具 ─────────────────────────────────────
if ! xcode-select -p >/dev/null 2>&1; then
    echo "错误：Xcode 命令行工具未安装。"
    echo "请运行：xcode-select --install"
    exit 1
fi

# ── 2. 检查 Homebrew ──────────────────────────────────────────────
if ! command -v brew >/dev/null 2>&1; then
    echo "错误：Homebrew 未安装。"
    echo "请访问 https://brew.sh 安装 Homebrew"
    exit 1
fi

# ── 3. 安装构建依赖 ──────────────────────────────────────────────
echo ""
echo "[1/5] 安装/检查 Homebrew 依赖..."

BREW_DEPS=(
    autoconf
    automake
    libtool
    bison
    pkg-config
    glfw
    vulkan-headers
    readline
)

# 可选依赖（增强功能）
OPTIONAL_DEPS=(
    gsl
    fftw
    eigen
    curl
)

for dep in "${BREW_DEPS[@]}"; do
    if ! brew list "$dep" >/dev/null 2>&1; then
        echo "  安装 $dep ..."
        brew install "$dep"
    else
        echo "  ✓ $dep 已安装"
    fi
done

for dep in "${OPTIONAL_DEPS[@]}"; do
    if brew list "$dep" >/dev/null 2>&1; then
        echo "  ✓ $dep 已安装（可选）"
    else
        echo "  ○ $dep 未安装（可选，跳过）"
    fi
done

# ── 4. 确保使用 Homebrew 的 bison ────────────────────────────────
# macOS 自带的 bison 2.3 太旧，需要 3.x
export PATH="/opt/homebrew/opt/bison/bin:$PATH"
# 清除 shell 命令缓存，否则 bash 仍可能找到系统自带的 bison 2.3
hash -r 2>/dev/null || true
if ! bison --version | grep -q "GNU Bison) 3"; then
    echo "错误：未能找到 Homebrew 安装的 Bison 3.x"
    echo "尝试路径: /opt/homebrew/opt/bison/bin/bison"
    ls -la /opt/homebrew/opt/bison/bin/bison 2>/dev/null || echo "该路径不存在"
    exit 1
fi
echo "  ✓ 使用 Bison: $(bison --version | head -1)"

# ── 5. 设置编译环境变量 ──────────────────────────────────────────
echo ""
echo "[2/5] 配置编译环境..."

# 包含 Homebrew 头文件和库路径
export CPPFLAGS="-I/opt/homebrew/include ${CPPFLAGS:-}"
export LDFLAGS="-L/opt/homebrew/lib ${LDFLAGS:-}"

# 检测架构
ARCH=$(uname -m)
echo "  架构: $ARCH"

# ── 6. 生成 configure ────────────────────────────────────────────
echo ""
echo "[3/5] 生成 configure 脚本..."
if [ ! -f ./configure ] || [ ./configure.ac -nt ./configure ]; then
    ./autogen.sh
else
    echo "  ✓ configure 已存在且最新"
fi

# ── 7. 运行 configure ────────────────────────────────────────────
echo ""
echo "[4/5] 运行 configure..."

# 配置选项说明：
# --disable-vulkan    禁用 Vulkan 3D 渲染（macOS 上需要完整 Vulkan SDK）
# --disable-gl        禁用 OpenGL
# --disable-lsp       禁用 LSP（需要额外的 LspCpp 子模块构建）
# 如果你需要这些功能，可以去掉对应的 --disable 选项并安装相应依赖

CONFIGURE_ARGS=(
    --disable-vulkan
    --disable-gl
    --disable-lsp
)

# Apple Silicon 不需要 universal binary；Intel Mac 也不需要
# 如果你需要同时支持两种架构，可以加上 --enable-macos-universal
if [ "$ARCH" = "arm64" ]; then
    echo "  检测到 Apple Silicon，构建 arm64 二进制"
else
    echo "  检测到 Intel Mac，构建 x86_64 二进制"
fi

echo "  configure 参数: ${CONFIGURE_ARGS[*]}"
./configure "${CONFIGURE_ARGS[@]}"

# ── 8. 编译 ──────────────────────────────────────────────────────
echo ""
echo "[5/5] 编译 asy..."

CPU_COUNT=$(sysctl -n hw.logicalcpu)
echo "  使用 $CPU_COUNT 个并行任务"

make -j"$CPU_COUNT" asy

# ── 9. 验证 ──────────────────────────────────────────────────────
echo ""
echo "========================================"
if [ -f ./asy ]; then
    echo "✅ 构建成功！"
    echo ""
    echo "二进制信息:"
    file ./asy
    echo ""
    echo "版本信息:"
    ./asy --version 2>&1 | head -3
    echo ""
    echo "使用方法:"
    echo "  ./asy --help              查看帮助"
    echo "  ./asy yourfile.asy        运行 asy 文件"
    echo ""
    echo "如需安装到系统:"
    echo "  make install              安装到 /usr/local（默认）"
    echo "  make install prefix=~/asy 安装到自定义目录"
else
    echo "❌ 构建失败，未找到 asy 二进制文件"
    exit 1
fi
