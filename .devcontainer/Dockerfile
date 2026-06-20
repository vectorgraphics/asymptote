# Dev container image for building Asymptote with CMake + vcpkg.
#
# Based on Microsoft's C++ dev container image, which ships with GCC, git, and
# vcpkg pre-installed at /usr/local/vcpkg (with VCPKG_ROOT already exported). We
# pin Ubuntu 22.04 to match the GitHub Actions CI runners. Note that this base
# image's CMake (3.22, from the Ubuntu repos) is too old for our presets; a
# newer CMake is installed below.
FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04

# Controls whether the (large) LaTeX + Ghostscript toolchain needed for the
# `docgen` target (asymptote.pdf etc.) is baked into the image. On by default;
# set to "false" at build time for a leaner compiler-only image.
ARG INSTALL_DOC_DEPS=true

# Core build dependencies. These mirror the CI `initialize-linux-env` action:
#   - ninja-build, pkg-config, ccache: build tooling
#   - flex, bison: lexer/parser generation (camp.l, camp.y)
#   - perl, python3 (+ venv/pip): build-time code generation scripts and the
#     developer virtualenv created in post-create.sh
#   - libxinerama/libxcursor/xorg/libglu/libwayland-dev + autopoint: GLFW / GL
#     windowing dependencies pulled in by the vulkan/glfw3 vcpkg ports
#   - autoconf/automake/autoconf-archive + libtool/libtool-bin/libltdl-dev:
#     the `libxcrypt` vcpkg port regenerates its build system with
#     `autoreconf -vfi` (which needs autoconf/automake/autoconf-archive) and
#     `libtoolize` (libtool); libxcrypt also links against system libltdl.
#   - libglm-dev, libgsl-dev, libcurl4-openssl-dev, libreadline-dev,
#     libncurses-dev: system libraries for the `linux/sandbox` preset (and its
#     `linux/sandbox/devcontainer` relocation). Unlike `linux/release`, that
#     preset does NOT use vcpkg -- it builds from system deps only so it can run
#     in a no-network sandbox -- and these are the mandatory libraries (glm,
#     GSL, curl, readline, curses) that the production build would otherwise
#     pull from vcpkg. The release build is unaffected: it keeps getting these
#     from vcpkg via its toolchain file.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        ninja-build \
        pkg-config \
        ccache \
        flex \
        bison \
        perl \
        python3 \
        python3-venv \
        python3-pip \
        libxinerama-dev \
        libxcursor-dev \
        xorg-dev \
        libglu1-mesa-dev \
        libwayland-dev \
        autopoint \
        autoconf \
        automake \
        autoconf-archive \
        libtool \
        libtool-bin \
        libltdl-dev \
        bubblewrap \
        socat \
        libglm-dev \
        libgsl-dev \
        libcurl4-openssl-dev \
        libreadline-dev \
        libncurses-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# System Vulkan/GLFW for the autoconf (`./configure && make`) build.
#
# Unlike the CMake/vcpkg build -- which pulls the `vulkan` and `glfw3` ports
# from vcpkg -- the legacy `make` build links against *system* headers and
# libraries. configure.ac requires `vulkan/vulkan.hpp` to define
# VK_API_VERSION_1_4 (a C++ compile test); Ubuntu 22.04's stock libvulkan-dev
# is only 1.3.204, too old, so configure would silently build *without* the
# Vulkan/OpenGL renderer. The LunarG Vulkan SDK provides a current (>= 1.4)
# Vulkan-Hpp header set plus the glslang/SPIRV libraries the renderer links
# against; libglfw3-dev (from the stock repos) supplies GLFW. Pinned to a
# specific SDK release for reproducibility. Its own cached layer so the SDK is
# not re-downloaded on unrelated rebuilds.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends wget gnupg ca-certificates \
    && wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc \
        | tee /etc/apt/trusted.gpg.d/lunarg.asc \
    && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.4.313-jammy.list \
        https://packages.lunarg.com/vulkan/1.4.313/lunarg-vulkan-1.4.313-jammy.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        vulkan-sdk \
        libglfw3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Mark the pre-installed vcpkg clone as a safe git directory. The base image's
# vcpkg lives at /usr/local/vcpkg owned by root:vcpkg; the build runs as the
# `vscode` user (a member of the vcpkg group, so it can write), but git refuses
# to operate on a directory it does not own ("detected dubious ownership").
# That refusal makes vcpkg's *own* internal `git fetch` fail when it needs to
# pull the builtin-baseline pinned in vcpkg.json (which is newer than the
# commit baked into the base image's shallow clone), surfacing as:
#   error: ... failed to `git show` versions/baseline.json ... fix with `git fetch`
# Configuring this system-wide lets vcpkg fetch the baseline on demand.
RUN git config --system --add safe.directory /usr/local/vcpkg

# Modern CMake. The base image ships CMake 3.22 (Ubuntu 22.04), but our
# CMakePresets.json uses preset schema v6 and requires CMake >= 3.27, so
# `cmake --preset linux/release` fails with "Unrecognized version field" on the
# stock CMake. Install a current CMake system-wide via pip; the manylinux wheel
# drops a self-contained binary into /usr/local/bin, which precedes /usr/bin on
# PATH. Its own image layer, so it is cached and not re-downloaded on rebuilds.
RUN pip3 install --no-cache-dir "cmake>=3.27" \
    && cmake --version

# Documentation-generation toolchain (texlive is multiple GB). Baked into the
# image layer so it persists across container rebuilds without re-downloading.
#
# texlive-plain-generic supplies tex/generic/epsf/epsf.tex. Texinfo's @image
# macro aborts the build ("epsf.tex not found, images will be ignored") whenever
# \epsfbox is undefined -- even for PDF output, which never actually calls epsf
# (it uses \dopdfimage). With --no-install-recommends the stock texlive package
# does not pull epsf.tex in, so `make man` / asymptote.pdf fails without this.
RUN if [ "${INSTALL_DOC_DEPS}" = "true" ]; then \
        apt-get update && export DEBIAN_FRONTEND=noninteractive \
        && apt-get install -y --no-install-recommends \
            texlive \
            texlive-latex-extra \
            texlive-plain-generic \
            texinfo \
            ghostscript \
            dvisvgm \
        && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi
