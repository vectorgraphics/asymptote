# Usage:
#   docker build --target builder --network host \
#     --build-arg http_proxy=http://127.0.0.1:33210 \
#     --build-arg https_proxy=http://127.0.0.1:33210 \
#     -f Dockerfile.dev -t asy-builder .
#
#   docker run --rm -v "$PWD":/workspace asy-builder sh -c '
#     cmake --preset linux/release/ci/with-ccache/docker &&
#     cmake --build --preset linux/release/ci/with-ccache/docker \
#       --target asy-with-basefiles -j$(nproc)
#   '
#
# Build docs (requires WITH_DOCS=1):
#   cmake --build --preset linux/release/ci/with-ccache/docker \
#     --target docgen -j$(nproc)

FROM ubuntu:22.04 AS builder

ARG WITH_DOCS=1
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG NO_PROXY
ARG no_proxy

# Proxy vars are intentionally set as ENV (not ARG) so that *every* RUN
# step that touches the network picks them up automatically. They are
# cleared at the end of the build so they do NOT leak into the final
# image and break runtime when the proxy is not available.
ENV DEBIAN_FRONTEND=noninteractive \
    HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    NO_PROXY="${NO_PROXY},mirrors.aliyun.com" \
    no_proxy="${no_proxy},mirrors.aliyun.com"

# ── 0. Base dependencies ───────────────────────────────────────────
# Single RUN for all apt packages that do not require extra apt repos.
# Aliyun mirror is used for speed; it is added to no_proxy above.
RUN sed -i \
      -e 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
      -e 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' \
      /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       g++ ninja-build flex bison pkg-config autopoint \
       git ca-certificates curl unzip tar gnupg perl \
       python3 python3-jinja2 \
       libncursesw5-dev libxinerama-dev libxcursor-dev \
       xorg-dev libglu1-mesa-dev libwayland-dev libtirpc-dev \
       ccache autoconf automake autoconf-archive libtool libltdl-dev \
       zip gettext make \
       $(if [ "$WITH_DOCS" = "1" ]; then \
            echo "texlive texlive-latex-extra texlive-plain-generic texinfo ghostscript"; \
          fi) \
    && rm -rf /var/lib/apt/lists/*

# ── 1. CMake >= 3.27 ──────────────────────────────────────────────
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${UBUNTU_CODENAME} main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Ninja (v1.13.2+) ───────────────────────────────────────────
# apt's ninja-build 1.10.x is too old for vcpkg compiler detection.
RUN curl -fsSL "https://github.com/ninja-build/ninja/releases/download/v1.13.2/ninja-linux.zip" \
      -o /tmp/ninja.zip && \
    unzip -o /tmp/ninja.zip -d /usr/local/bin && \
    rm /tmp/ninja.zip && \
    chmod +x /usr/local/bin/ninja

# ── 3. vcpkg bootstrap ────────────────────────────────────────────
ENV VCPKG_ROOT=/opt/vcpkg
RUN for i in 1 2 3 4 5; do \
      echo "==> vcpkg clone attempt $i/5"; \
      git clone --depth 1 https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT" && break; \
      echo "clone failed, retrying in 10s..."; sleep 10; \
    done \
    && test -f "$VCPKG_ROOT"/bootstrap-vcpkg.sh || { echo "ERROR: vcpkg clone failed after 5 attempts"; exit 1; } \
    && cd "$VCPKG_ROOT" \
    && for i in 1 2 3 4 5; do \
      echo "==> vcpkg baseline fetch attempt $i/5"; \
      git fetch --depth 1 origin f33cc491c85a7d643c5ab6da1667c1458e6d7abf && break; \
      echo "fetch failed, retrying in 10s..."; sleep 10; \
    done \
    && for i in 1 2 3 4 5; do \
      echo "==> vcpkg bootstrap attempt $i/5"; \
      ./bootstrap-vcpkg.sh && break; \
      echo "bootstrap failed, retrying in 10s..."; sleep 10; \
    done

# ── 4. Pre-build vcpkg dependencies ───────────────────────────────
# libxcrypt autotools configure needs a no-op dlltool and 'make'.
RUN ln -s /usr/bin/true /usr/local/bin/dlltool

# Build release only to cut image build time roughly in half.
RUN echo 'set(VCPKG_BUILD_TYPE release)' >> "$VCPKG_ROOT"/triplets/x64-linux.cmake

COPY vcpkg.json /tmp/vcpkg-deps/vcpkg.json
RUN cd /tmp/vcpkg-deps && \
    ( "$VCPKG_ROOT"/vcpkg install \
        --triplet x64-linux \
        --x-feature=readline \
        --x-feature=curl \
        --x-feature=threading \
        --x-feature=gsl \
        --x-feature=eigen3 \
        --x-feature=fftw3 \
        --x-feature=lsp \
        --x-install-root=/opt/vcpkg-deps \
      || ( echo "=== LIBCRYPT BUILD LOGS ==="; \
           find /opt/vcpkg/buildtrees/libxcrypt -name "*.log" \
             -exec echo "--- {} ---" \; -exec tail -100 {} \; ; \
           exit 1 ) \
    ) && \
    rm -rf /tmp/vcpkg-deps

# Shrink layer: remove vcpkg intermediate build artifacts (~2 GB).
RUN rm -rf \
    "$VCPKG_ROOT"/buildtrees \
    "$VCPKG_ROOT"/downloads \
    "$VCPKG_ROOT"/packages

# ── 5. ccache ─────────────────────────────────────────────────────
ENV CCACHE_DIR=/home/builder/.ccache \
    CCACHE_MAXSIZE=1G
RUN mkdir -p /home/builder/.ccache

# ── 6. Clear proxy so runtime does not fail when proxy is gone ────
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    NO_PROXY="" \
    no_proxy=""

# ── 7. Workspace ──────────────────────────────────────────────────
WORKDIR /workspace
