# Dev container for Asymptote

This directory provides a [Dev Container](https://containers.dev/) for building
Asymptote with the CMake + vcpkg toolchain. It mirrors the Linux CI environment
(Ubuntu 22.04) so a local build matches what GitHub Actions does.

## What's inside

- **Base image:** `mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04`, which
  ships GCC, git, and **vcpkg** pre-installed at `/usr/local/vcpkg`. Its stock
  CMake (3.22) is too old for our presets, so the `Dockerfile` installs a
  current CMake (>= 3.27) system-wide via pip.
- **System deps** (`Dockerfile`): `ninja-build`, `flex`, `bison`, `perl`,
  `python3`, `pkg-config`, `ccache`, the X11/GL/Wayland dev libraries that the
  `glfw3`/`vulkan` vcpkg ports need, and the autotools/libtool stack
  (`autoconf`, `automake`, `autoconf-archive`, `libtool`, `libtool-bin`,
  `libltdl-dev`) that the `libxcrypt` vcpkg port uses to regenerate its build
  system. The `glm`, GSL, curl, readline and curses dev packages
  (`libglm-dev`, `libgsl-dev`, `libcurl4-openssl-dev`, `libreadline-dev`,
  `libncurses-dev`) are also installed so the no-vcpkg `linux/sandbox` preset
  (see [Sandbox build](#sandbox-build-no-network)) can build from system
  libraries alone.
- **vcpkg baseline access** (`Dockerfile`): `/usr/local/vcpkg` is registered as
  a git `safe.directory` so vcpkg can fetch the `builtin-baseline` pinned in
  [`vcpkg.json`](../vcpkg.json) (it is owned by root but the build runs as
  `vscode`; without this, baseline resolution fails with a `git show
  versions/baseline.json` error).
- **CPU rendering** (`Dockerfile` + `devcontainer.json`): Mesa's software
  rasterizers are installed and forced on so the Vulkan/OpenGL renderers can
  produce rasterized images without a GPU — `mesa-vulkan-drivers` (lavapipe) for
  Vulkan, `libgl1-mesa-dri` (llvmpipe) for OpenGL, plus `xvfb` for the X display
  the GL path needs. See [CPU rendering](#cpu-rendering-llvmpipe--lavapipe).
- **Documentation toolchain:** LaTeX + Ghostscript + texinfo are baked into the
  image (on by default) so the `docgen` target works out of the box. Ubuntu
  22.04's stock Ghostscript (9.55) is too old for asy's default `png16malpha`
  PNG device, so the `Dockerfile` also builds a current Ghostscript from source
  and installs it ahead of the apt one on `PATH` (see [CPU
  rendering](#cpu-rendering-llvmpipe--lavapipe)).
- **vcpkg dependencies** are declared in the top-level [`vcpkg.json`](../vcpkg.json)
  and are resolved automatically by CMake during configure.
- **Python developer virtualenv** (`post-create.sh`): a venv is created at
  `~/.venv/asymptote` with the packages from [`requirements-dev.txt`](../requirements-dev.txt)
  (linting tools — black, isort, pylint — plus `jinja2`, which the `libsystemd`
  vcpkg port's meson build needs at configure time). VS Code is pointed at it via
  `python.defaultInterpreterPath`, and it is auto-activated in integrated
  terminals. It lives *outside* the workspace on purpose — see
  [Python virtualenv](#python-virtualenv) below.

## Getting started

Open the repository in VS Code and choose **"Reopen in Container"** (or use the
Dev Containers CLI). Once the container is up:

```bash
# Configure (first run downloads + builds vcpkg deps — this can take a while)
cmake --preset linux/release

# Build the asy binary plus base files
cmake --build --preset linux/release --target asy-with-basefiles

# Build everything (including tests)
cmake --build --preset linux/release
```

The resulting binary lands in `cmake-build-linux/release/asy`.

> **Tip:** to keep build artifacts *out* of the bind-mounted workspace, use the
> `linux/release/devcontainer` preset instead (see
> [Out-of-workspace build directory](#out-of-workspace-build-directory) below).
> Its binary lands in `~/.local/asy-build/release/asy`.

### Running tests

```bash
ctest --test-dir cmake-build-linux/release/
```

### Documentation builds

The LaTeX + Ghostscript toolchain needed by the `docgen` target is installed by
default and cached in the local image, so building the docs works immediately:

```bash
cmake --build --preset linux/release --target docgen
```

If you want a leaner image without these (they add several GB), set the
`INSTALL_DOC_DEPS` build arg to `"false"` in
[`devcontainer.json`](./devcontainer.json) and rebuild the container.

## CPU rendering (llvmpipe / lavapipe)

The container has no GPU, but it can still exercise the Vulkan and OpenGL
renderers via Mesa's software rasterizers and write the result to a rasterized
file (PNG, etc.). This is for *testing* the renderers — producing and inspecting
images — not for interactive viewing.

What's set up for you (see [`Dockerfile`](./Dockerfile) and
[`devcontainer.json`](./devcontainer.json)):

- **lavapipe** (`mesa-vulkan-drivers`) — Mesa's software Vulkan driver.
  `VK_ICD_FILENAMES` is pinned to its ICD
  (`/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`) so device selection is
  deterministic.
- **llvmpipe** (`libgl1-mesa-dri`) — Mesa's software OpenGL rasterizer, forced
  on with `LIBGL_ALWAYS_SOFTWARE=true` and `GALLIUM_DRIVER=llvmpipe`.
- **xvfb** — a virtual X server, because asy's OpenGL renderer always opens a
  (hidden) GLFW window and so needs a display. Vulkan export renders to
  offscreen buffers and needs no display.

These renderers are only present in builds configured with them. The full
`linux/release` vcpkg build builds the **Vulkan** renderer (`libasyvulkan.so`,
dlopened at runtime); the OpenGL renderer (`libasyopengl.so`) is built only by
the autotools (`./configure && make`) build. The no-network `linux/sandbox`
preset builds neither.

```bash
# Vulkan -> PNG (no display needed): renders via lavapipe.
# Run from any directory OTHER than the workspace root (see the note below),
# and point -dir at the build's base/ so the dlopened libasyvulkan.so is found.
cd /tmp
~/.local/asy-build/release/asy -dir ~/.local/asy-build/release/base \
    -f png -render=4 -o teapot.png \
    /workspaces/asymptote/ex/teapot

# OpenGL -> PNG: wrap in xvfb-run so the GL renderer has an X display.
# (Requires an autotools build that produced libasyopengl.so.)
xvfb-run -a ./asy -dir ./base -novulkan -f png -render=4 -o out.png myscene.asy

# Diagnostics:
vulkaninfo 2>/dev/null | grep -m1 deviceName     # -> llvmpipe (LLVM ...)
xvfb-run -a glxinfo | grep -E 'OpenGL renderer'  # -> llvmpipe
```

Verified working: the teapot above renders on `Device 0: llvmpipe` and writes a
valid RGBA PNG.

**Harmless `XDG_RUNTIME_DIR` warning.** On a Vulkan render you may see, once or
twice on stderr:

    error: XDG_RUNTIME_DIR not set in the environment.

Despite the `error:` prefix this is **not** an asy error and **not** fatal — the
render still succeeds and the PNG is written. The message comes from
`libwayland`: Mesa's Vulkan WSI probes for a Wayland session during instance
creation by calling `wl_display_connect()`, and libwayland prints this when
`XDG_RUNTIME_DIR` is unset (any Vulkan program does it here — e.g. `vulkaninfo`
prints the same). This container therefore sets `XDG_RUNTIME_DIR` (in
[`devcontainer.json`](./devcontainer.json), with the directory created by
[`post-create.sh`](./post-create.sh)), which silences it; the line above only
appears if you unset that variable or run in an environment that lacks it (in
which case `export XDG_RUNTIME_DIR=/tmp/runtime-vscode` makes it go away). It is
safe to ignore.

**Gotcha — Vulkan render hangs forever (the `device_select` layer).** Mesa ships
an implicit Vulkan layer, `VK_LAYER_MESA_device_select`, that during
`vkEnumeratePhysicalDevices()` connects to the X server named by `DISPLAY` (VS
Code sets `DISPLAY=:0`) to learn which GPU drives the display so it can rank
devices. This container has no working X server — only a stale
`/tmp/.X11-unix/X0` socket with nothing behind it — so the layer's
`xcb_wait_for_reply()` blocks in `poll()` and the render hangs indefinitely
(intermittently, depending on the socket's state). A backtrace of the stuck
process shows `xcb_wait_for_reply` → `libVkLayer_MESA_device_select.so` →
`vkEnumeratePhysicalDevices` → `AsyVkRender::pickPhysicalDevice`.
[`devcontainer.json`](./devcontainer.json) disables the layer with
`VK_LOADER_LAYERS_DISABLE=VK_LAYER_MESA_device_select` (it has nothing to select
anyway — `VK_ICD_FILENAMES` pins the single lavapipe device). If you hit this in
an environment without that variable, `export
VK_LOADER_LAYERS_DISABLE=VK_LAYER_MESA_device_select` (or `unset DISPLAY`) before
rendering.

**Gotcha — stale workspace renderer libs / current directory.** asy locates the
renderer library through its search path, which is tried in order: the *current
directory* first, then `-dir`, then the system dir. An autotools `make` in the
workspace leaves `libasyvulkan.so` / `libasyopengl.so` in the repo root; built
on the host they are usually incompatible with the container
(`GLIBCXX_… not found`). Running the container's `asy` *from the workspace root*
picks up those stale copies first and fails to render. Avoid this by running
from another directory (as above), or remove the host-built
`/workspaces/asymptote/libasy*.so`. The relocatable build installs its own
`libasyvulkan.so` into `~/.local/asy-build/release/base/`, so `-dir <that base>`
finds the correct one from any non-workspace directory.

**PNG driver.** asy defaults to the Ghostscript `png16malpha` device, which
Ubuntu 22.04's stock Ghostscript (9.55) does not provide (`Unknown device:
png16malpha` → `shipout failed`). The `Dockerfile` builds a current Ghostscript
from source and puts it ahead of the apt one on `PATH`, so the default driver
works and no `-pngdriver` override is needed. (This newer `gs` is part of the
doc toolchain, so it is present only when `INSTALL_DOC_DEPS=true`, the default;
without it the image has no `gs` at all and PNG export is unavailable. If you
ever do hit a `png16malpha` error, fall back with `-pngdriver pngalpha` or
`settings.pngdriver="pngalpha";`.)

**Note (Vulkan version).** asy requests Vulkan 1.4
(`apiVersion=VK_API_VERSION_1_4` in `vkrender.cc`, VMA configured for 1.4) while
Ubuntu 22.04's lavapipe advertises 1.3. In practice this works — the loader
tolerates an application requesting a higher API version than the device — and
rendering succeeds on lavapipe. If a future change makes asy hard-require 1.4
device features, install modern Mesa from the `kisak-mesa` PPA (the Linux
analogue of building Mesa from source as `doc/lavapipe.txt` does on macOS).

## Using this without VS Code

`devcontainer.json` is an open spec ([containers.dev](https://containers.dev/)),
not a VS Code feature, so the same config works from other tooling:

- **The `devcontainer` CLI** — the editor-agnostic reference implementation.
  Build/start the container and get a shell in it with nothing but Docker (or
  Podman) and Node:

  ```bash
  npm install -g @devcontainers/cli      # one-time
  devcontainer up --workspace-folder .   # build + start
  devcontainer exec --workspace-folder . bash   # shell inside; run cmake/ctest here
  ```

- **JetBrains IDEs** (IntelliJ family, **CLion** — relevant for this C++
  project) read `devcontainer.json` natively. The `customizations.vscode` block
  is simply ignored by them, not an error.

- **Visual Studio 2022** (17.4+) supports dev containers for C++ projects that
  use CMake Presets — which this repo does. Enable the *Linux and embedded
  development with C++* workload, install Docker Desktop, and VS drives the
  container as a remote build target (it likewise ignores the VS-Code-specific
  `customizations`). See Microsoft's
  [Dev Containers for C++ in Visual Studio](https://devblogs.microsoft.com/cppblog/dev-containers-for-c-in-visual-studio/).

- **GitHub Codespaces** uses this `devcontainer.json` directly as the cloud-hosted
  equivalent.

Coverage varies by tool — VS Code and the CLI are the most complete — but the
core fields this config relies on (`build`/`Dockerfile`, `runArgs`,
`containerEnv`, `mounts`, `postCreateCommand`) are honored broadly.

## Using this from Emacs

Emacs has no native Dev Container integration (no "reopen in container"), but it
can drive the same container two ways. Either way, start it with the
[`devcontainer` CLI](#using-this-without-vs-code) rather than VS Code
(`devcontainer up --workspace-folder .`).

**1. Edit from host Emacs over TRAMP (recommended).** Emacs 29+ ships
`tramp-container.el` with built-in `docker` and `podman` methods (older Emacs:
the `docker-tramp` package). There is *no* built-in `devcontainer` method, so
find the running container's name and use the `docker`/`podman` method directly:

```bash
docker ps   # note the container NAME the `devcontainer up` step created
```

```
M-x find-file RET /docker:<NAME>:/workspaces/asymptote/README.md
```

Use `/podman:<NAME>:…` under rootless Podman. Emacs runs on the host while
Eglot/LSP, `M-x compile`, and shells (`M-x shell`) all execute *inside* the
container over TRAMP, so the build matches everyone else's. Community packages
[`devcontainer.el`](https://github.com/lina-bh/devcontainer.el) and
[`emacs-dev-containers`](https://github.com/alexispurslane/emacs-dev-containers)
automate the `devcontainer` CLI lifecycle and the TRAMP hand-off if you prefer
not to do it by hand.

**2. Run Emacs inside the container.** `devcontainer exec --workspace-folder .
bash`, then `emacs -nw` (or a GUI Emacs over X/Wayland forwarding). The shared
image does **not** install Emacs; add it for yourself with a personal
[`local/` hook](#per-developer-provisioning-hooks) (copy
`local/example-hook.sh.example` to `local/emacs.sh` and `apt`-install it there)
so it never lands in the image other developers and CI build.

## Python virtualenv

The repo's own `./venv` is gitignored and is typically created on the **host**,
so it carries the host's Python version and platform-specific binaries. Because
the workspace is bind-mounted into the container, reusing that venv inside the
container (Ubuntu 22.04 / Python 3.10) would fail, and recreating it in place
would break it for host use.

To avoid that, `post-create.sh` builds a separate, container-owned venv at
`~/.venv/asymptote` and `requirements-dev.txt` is installed into it. VS Code
selects it automatically (`python.defaultInterpreterPath`), and a `source
.../activate` line is appended to `~/.bashrc` so new terminals pick it up too.

If you need the GUI dependencies as well, install them into the same venv:

```bash
pip install -r GUI/requirements.txt
```

## Caching

vcpkg's binary cache (`~/.cache/vcpkg`) and `ccache` (`~/.ccache`) are stored in
named Docker volumes, so dependency builds are not repeated on every container
rebuild. See the `mounts` entry in `devcontainer.json`.

## Nested sandboxes (bubblewrap) and `/proc` masking

Some tools run their own [bubblewrap](https://github.com/containers/bubblewrap)
(`bwrap`) sandbox *inside* this container — Claude Code is one. A bwrap sandbox
creates a new PID namespace and mounts a fresh procfs into it, and the kernel
only permits that nested `mount -t proc` when the container's existing `/proc`
is **fully visible** (no masked sub-paths).

Rootless **Podman** masks paths under `/proc` (`/proc/sys`, `/proc/kcore`,
`/proc/irq`, …) with read-only / tmpfs overmounts, so the nested mount fails
with:

    Can't mount proc on /newroot/proc: Operation not permitted

The fix is to unmask `/proc` for the container (no full `--privileged` needed).
Because that slightly widens what the container can see, it is **off by
default** and opt-in via a **host** environment variable, read when the
container is created — set it in the environment that launches VS Code or the
Dev Containers CLI, then rebuild/reopen the container:

| Runtime | Set on the host | Effect |
|---------|-----------------|--------|
| **Podman** (rootless) | `ASY_DEVCONTAINER_SECURITY_OPT=unmask=/proc/*` | drops only the `/proc` masks |
| **Docker** | `ASY_DEVCONTAINER_SECURITY_OPT=systempaths=unconfined` | unmasks all of `/proc` and `/sys` (broader; Docker has no `unmask=`) |
| _unset (default)_ | — | expands to `no-new-privileges=false`, a no-op on both runtimes |

If you do not run a nested sandbox inside the container, leave it unset. The
variable feeds the `runArgs` `${localEnv:…}` substitution in
[`devcontainer.json`](./devcontainer.json).

> **Where to set it:** `${localEnv:…}` reads the environment of whatever
> *launches* VS Code or the Dev Containers CLI, which is not always your shell.
>
> - **Launched from a terminal** (`code .`): export it from your shell rc file
>   (`~/.bashrc`/`~/.zshrc`/`~/.profile`). `export` is required so child
>   processes inherit it, and single-quote the value:
>
>   ```sh
>   export ASY_DEVCONTAINER_SECURITY_OPT='unmask=/proc/*'
>   ```
>
> - **Launched from a desktop launcher / dock icon**: that process does **not**
>   inherit your shell rc files, so set it in your login (graphical-session)
>   environment instead. On Linux, add a line to `~/.config/environment.d/*.conf`
>   and log out and back in. This is systemd, not a shell — use **no** `export`
>   and **no** quotes:
>
>   ```ini
>   ASY_DEVCONTAINER_SECURITY_OPT=unmask=/proc/*
>   ```
>
> The value must reach the runtime as the literal string `unmask=/proc/*` (the
> `runArgs` are passed straight to podman/docker as argv, with no shell in
> between), which both forms above achieve. It is read at container-create time,
> so rebuild/reopen the container after changing it.

## Per-developer provisioning hooks

After the standard provisioning, `post-create.sh` runs every `*.sh` file in
`.devcontainer/local/` (if that directory exists), in sorted order. The
executable bit is not required — each hook is invoked via `bash`, so `chmod +x`
is unnecessary. This is where personal, optional setup lives — it is
intentionally separate from the shared container definition, so it never affects
other developers or CI. A hook that exits non-zero is reported but does not abort
the rest of provisioning.

Everything in `local/` is gitignored **except** a committed template pair —
[`local/README.example.md`](./local/README.example.md) and
`local/example-hook.sh.example` — which documents the hook contract and gives a
starting point to copy. Your own `README.md` and `*.sh` hooks there stay private.

## Out-of-workspace build directory

The workspace is bind-mounted from the host, so anything CMake writes under
`cmake-build-linux/` is visible on the host too — which makes it easy to
accidentally launch a container-built `asy` (linked against the container's
libraries) outside the container.

A committed example hook,
[`local/relocate-build-dir.sh.example`](./local/relocate-build-dir.sh.example),
avoids that. Copy it to `local/relocate-build-dir.sh` to enable it (it then runs
on every container (re)provision; you can also run it once by hand with `bash
.devcontainer/local/relocate-build-dir.sh`). It generates a (gitignored)
`CMakeUserPresets.json` at the repo root with two relocated presets whose
`binaryDir` is on the container's own filesystem, *outside* the bind mount, so
the host never sees those binaries. Both inherit their shipped counterpart and
additionally enable `ccache` (whose cache is a persistent volume), so although
the build trees themselves are not persisted across container rebuilds,
recompiles stay cheap:

- **`linux/release/devcontainer`** — the full vcpkg build, in
  `~/.local/asy-build/release`.
- **`linux/sandbox/devcontainer`** — the no-vcpkg sandbox build, in
  `~/.local/asy-build/sandbox` (see [Sandbox build](#sandbox-build-no-network)).

```bash
cmake --preset linux/release/devcontainer
cmake --build --preset linux/release/devcontainer --target asy-with-basefiles
ctest --test-dir ~/.local/asy-build/release/ -R "asy.types.*"
```

The hook will not overwrite a `CMakeUserPresets.json` you wrote yourself (see
below); in that case it prints the `binaryDir` to add manually and exits.

## Sandbox build (no network)

The production presets need vcpkg, which downloads dependencies and so cannot
run in a no-network sandbox (e.g. a coding agent's bash sandbox). The
`linux/sandbox` preset instead builds from the **system** libraries installed by
the `Dockerfile` and disables the features whose deps only come through vcpkg
(`ENABLE_VULKAN`, `ENABLE_FFTW3`, `ENABLE_EIGEN3`, `ENABLE_LSP`,
`ENABLE_ASY_CXXTEST`, `ENABLE_DOCGEN`, `ENABLE_MISCFILES_GEN`). Coverage is
partial, but it exercises the bulk of `cmake-scripts/` end-to-end — enough to
catch most build-script regressions without leaving the sandbox.

The relocated `linux/sandbox/devcontainer` variant (from the hook above) keeps
its build tree and generated base files out of the bind mount:

```bash
cmake --preset linux/sandbox/devcontainer
cmake --build --preset linux/sandbox/devcontainer --target asy-with-basefiles
ctest --test-dir ~/.local/asy-build/sandbox/
~/.local/asy-build/sandbox/asy -dir ~/.local/asy-build/sandbox/base -c "write('hi');"
```

Because Ubuntu 22.04's `libglm-dev` ships a `glmConfig.cmake` with a
pre-3.5 `cmake_minimum_required` that the image's modern CMake refuses to load,
the relocated preset sets `CMAKE_POLICY_VERSION_MINIMUM=3.5`.

## Custom debug presets

The shipped presets only cover release builds. For a debug configuration, add a
`CMakeUserPresets.json` at the repo root inheriting from
`base/buildBaseWithVcpkg` + `base/debug`. See [`../INSTALL-VCPKG.md`](../INSTALL-VCPKG.md)
for an example.
