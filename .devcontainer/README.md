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
- **Documentation toolchain:** LaTeX + Ghostscript + texinfo are baked into the
  image (on by default) so the `docgen` target works out of the box.
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
