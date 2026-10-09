#!/usr/bin/env bash
#
# Dev-container provisioning that needs the mounted workspace (and therefore
# cannot live in the Dockerfile). Runs once, as the non-root `vscode` user,
# from the workspace root after the container is created.
set -euo pipefail

# ---------------------------------------------------------------------------
# Cache volumes: make the vcpkg / ccache mount points writable by `vscode`.
# (These are named Docker volumes mounted in devcontainer.json.)
# ---------------------------------------------------------------------------
sudo chown -R vscode:vscode /home/vscode/.cache/vcpkg /home/vscode/.ccache

# ---------------------------------------------------------------------------
# Sync the vcpkg checkout to the builtin-baseline pinned in vcpkg.json.
#
# Lives in its own script because devcontainer.json also runs it on every
# container *start*: the baseline can be bumped by a commit pulled long after
# this container was created, and a stale checkout breaks configure. See
# sync-vcpkg-baseline.sh for what vcpkg actually requires.
# ---------------------------------------------------------------------------
bash "$(dirname "$0")/sync-vcpkg-baseline.sh"

# ---------------------------------------------------------------------------
# Runtime dir for the (absent) Wayland session.
#
# devcontainer.json sets XDG_RUNTIME_DIR=/tmp/runtime-vscode to silence the
# spurious "XDG_RUNTIME_DIR not set in the environment." that Mesa's Vulkan WSI
# triggers via libwayland on every render. Create the directory it points at
# with the mode 0700 the XDG spec mandates so it is a valid runtime dir for
# anything that actually uses it. Skip if XDG_RUNTIME_DIR is unset (nothing to
# create).
# ---------------------------------------------------------------------------
if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
fi

# ---------------------------------------------------------------------------
# Python developer virtualenv.
#
# Deliberately created OUTSIDE the bind-mounted workspace: the repo's own
# `./venv` is gitignored and is created on the *host* (potentially a different
# Python version / OS), so reusing or overwriting it from inside the container
# would break one side or the other. This container-owned venv is what VS Code
# is pointed at via `python.defaultInterpreterPath` in devcontainer.json.
# ---------------------------------------------------------------------------
VENV_DIR="$HOME/.venv/asymptote"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements-dev.txt

# Auto-activate the venv in interactive bash shells (terminals opened in the
# container). This is the sole activation path: the VS Code Python extension's
# own terminal activation (python.terminal.activateEnvironment) is disabled in
# devcontainer.json because it injects asynchronously and sends a Ctrl-C first,
# which can interrupt a long-running command in the terminal.
ACTIVATE_LINE="source \"$VENV_DIR/bin/activate\""
if ! grep -qsF "$ACTIVATE_LINE" "$HOME/.bashrc"; then
    {
        printf '\n# Activate the Asymptote developer virtualenv\n'
        printf '%s\n' "$ACTIVATE_LINE"
    } >> "$HOME/.bashrc"
fi

# ---------------------------------------------------------------------------
# Per-developer provisioning hooks.
#
# Anything in .devcontainer/local/ is personal, optional setup that is not part
# of the shared container definition. If the directory exists, run every
# executable *.sh in it (sorted) after the standard provisioning above. A failing
# hook is reported but does not abort the rest of provisioning. See, e.g.,
# .devcontainer/local/relocate-build-dir.sh, which moves the CMake build tree out
# of the bind-mounted workspace.
# ---------------------------------------------------------------------------
LOCAL_HOOK_DIR="$(dirname "$0")/local"
if [ -d "$LOCAL_HOOK_DIR" ]; then
    for hook in "$LOCAL_HOOK_DIR"/*.sh; do
        [ -e "$hook" ] || continue   # glob did not match -> nothing to run
        echo "Running local provisioning hook: $hook"
        bash "$hook" || echo "warning: local hook '$hook' exited non-zero; continuing"
    done
fi
