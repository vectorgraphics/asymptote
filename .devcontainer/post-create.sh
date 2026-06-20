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
# Ensure the vcpkg builtin-baseline pinned in vcpkg.json is present locally.
#
# The base image ships a shallow vcpkg clone whose tip can predate the baseline
# commit pinned in vcpkg.json. Contrary to the assumption that a `safe.directory`
# config lets vcpkg fetch it on demand, vcpkg does NOT reliably fetch a missing
# baseline -- it aborts configure with "failed to `git show`
# versions/baseline.json ... This may be fixed by fetching commits with
# `git fetch`". Fetch the pinned commit explicitly so `cmake --preset` works on
# a fresh container regardless of how old the image's vcpkg clone is. Reading the
# baseline from vcpkg.json keeps this correct when the pin is later bumped.
# ---------------------------------------------------------------------------
VCPKG_DIR="${VCPKG_ROOT:-/usr/local/vcpkg}"
VCPKG_BASELINE="$(python3 -c 'import json; print(json.load(open("vcpkg.json")).get("builtin-baseline", ""))')"
if [ -n "$VCPKG_BASELINE" ] \
    && ! git -C "$VCPKG_DIR" cat-file -e "${VCPKG_BASELINE}^{commit}" 2>/dev/null; then
    echo "Fetching pinned vcpkg baseline $VCPKG_BASELINE into $VCPKG_DIR"
    git -C "$VCPKG_DIR" fetch --depth 1 origin "$VCPKG_BASELINE"
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
# container). VS Code also activates it automatically for tasks/debugging via
# python.terminal.activateEnvironment.
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
