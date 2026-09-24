#!/usr/bin/env bash
#
# Bring the container's vcpkg checkout in line with the `builtin-baseline`
# pinned in vcpkg.json.
#
# Why this is needed
# ------------------
# The base image ships a *shallow* vcpkg clone whose tip is whatever was current
# when the image was built. vcpkg then resolves dependencies from two different
# places:
#
#   * `versions/baseline.json` -- read out of the pinned baseline *commit*
#     (which version of each port to use);
#   * `versions/<x>-/<port>.json` -- read from the *checked-out working tree*
#     (which git-tree each version maps to).
#
# So merely fetching the baseline commit is not enough. It satisfies the first
# read and fails the second with
#
#   error: no version database entry for curl at 8.17.0.
#
# whenever the pin is *newer* than the image's checkout. (A pin older than the
# checkout happens to work without any of this, because the version database is
# append-only -- which is why this only started biting when the baseline was
# bumped forward past the image's vcpkg.) The checkout itself has to move, and
# the vcpkg tool has to be re-bootstrapped afterwards because the pinned tool
# release in `scripts/vcpkg-tool-metadata.txt` moves with it.
#
# HEAD is synced to exactly the pinned baseline rather than to some newer tip.
# That does not change which port versions get built -- those come from
# baseline.json at the pinned commit either way -- it just makes the tool and
# the version database deterministic for a given vcpkg.json.
#
# This runs on container start as well as on create (see devcontainer.json), so
# that pulling a commit which bumps the baseline does not strand an existing
# container. It is a cheap no-op once the checkout is in sync.
#
# Failures here are reported but non-fatal: they must not block container start,
# and the no-vcpkg `linux/sandbox` preset does not need any of this.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VCPKG_DIR="${VCPKG_ROOT:-/usr/local/vcpkg}"

give_up() {
    echo "warning: could not sync '$VCPKG_DIR' to the vcpkg baseline pinned in" >&2
    echo "         vcpkg.json ($BASELINE): $1" >&2
    echo "         Presets that use vcpkg (e.g. linux/release) will fail to" >&2
    echo "         configure until this is resolved. To do it by hand:" >&2
    echo "           git -C $VCPKG_DIR fetch --depth 1 origin $BASELINE" >&2
    echo "           git -C $VCPKG_DIR checkout --detach $BASELINE" >&2
    echo "           $VCPKG_DIR/bootstrap-vcpkg.sh" >&2
    exit 0
}

BASELINE="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("builtin-baseline", ""))' \
    "$REPO_ROOT/vcpkg.json")"

# No pin (or no manifest) -> nothing to enforce.
if [ -z "$BASELINE" ]; then
    exit 0
fi

if [ ! -d "$VCPKG_DIR/.git" ]; then
    give_up "'$VCPKG_DIR' is not a git checkout"
fi

# Fast path: already in sync, and the tool has been bootstrapped. This is the
# usual case on container start, and costs one rev-parse.
if [ "$(git -C "$VCPKG_DIR" rev-parse HEAD 2>/dev/null || true)" = "$BASELINE" ] \
    && [ -x "$VCPKG_DIR/vcpkg" ]; then
    exit 0
fi

if [ ! -w "$VCPKG_DIR/.git" ]; then
    give_up "no write permission (it should be group-writable by 'vcpkg')"
fi

# The clone is shallow, so the pinned commit usually has to be fetched. A
# depth-1 fetch brings the whole tree at that commit, which is all vcpkg needs.
if ! git -C "$VCPKG_DIR" cat-file -e "${BASELINE}^{commit}" 2>/dev/null; then
    echo "Fetching pinned vcpkg baseline $BASELINE into $VCPKG_DIR"
    if ! git -C "$VCPKG_DIR" fetch --depth 1 origin "$BASELINE"; then
        give_up "fetch failed (offline?)"
    fi
fi

echo "Checking out vcpkg baseline $BASELINE in $VCPKG_DIR"
if ! git -C "$VCPKG_DIR" checkout --detach "$BASELINE"; then
    give_up "checkout failed (local modifications in the vcpkg tree?)"
fi

# The tool release is pinned per-commit, so re-bootstrap after moving HEAD.
echo "Bootstrapping vcpkg tool for baseline $BASELINE"
if ! "$VCPKG_DIR/bootstrap-vcpkg.sh"; then
    give_up "bootstrap-vcpkg.sh failed"
fi
