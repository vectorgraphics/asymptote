# Personal dev-container hooks (`local/`) — example

> **This is a committed template.** Copy it to `README.md` and add your own
> `*.sh` hooks in this directory. Everything in `local/` is gitignored *except*
> this example pair (`README.example.md`, `example-hook.sh.example`), so your
> personal files — including your own `README.md` — never get committed.

This directory holds per-developer, optional setup that is intentionally *not*
part of the shared container definition. After the standard provisioning, the
shared [`../post-create.sh`](../post-create.sh) runs every `*.sh` file in here,
in sorted order. The executable bit is **not** required — each hook is invoked
via `bash`, so `chmod +x` is unnecessary. A hook that exits non-zero is reported
but does not abort the rest of provisioning.

## Getting started

1. Copy [`example-hook.sh.example`](./example-hook.sh.example) to a name ending
   in `.sh` (e.g. `my-setup.sh`).
2. Edit it to do whatever personal provisioning you want.
3. Rebuild / reopen the container, or run `bash .devcontainer/post-create.sh`.

The `.example` suffix keeps both template files inert: the `*.sh` glob in
post-create.sh does not match them, so they never run.

## Ideas for hooks

- Relocate the CMake build tree out of the bind mount via a generated
  `CMakeUserPresets.json` — a ready-made hook ships as
  [`relocate-build-dir.sh.example`](./relocate-build-dir.sh.example); copy it to
  `relocate-build-dir.sh` to enable it (see the shared
  [`../README.md`](../README.md)).
- Append `export MAKEFLAGS="-j$(nproc)"` to `~/.bashrc`.
- Install extra pip packages into the container-owned `~/.venv/asymptote`.
- Install a coding-agent CLI and persist its config in a named volume.
