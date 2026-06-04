# Vendored licenses for optionally-bundled macOS dylibs

The texts in this directory are reproduced **verbatim** from upstream. They are
the redistribution-required license texts for the dynamic libraries that the
macOS bundling build (`--enable-macos-bundling`, `BUNDLE_VULKAN=yes`) copies
into `./lib/*.dylib`. They are vendored here — rather than copied from a build
host's SDK — because the LunarG Vulkan SDK ships no usable per-component license
texts (only a pointer file). See
[../../local/dylib-licenses-plan.md](../../local/dylib-licenses-plan.md) and
[../../MAINTAINER-LICENSE-GUIDE.md](../../MAINTAINER-LICENSE-GUIDE.md).

These are consumed by the registry
[build-scripts/bundled-dylib-licenses.tsv](../../build-scripts/bundled-dylib-licenses.tsv),
which drives both the build-time staging into `doc/licenses/` and the runtime
`asy --licenses[=full]` output.

## Provenance

Component versions match the **LunarG Vulkan SDK 1.4.350.0** (commit hashes from
the SDK's `VERSIONS.txt`). Do not edit the license bodies; to update, re-fetch at
the new SDK's component commits and update the table below.

| File | Upstream | Commit (pinned) | License | Copyright holders |
|------|----------|-----------------|---------|-------------------|
| `glfw-LICENSE.md` | [glfw/glfw](https://github.com/glfw/glfw) | (from `~/glfw` source tree) | Zlib | Marcus Geelnard (2002–2006); Camilla Löwy (2006–2019) |
| `vulkan-LICENSE.txt` | [KhronosGroup/Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader) | `a9e72c66d5cb79911eb9a9063bf4016dd0a3a123` | Apache-2.0 | The Khronos Group Inc. and Vulkan-Loader contributors |
| `glslang-LICENSE.txt` | [KhronosGroup/glslang](https://github.com/KhronosGroup/glslang) | `275822a6261ee689aadb1da5f09a0ec2f058685c` | BSD-3-Clause AND BSD-2-Clause AND MIT AND Apache-2.0 AND GPL-3.0-or-later WITH Bison-exception-2.2 (composite; self-contained) | 3Dlabs Inc. Ltd., Google Inc., LunarG, and glslang contributors |
| `moltenvk-LICENSE.txt` | [KhronosGroup/MoltenVK](https://github.com/KhronosGroup/MoltenVK) | `db445ff2042d9ce348c439ad8451112f354b8d2a` | Apache-2.0 | The Brenwill Workshop Ltd. and The Khronos Group Inc. |

## Notes

- **`libSPIRV.*.dylib` belongs to glslang**, not KhronosGroup/SPIRV-Tools or
  SPIRV-Cross. It is glslang's SPIR-V backend (its SONAME version tracks
  `libglslang`) and is covered by `glslang-LICENSE.txt`. SPIRV-Tools and
  SPIRV-Cross are **not** bundled by this build (they are static archives in the
  SDK and are not referenced by `asy`); no license text for them is vendored.
- **No NOTICE files.** Verified via the GitHub contents API that neither
  Vulkan-Loader (`a9e72c66`) nor MoltenVK (`db445ff2`) ships a `NOTICE`/`NOTICE.txt`
  at the pinned commit. Apache-2.0 §4(d) only obliges reproducing a NOTICE when
  upstream provides one, so shipping the `LICENSE` text alone is sufficient. No
  empty placeholder NOTICE files are committed.
- The glslang composite `LICENSE.txt` is self-contained; its GPL-3.0 section
  applies only to generated parser files and carries the Bison exception, so it
  imposes no copyleft on the combined work.
