# Third-Party Licenses

Asymptote incorporates the following third-party components:

| Component | License | Source |
|-----------|---------|--------|
| span.hpp | Boost 1.0 | https://github.com/martinmoene/span-lite |
| wyhash | The Unlicense (Public Domain) | https://github.com/wangyi-fudan/wyhash |
| Boehm GC | Custom permissive | https://www.hboehm.info/gc/ |
| LspCpp | MIT | https://github.com/kuafuwang/LspCpp |
| libatomic_ops | MIT (core) / GPL-2.0 (extensions) | https://github.com/ivmai/libatomic_ops |
| GLEW | BSD 3-Clause | https://glew.sourceforge.net/ |
| TinyEXR | BSD 3-Clause | https://github.com/syoyo/tinyexr |

Full copyright notices and license texts for all components (including
Asymptote's own LGPL) are available via `asy --licenses=full`, which reads
from the `licenses/` directory installed alongside the binary.

## Optionally bundled on macOS

The macOS bundling build (`--enable-macos-bundling`, which sets
`BUNDLE_VULKAN=yes`) copies the Vulkan / GLFW runtime dynamic libraries into
`./lib/*.dylib` so the binary is self-contained. Those builds also ship the
following additional components and their license texts. **These rows apply
only to such builds** — a stock build that bundles no dylibs neither ships
them nor lists them.

| Component | License | Source |
|-----------|---------|--------|
| GLFW | Zlib | https://github.com/glfw/glfw |
| Vulkan-Loader | Apache-2.0 | https://github.com/KhronosGroup/Vulkan-Loader |
| glslang (incl. its `libSPIRV` backend) | BSD-3-Clause AND BSD-2-Clause AND MIT AND Apache-2.0 AND GPL-3.0-or-later WITH Bison-exception-2.2 | https://github.com/KhronosGroup/glslang |
| MoltenVK | Apache-2.0 | https://github.com/KhronosGroup/MoltenVK |

The single source of truth for this set is
[build-scripts/bundled-dylib-licenses.tsv](build-scripts/bundled-dylib-licenses.tsv);
the vendored license texts live in
[licenses/bundled/](licenses/bundled/). Which extras a given binary actually
ships is reported by `asy --licenses=full`.

See [MAINTAINER-LICENSE-GUIDE.md](MAINTAINER-LICENSE-GUIDE.md) for
distribution and maintenance procedures.
