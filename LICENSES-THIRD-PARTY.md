# Third-Party Licenses

Asymptote incorporates the following third-party components:

| Component | License | Location |
|-----------|---------|----------|
| span.hpp | Boost 1.0 | [span.hpp](span.hpp) |
| wyhash | The Unlicense (Public Domain) | [wyhash.h](wyhash.h) |
| Boehm GC | Custom permissive | [gc/](gc/) |
| LspCpp | MIT | [LspCpp/](LspCpp/) |
| libatomic_ops | MIT / GPL-2.0 | [libatomic_ops/](libatomic_ops/) |
| GLEW | BSD 3-Clause | [backports/glew/](backports/glew/) |
| TinyEXR | BSD 3-Clause | [tinyexr/](tinyexr/) |

## Component Details

### span.hpp (Boost 1.0)
- Header-only C++ library for span types
- Source: https://github.com/martinmoene/span-lite by Martin Moene
- License: [LICENSE-BOOST.txt](LICENSE-BOOST.txt)
- **Cannot be relicensed; must retain Boost license and Martin Moene attribution**

### wyhash (The Unlicense — Public Domain)
- Fast hash algorithm by Wang Yi
- Source: https://github.com/wangyi-fudan/wyhash
- License: Public domain (The Unlicense) — see header of [wyhash.h](wyhash.h)

### Boehm-Demers-Weiser GC (Custom)
- Conservative garbage collector
- License headers in gc/ source files
- Version 8.2.8

### LspCpp (MIT)
- Language Server Protocol implementation
- License in [LspCpp/LICENSE](LspCpp/LICENSE)

### libatomic_ops (MIT / GPL-2.0)
- Core library: MIT License (core)
- Extensions: GPL-2.0 (libatomic_ops_gpl.a)
- License in [libatomic_ops/LICENSE](libatomic_ops/LICENSE)

### GLEW (BSD 3-Clause)
- OpenGL Extension Wrangler Library
- License header in [backports/glew/src/glew.c](backports/glew/src/glew.c)

### TinyEXR (BSD 3-Clause)
- OpenEXR image library
- License header in [tinyexr/tinyexr.h](tinyexr/tinyexr.h)

## Distribution Requirements

**All components must be included with their license files in source distributions.**

For modifications: Permissive licenses (MIT, BSD, Apache, Boost, Custom) allow modification if copyright and license are retained. GPL-2.0 code requires source distribution. Always document changes.

For detailed distribution guidance, see [DISTRIBUTION-LICENSE-NOTICE.md](DISTRIBUTION-LICENSE-NOTICE.md).
