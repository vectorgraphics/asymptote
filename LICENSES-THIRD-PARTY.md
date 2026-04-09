# Third-Party Licenses

Asymptote incorporates the following third-party components:

| Component | License | Location |
|-----------|---------|----------|
| span.hpp | Boost 1.0 | [backports/span/span.hpp](backports/span/span.hpp) |
| GNU getopt | LGPL 2.1+ | [backports/getopt/](backports/getopt/) |
| wyhash | The Unlicense (Public Domain) | [wyhash/](wyhash/) |
| Boehm GC | Custom permissive | [gc/](gc/) |
| LspCpp | MIT | [LspCpp/](LspCpp/) |
| libatomic_ops | MIT / GPL-2.0 | [libatomic_ops/](libatomic_ops/) |
| GLEW | BSD 3-Clause | [backports/glew/](backports/glew/) |
| TinyEXR | BSD 3-Clause | [tinyexr/](tinyexr/) |

## Component Details

### span.hpp (Boost 1.0)
- Header-only C++ library for span types
- Source: https://github.com/martinmoene/span-lite by Martin Moene
- Location: [backports/span/span.hpp](backports/span/span.hpp)
- License: [backports/span/LICENSE.txt](backports/span/LICENSE.txt) 
- **Cannot be relicensed; must retain Boost license and Martin Moene attribution**

### GNU getopt (LGPL 2.1+)
- Command-line option parsing from the GNU C Library
- Source: https://www.gnu.org/software/libc/ by Free Software Foundation, Inc.
- Location: [backports/getopt/](backports/getopt/)
- License: [backports/getopt/LICENSE.txt](backports/getopt/LICENSE.txt) (LGPL 2.1 or later; compatible with project's LGPL 3+)

### wyhash (The Unlicense — Public Domain)
- Fast hash algorithm by Wang Yi
- Source: https://github.com/wangyi-fudan/wyhash
- Location: [wyhash/wyhash.h](wyhash/wyhash.h)
- License: Public domain (The Unlicense) — [wyhash/UNLICENSE.txt](wyhash/UNLICENSE.txt)

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
- License: [backports/glew/LICENSE.txt](backports/glew/LICENSE.txt)
- Also covers Mesa (Brian Paul, MIT-style) and Khronos Group (MIT-style) portions

### TinyEXR (BSD 3-Clause)
- OpenEXR image library
- License header in [tinyexr/tinyexr.h](tinyexr/tinyexr.h)

## Distribution Requirements

**All components must be included with their license files in source distributions.**

For modifications: Permissive licenses (MIT, BSD, Apache, Boost, Custom) allow modification if copyright and license are retained. GPL-2.0 code requires source distribution. Always document changes.

For detailed distribution guidance, see [DISTRIBUTION-LICENSE-NOTICE.md](DISTRIBUTION-LICENSE-NOTICE.md).
