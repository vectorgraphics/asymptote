# Asymptote Licensing Structure - Maintainer Reference

## Overview

**Primary**: Asymptote is LGPL v3+. **Third-party components**: Each licensed separately.
- **backports/span/span.hpp** — Boost 1.0 | **backports/getopt/** — LGPL 2.1+ | **wyhash/** — The Unlicense (Public Domain) | **gc/** — Custom permissive | **LspCpp/** — MIT
- **libatomic_ops/** — MIT (core) / GPL-2.0 (extensions) | **backports/glew/** — BSD 3-Clause | **tinyexr/** — BSD 3-Clause

See [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md) for complete component details.

## Core License Rules

| License | Scope | Key Constraint |
|---------|-------|-----------------|
| LGPL v3+ | All .cc/.h files (except third-party) | Default for new files |
| Boost 1.0 | backports/span/span.hpp only | ⚠️ Cannot be relicensed |
| LGPL 2.1+ | backports/getopt/ | Retain FSF copyright; LGPL 3 satisfies "or later" |
| The Unlicense | wyhash/ | Public domain; compatible with everything |
| MIT | LspCpp/, libatomic_ops (core) | Include copyright notice |
| BSD 3-Clause | backports/glew/, tinyexr/ | Include copyright notice |
| Custom Permissive | gc/ | See gc/alloc.c for terms |
| GPL 2.0 | libatomic_ops (extensions) | Only in binaries with GPL libs |

## Essential Files

| File | Purpose |
|------|---------|
| LICENSE / LICENSE.LESSER | LGPL text |
| backports/span/LICENSE.txt | Boost 1.0 text (alongside span.hpp) |
| backports/glew/LICENSE.txt | BSD 3-Clause + Mesa + Khronos text |
| backports/getopt/LICENSE.txt | LGPL 2.1+ notice |
| LICENSES-THIRD-PARTY.md | Component catalog |
| DISTRIBUTION-LICENSE-NOTICE.md | Distribution guidance |

## Maintenance Checklist

### Adding New Files
- Use LGPL v3+ header (see example below)
- Include copyright year/authors
- Do NOT modify span.hpp unless absolutely necessary

### Modifying backports/span/span.hpp
⚠️ **Cannot be relicensed**. If modifying:
1. Retain Martin Moene copyright (2018-2021)
2. Document all changes clearly
3. Distribute under Boost license
4. Update version number if warranted

### Updating Third-Party Components
Retain original license for: wyhash/, gc/, LspCpp/, libatomic_ops/, backports/span/, backports/glew/, backports/getopt/, tinyexr/
_Note: wyhash/ is public domain, but retain the original header comment crediting Wang Yi._

When adding, removing, or updating third-party components, also update
`licenses.h`, both the short version and the full version.

### Distributing Asymptote
✓ Include: LICENSE, LICENSE.LESSER, LICENSES-THIRD-PARTY.md, DISTRIBUTION-LICENSE-NOTICE.md, README; and in a licenses/ subdirectory: backports/span/LICENSE.txt, backports/glew/LICENSE.txt, backports/getopt/LICENSE.txt, wyhash/UNLICENSE.txt, LspCpp/LICENSE, libatomic_ops/LICENSE, libatomic_ops/COPYING
✓ Follow: DISTRIBUTION-LICENSE-NOTICE.md for your scenario (binary/TeXLive/package manager)
✓ Binaries: Ensure users can access license texts (also available via `asy --licenses=full`)

### Contributing
- All contributions: compatible with LGPL v3+
- span.hpp modifications: Boost license terms apply
- Third-party modifications: retain original license terms
- Adding new components: must be compatible or documented; update LICENSES-THIRD-PARTY.md, licenses.h, DISTRIBUTION-LICENSE-NOTICE.md, MAINTAINER-LICENSE-GUIDE.md (this file), and README.

## LGPL Header Template

```cpp
/*
 * Brief description
 * Copyright YEAR Author
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 */
```

## Quick Scenarios

| Question | Answer |
|----------|--------|
| Can I modify Asymptote? | Yes, under LGPL v3+. If modifying span.hpp: Boost license applies. |
| What if no licenses in distribution? | Violations of LGPL/GPL. Report to maintainers. |
| TeXLive distribution? | Follow DISTRIBUTION-LICENSE-NOTICE.md. |
| What about span.hpp license compliance? | span.hpp is at backports/span/span.hpp; its header references LICENSE.txt in the same directory (Boost license). Original copyright (Martin Moene, 2018-2021) retained. |

## Reference Links

- LGPL/GPL: https://www.gnu.org/licenses/
- Boost License: https://www.boost.org/LICENSE_1_0.txt
- span-lite: https://github.com/martinmoene/span-lite
- Asymptote: https://github.com/vectorgraphics/asymptote

---
*Asymptote v2.0+*
