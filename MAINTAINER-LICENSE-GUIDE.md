# Asymptote Licensing Structure - Maintainer Reference

## Overview

**Primary**: Asymptote is LGPL v3+. **Third-party components**: Each licensed separately.
- **span.hpp** — Boost 1.0 | **wyhash.h** — The Unlicense (Public Domain) | **gc/** — Custom permissive | **LspCpp/** — MIT
- **backports/glew/** — BSD 3-Clause | **tinyexr/** — BSD 3-Clause

See [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md) for complete component details.

## Core License Rules

| License | Scope | Key Constraint |
|---------|-------|-----------------|
| LGPL v3+ | All .cc/.h files (except third-party) | Default for new files |
| Boost 1.0 | span.hpp only | ⚠️ Cannot be relicensed |
| The Unlicense | wyhash.h | Public domain; compatible with everything |
| MIT | LspCpp/ | Include copyright notice |
| BSD 3-Clause | backports/glew/, tinyexr/ | Include copyright notice |
| Custom Permissive | gc/ | See gc/alloc.c for terms |

## Essential Files

| File | Purpose |
|------|---------|
| LICENSE / LICENSE.LESSER | LGPL text |
| LICENSE-BOOST.txt | Boost 1.0 text |
| LICENSES-THIRD-PARTY.md | Component catalog |
| DISTRIBUTION-LICENSE-NOTICE.md | Distribution guidance |

## Maintenance Checklist

### Adding New Files
- Use LGPL v3+ header (see example below)
- Include copyright year/authors
- Do NOT modify span.hpp unless absolutely necessary

### Modifying span.hpp
⚠️ **Cannot be relicensed**. If modifying:
1. Retain Martin Moene copyright (2018-2021)
2. Document all changes clearly
3. Distribute under Boost license
4. Update version number if warranted

### Updating Third-Party Components
Retain original license for: wyhash.h, gc/, LspCpp/, backports/glew/, tinyexr/
_Note: wyhash.h is public domain, but retain the original header comment crediting Wang Yi._

When adding, removing, or updating third-party components, also update the
`licensesOption` output in `settings.cc` (printed by `asy --licenses`).

### Distributing Asymptote
✓ Include: LICENSE, LICENSE.LESSER, LICENSE-BOOST.txt, README, LICENSES-THIRD-PARTY.md
✓ Follow: DISTRIBUTION-LICENSE-NOTICE.md for your scenario (binary/TeXLive/package manager)
✓ Binaries: Ensure users can access license texts (also available via `asy --licenses`)

### Contributing
- All contributions: compatible with LGPL v3+
- span.hpp modifications: Boost license terms apply
- Third-party modifications: retain original license terms
- Adding new components: must be compatible or documented; update LICENSES-THIRD-PARTY.md

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
| What about span.hpp license compliance? | span.hpp header has been updated to reference local LICENSE-BOOST.txt (not external LICENSE.txt) and use HTTPS. Original copyright (Martin Moene, 2018-2021) retained. |

## Reference Links

- LGPL/GPL: https://www.gnu.org/licenses/
- Boost License: https://www.boost.org/LICENSE_1_0.txt
- span-lite: https://github.com/martinmoene/span-lite
- Asymptote: https://github.com/vectorgraphics/asymptote

---
*Asymptote v2.0+*
