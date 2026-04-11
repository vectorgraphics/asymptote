# Asymptote Licensing — Maintainer Reference

## Overview

Asymptote is **LGPL v3+**. It incorporates seven third-party components,
each licensed separately. See [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md)
for the canonical component table.

Full copyright notices and license texts for every component are embedded in
the binary and available via `asy --licenses=full` (source:
[licenses.h](licenses.h)).

## License File Locations

Each third-party component has a license file in its source directory.
The install rules copy them into a `licenses/` subdirectory with
component-prefixed names to avoid collisions. The canonical list of
source paths → installed names is in
[cmake-scripts/install-third-party-licenses.cmake](cmake-scripts/install-third-party-licenses.cmake).

The same files are also listed in the `install-licenses` / `uninstall-docdir`
targets of [Makefile.in](Makefile.in).

## Adding or Updating Third-Party Components

When adding, removing, or updating a third-party component:
1. Update [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md) (the component table).
2. Update [licenses.h](licenses.h) — both the `summary` and `full` strings.
3. Update [cmake-scripts/install-third-party-licenses.cmake](cmake-scripts/install-third-party-licenses.cmake).
4. Update [Makefile.in](Makefile.in) — `install-licenses` and `uninstall-docdir`.
5. Retain the component's original license file and headers unchanged.

_Note: wyhash/ is public domain, but retain the original header comment crediting Wang Yi._

## Adding New Asymptote Source Files

- Use the LGPL v3+ header template below.
- Include copyright year and authors.

## Modifying backports/span/span.hpp

⚠️ **Cannot be relicensed.** If modifying:
1. Retain Martin Moene copyright (2018-2021).
2. Document all changes clearly.
3. Distribute under the Boost license.

## Distributing Asymptote

### Source distributions
Include all components unchanged with their license files. No additional
action is required beyond what the repository already contains.

### Binary distributions
Include the `licenses/` folder alongside the binary, plus LICENSE,
LICENSE.LESSER, and LICENSES-THIRD-PARTY.md. When bundling files is not
practical, the `asy --licenses=full` output is intended to satisfy the
notice requirements for all included components on its own.

### OS package managers (apt, yum, brew, macports)
- SPDX license metadata: `LGPL-3.0-or-later AND Unlicense AND MIT AND Boost-1.0 AND BSD-3-Clause AND GPL-2.0-only`
- Place license files in the distro-standard location (e.g.
  `/usr/share/licenses/asymptote/` or `/usr/share/doc/asymptote/licenses`).
- SPEC/control file should list all applicable licenses.

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

## Reference Links

- LGPL/GPL: https://www.gnu.org/licenses/
- Boost License: https://www.boost.org/LICENSE_1_0.txt
- Asymptote: https://github.com/vectorgraphics/asymptote
