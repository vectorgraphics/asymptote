# Asymptote Licensing — Maintainer Reference

## Overview

Asymptote is **LGPL v3+**. It incorporates seven third-party components,
each licensed separately. See [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md)
for the canonical component table.

Full copyright notices and license texts for every component are available via
`asy --licenses=full`, which reads the license files from the `licenses/`
directory at runtime.

## License File Locations

The Asymptote license files (`LICENSE`, `LICENSE.LESSER`) and all third-party
component license files live in a single `licenses/` directory (with
component-prefixed names to avoid collisions).
The canonical mapping (source path → installed name) lives in
[cmake-scripts/copy-build-licenses.cmake](cmake-scripts/copy-build-licenses.cmake).

The same files are also listed in the `copy-licenses` / `uninstall-docdir`
targets of [Makefile.in](Makefile.in).

At build time (`make asy` or CMake build), all license files are copied into
`doc/licenses/` in the build tree so that a locally built `asy` binary can
find them without being installed.

## Configuring the License Installation Directory

By default, license files are installed to `$(docdir)/licenses`
(`${CMAKE_INSTALL_DOCDIR}/licenses` on Linux/CMake). Many distributions
have their own preferred location. This can be overridden:

### Autoconf
```sh
./configure --with-licensedir=/usr/share/licenses/asymptote
```
The default is `$docdir/licenses` (typically
`/usr/share/doc/asymptote/licenses`).

### CMake
```sh
cmake -DASY_LICENSEDIR=/usr/share/licenses/asymptote \
      -DASY_LICENSEDIR_INSTALL_DEST=/usr/share/licenses/asymptote ...
```
`ASY_LICENSEDIR` is the **compile-time** absolute path baked into the binary
(where the binary searches for license files at runtime).
`ASY_LICENSEDIR_INSTALL_DEST` is the CMake `install()` **destination** for the
license files. By default it is derived from `ASY_LICENSEDIR` as a path
relative to `CMAKE_INSTALL_PREFIX` so that `cmake --install --prefix` works
correctly. When `ASY_LICENSEDIR` falls outside `CMAKE_INSTALL_PREFIX`,
`ASY_LICENSEDIR_INSTALL_DEST` defaults to the same absolute path.

The default for both is `${CMAKE_INSTALL_FULL_DATADIR}/doc/asymptote/licenses`.

### Common distro paths
| Distribution | Typical path |
|---|---|
| Fedora / RHEL | `/usr/share/licenses/asymptote` |
| Debian / Ubuntu | `/usr/share/doc/asymptote/licenses` (default) |
| Arch Linux | `/usr/share/licenses/asymptote` |
| macOS (Homebrew) | `$(brew --prefix)/share/doc/asymptote/licenses` |
| macOS (MacPorts) | `/opt/local/share/doc/asymptote/licenses` |

## Adding or Updating Third-Party Components

When adding, removing, or updating a third-party component:
1. Update [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md) (the component table).
2. Update the `licensesSummary` constant and the `printLicensesFull()` function
   in [settings.cc](settings.cc) — add/remove the hardcoded section header and
   the `requireFile()` call for the new component.
3. Update [cmake-scripts/copy-build-licenses.cmake](cmake-scripts/copy-build-licenses.cmake).
4. Update [Makefile.in](Makefile.in) — `copy-licenses` and `uninstall-docdir`.
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
Include the `licenses/` folder alongside the binary, plus
`LICENSES-THIRD-PARTY.md`. The `licenses/` folder contains `LICENSE`,
`LICENSE.LESSER`, and all third-party license files. The `asy --licenses=full`
command reads from this folder and is suitable for auditing and convenience
but is not a substitute for distributing the actual `licenses/` folder.

### OS package managers (apt, yum, brew, macports)
- SPDX license metadata: `LGPL-3.0-or-later AND Unlicense AND MIT AND Boost-1.0 AND BSD-3-Clause AND GPL-2.0-only`
- Place license files using the distro-standard location (see
  "Configuring the License Installation Directory" above).
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
