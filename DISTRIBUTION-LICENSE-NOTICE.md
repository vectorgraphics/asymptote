# Distribution License Compliance Guide

Asymptote incorporates 7 third-party components. All must be included with their licenses in all distributions.

See [LICENSES-THIRD-PARTY.md](LICENSES-THIRD-PARTY.md) for component details.

## License Files Required

| Component | License File | License Type |
|-----------|--------------|--------------|
| Asymptote Core | LICENSE, LICENSE.LESSER | GNU LGPL v3+ |
| span.hpp | backports/span/LICENSE.txt | Boost 1.0 |
| wyhash | wyhash/UNLICENSE.txt | The Unlicense (Public Domain) |
| LspCpp | LspCpp/LICENSE | MIT |
| libatomic_ops | libatomic_ops/LICENSE, COPYING | MIT / GPL-2.0 |
| backports/glew | backports/glew/LICENSE.txt | BSD 3-Clause |
| tinyexr | tinyexr/tinyexr.h | BSD 3-Clause |
| Boehm GC | gc/alloc.c | Custom permissive |

**Documentation files to include**: LICENSES-THIRD-PARTY.md, DISTRIBUTION-LICENSE-NOTICE.md (this file), README

## Source Distribution

✓ Include all components unchanged
✓ Include all license files from above table
✓ Include documentation files
✓ No additional action required

## Binary Distribution

✓ Create LICENSES folder with all license files
✓ Add to Help/About: links to licenses and source availability
✓ Include written notice that source is available at https://github.com/vectorgraphics/asymptote
✓ Include LICENSES-THIRD-PARTY.md with prominent visibility

**Note**: The `asy --licenses=full` command embeds and prints all required copyright
notices and license texts at runtime. It serves as a fallback compliance
mechanism for binary distributions where the separate license files may not be
present. When bundling license files alongside the binary is not practical,
`asy --licenses=full` output alone is intended to satisfy the binary redistribution notice
requirements for all included components. Bundling the `LICENSES` folder and
`LICENSES-THIRD-PARTY.md` alongside the binary is still preferred when feasible.

## TeXLive / Package Manager Distribution

### TeXLive Structure
```
asymptote/
├── doc/asymptote/: LICENSES-THIRD-PARTY.md, DISTRIBUTION-LICENSE-NOTICE.md
└── licenses/asymptote/: All license files from table above
```

### OS Package Managers (apt, yum, brew, macports)
- License metadata: LGPL-3.0-or-later AND Unlicense AND MIT AND Boost-1.0 AND BSD-3-Clause AND GPL-2.0
- Place license files in `/usr/share/licenses/asymptote/` or
`/usr/share/doc/asymptote/licenses`, whichever is standard practice in your distribution.
- Place LICENSES-THIRD-PARTY.md in `/usr/share/doc/asymptote/`
- SPEC/control file should list all applicable licenses

## Modifications and Derivative Works

| Component | License | Key Constraint |
|-----------|---------|-----------------|
| span.hpp | Boost 1.0 | ⚠️ Cannot relicense; retain Martin Moene copyright |
| wyhash | The Unlicense (Public Domain) | No restrictions; attribution appreciated |
| LspCpp | MIT | Retain copyright; incorporate freely into LGPL |
| libatomic_ops | MIT/GPL-2.0 | Core (MIT): retain license. Extensions (GPL-2.0): must be GPL in binaries |
| GLEW, TinyEXR | BSD 3-Clause | Retain license and copyright headers |
| Boehm GC | Custom | Retain copyright notice from source files |

For LGPL core code: retain original copyright, include LICENSE.LESSER, document modifications.

## License Disclosure Examples

### Minimal (About dialog, etc.)
```
Asymptote is free software under GNU LGPL v3+. Includes third-party components:
- span.hpp (Boost License)
- wyhash (Public Domain)
- LspCpp (MIT)
- libatomic_ops (MIT / GPL-2.0)
- GLEW, TinyEXR (BSD 3-Clause)
- Boehm GC (Custom permissive)

See LICENSES-THIRD-PARTY.md for details.
```

### Complete (Documentation)
See third-party component catalog at LICENSES-THIRD-PARTY.md for full information.

## Distribution Verification Checklist

### Source Distribution
- [ ] All components included unchanged
- [ ] All license files present (see table)
- [ ] span.hpp header is unmodified
- [ ] LICENSES-THIRD-PARTY.md and README included

### Binary Distribution
- [ ] All license files accessible to end users
- [ ] LGPL compliance statement visible
- [ ] Source repository link provided
- [ ] Help/About section documents licenses

### TeXLive Package
- [ ] All license files in asymptote.licenses/
- [ ] LICENSES-THIRD-PARTY.md in doc/asymptote/
- [ ] Source repository linked
- [ ] tlpkg metadata lists all licenses

### OS Package Managers
- [ ] License metadata includes all licenses
- [ ] /usr/share/licenses/asymptote/ contains all files
- [ ] /usr/share/doc/asymptote/LICENSES-THIRD-PARTY.md present
- [ ] SPEC/control file documents all licenses

## Support and References

- **LGPL/GPL**: https://www.gnu.org/licenses/
- **Boost License**: https://www.boost.org/LICENSE_1_0.txt
- **Asymptote**: https://github.com/vectorgraphics/asymptote
- **span-lite**: https://github.com/martinmoene/span-lite
- **LspCpp**: https://github.com/kuafuwang/LspCpp
- **wyhash**: https://github.com/wangyi-fudan/wyhash
- **libatomic_ops**: https://boehm.info/atomic_ops/
- **Boehm GC**: https://www.hboehm.info/gc/

---
**Key Principle**: Ensure transparency—end users and developers must know all applicable licenses, access full license text and source code, and understand any restrictions.

