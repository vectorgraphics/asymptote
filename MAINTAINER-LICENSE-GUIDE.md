# Asymptote Licensing Structure - Maintainer Reference

## Overview

Asymptote is primarily licensed under the GNU Lesser General Public License (LGPL) v3 or later, with one exception: the span.hpp header file is distributed under the Boost Software License.

## Files and Their Purposes

### License Files

| File | Purpose |
|------|---------|
| LICENSE | Full text of GNU General Public License v3 |
| LICENSE.LESSER | Full text of GNU Lesser General Public License v3 |
| LICENSE-BOOST.txt | Full text of Boost Software License v1.0 |

### Documentation Files

| File | Purpose |
|------|---------|
| README | Project overview; includes license and span.hpp notice |
| LICENSES-THIRD-PARTY.md | Details of all third-party components and licenses |
| DISTRIBUTION-LICENSE-NOTICE.md | Guide for distributors (binary, TeXLive, packages) |
| MAINTAINER-LICENSE-GUIDE.md | This file |

## When to Use Each License

### GNU LGPL (LICENSE.LESSER) - Default
- All original Asymptote source files
- Most .cc, .h, .h files in the repository
- Use this unless explicitly otherwise noted
- To use: Include statement at top of file, or reference toward top
- For binaries: Include full copy of LICENSE and LICENSE.LESSER

### GNU GPL (LICENSE) - Linked Binaries Only
- Windows binaries linked against GPL libraries (GSL, Readline)
- Referenced in README under "MSWindows" section
- Not typically needed for source distribution
- Only applies to compiled binaries with GPL dependencies

### Boost Software License (LICENSE-BOOST.txt) - span.hpp Only
- **ONLY applies to**: span.hpp header file
- Modern C++98/11/17/20 span implementation
- Based on C++ Standards Committee Paper P0122R7
- Copyright: Martin Moene
- Source: https://github.com/martinmoene/span-lite
- Cannot be relicensed; modifications must retain Boost license

## Maintenance Tasks

### When Adding New Source Files
1. Use LGPL v3+ as default license
2. Include copyright year and authors
3. Include license reference comment
4. Ensure they're not modifying span.hpp

### When Updating span.hpp
⚠️ **Critical**: span.hpp cannot be relicensed
1. Document all changes
2. Keep original copyright notice (Martin Moene, 2018-2021)
3. Update version number if warranted
4. Add your copyright claim as contributor
5. Ensure distributed under Boost license
6. Update span_lite_MAJOR/MINOR/PATCH if applicable

### When Distributing Asymptote
1. ✓ Include all three license files (LICENSE, LICENSE.LESSER, LICENSE-BOOST.txt)
2. ✓ Include README with license section
3. ✓ Include LICENSES-THIRD-PARTY.md
4. ✓ Include DISTRIBUTION-LICENSE-NOTICE.md (recommended)
5. ✓ Follow guidance in DISTRIBUTION-LICENSE-NOTICE.md for your distribution type
6. ✓ For binaries: ensure users can access license texts

### When Contributing to Asymptote
1. Review CONTRIBUTOR licensing: all contributions should be compatible with LGPL v3+
2. For modifications to span.hpp: Boost license terms apply
3. Include copyright notice if creating new functions/modules
4. Respect existing copyright holders

## License Headers

### Standard Header for New LGPL Files
```cpp
/*
 * (Filename and brief description)
 * Copyright YEAR Author/Company
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 */
```

### Existing LGPL Header Example
See any .cc or .h file in the repository; most follow this pattern.

### Boost License Header
```cpp
//
// span for C++98 and later.
// Based on http://wg21.link/p0122r7
// For more information see https://github.com/martinmoene/span-lite
//
// Copyright 2018-2021 Martin Moene
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE-BOOST.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
```

**Note on Header References**: When distributing span.hpp in Asymptote, the header comment has been updated to:
- Point to the local `LICENSE-BOOST.txt` file instead of a generic `LICENSE.txt`
- Use HTTPS for the reference URL (for security and availability)

These are documentation-only changes that do not modify the software itself, the license terms, or the copyright attribution. The original copyright notice for Martin Moene and the span-lite project is retained.

## Common Scenarios and Solutions

### Scenario: User asks "Can I modify Asymptote and redistribute?"
**Answer**: Yes, under LGPL v3+. If modifying span.hpp, Boost license applies.

### Scenario: Binary being distributed without source
**Answer**: Ensure licenses are accessible. Provide link to source at https://github.com/vectorgraphics/asymptote

### Scenario: Including in TeXLive
**Answer**: Follow DISTRIBUTION-LICENSE-NOTICE.md section on "TeXLive Standard Package"

### Scenario: User finds Asymptote in Linux distro without licenses
**Answer**: Report to distro maintainers that they're violating LGPL/GPL. Point to DISTRIBUTION-LICENSE-NOTICE.md

### Scenario: Modifying span.hpp
**Answer**: ⚠️ You can modify but cannot change license. Document changes and distribute under Boost license.

## Questions?

- **LGPL/GPL Questions**: https://www.gnu.org/licenses/
- **Boost License**: https://www.boost.org/LICENSE_1_0.txt  
- **span-lite Project**: https://github.com/martinmoene/span-lite
- **Asymptote Issues**: https://github.com/vectorgraphics/asymptote/issues

---
*Last Updated: March 2026*
*This guide applies to Asymptote version 2.0 and later*
