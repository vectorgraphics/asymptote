# Distribution and License Compliance Guide

This document provides guidance on maintaining license compliance when distributing Asymptote in various forms, including as a binary, standalone application, or as part of larger distributions like TeXLive.

## Source Distribution

When distributing Asymptote source code:
- Include all license files: LICENSE, LICENSE.LESSER, LICENSE-BOOST.txt
- Include this guide: DISTRIBUTION-LICENSE-NOTICE.md
- Include LICENSES-THIRD-PARTY.md
- The span.hpp header file already contains its own license notice

**No additional action required beyond standard source distribution.**

## Binary Distribution

When distributing Asymptote as a compiled binary or application:

### Minimum Requirements
- Include or reference: LICENSE, LICENSE.LESSER, LICENSE-BOOST.txt
- Include LICENSES-THIRD-PARTY.md with prominent visibility
- Provide written notice that Asymptote is licensed under LGPL v3+
- Document the Boost Software License for span.hpp
- Provide information on how to obtain source code (GPL requirement for binaries)

### Best Practices
- Create a LICENSES folder/section in your distribution containing all license files
- Add to Help menu or About dialog: links to licenses and source availability
- On installation: display or provide link to license information
- Include a COPYING file in the binary distribution

## TeXLive or Package Manager Distribution

When including Asymptote in TeXLive, operating system packages, or other distributions:

### Scope
You must ensure End Users receive:
- Information about Asymptote's LGPL v3+ license
- Full text of LICENSE and LICENSE.LESSER
- Full text of LICENSE-BOOST.txt
- Reference to LICENSES-THIRD-PARTY.md
- Access to source code for GPL compliance

### Specific Requirements for TeXLive
- Include all license files in the Asymptote package directory
- Document the LGPL v3+ licensing in package metadata
- Provide the Boost license as accessible documentation
- If packaging includes source: include as-is without modification
- If packaging includes binaries only: provide link to source repository

### OS Package Managers (apt, yum, brew, etc.)
- License metadata: LGPL-3.0-or-later AND Boost-1.0
- Include docs: /usr/share/doc/asymptote/LICENSES-THIRD-PARTY.md
- Include: /usr/share/licenses/asymptote/LICENSE-BOOST.txt
- SPEC file or package control file should reference the license

## Modifications and Derivative Works

If you modify Asymptote source code:

### For LGPL Code
- Retain original copyright notices
- Include list of modifications
- Distribute modified source code under LGPL v3+
- Include original LICENSE and LICENSE.LESSER files

### For span.hpp
- Retain the copyright notice for Martin Moene and span-lite project
- Cannot relicense; must remain under Boost Software License
- Document all changes made
- Distribute modified source under Boost Software License
- Include LICENSE-BOOST.txt

## Documentation and Disclaimers

### What to Include in Documentation
```
Asymptote is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

See LICENSE and LICENSE.LESSER for the full text.

Asymptote includes span.hpp, distributed under the Boost Software License.
See LICENSE-BOOST.txt for details.
```

### For Package Index Pages
Include clear classification:
- **Primary License**: LGPL-3.0-or-later
- **Secondary Licenses**: Boost-1.0
- **Source Location**: https://github.com/vectorgraphics/asymptote

## Verification Checklist

Before distributing Asymptote, verify:

### Source Distribution
- [ ] All .cc, .h files are included with no modifications
- [ ] LICENSE, LICENSE.LESSER files present
- [ ] LICENSE-BOOST.txt present
- [ ] LICENSES-THIRD-PARTY.md present
- [ ] README includes span.hpp licensing information
- [ ] span.hpp header is unmodified or changes are documented

### Binary Distribution
- [ ] License files accessible to end users
- [ ] Help/About section documents licenses
- [ ] LGPL compliance statement visible
- [ ] Boost license information provided
- [ ] Source code availability documented
- [ ] span.hpp.txt information is included if span.hpp was modified

### TeXLive Standard Package
- [ ] package-name.doc/LICENSES-THIRD-PARTY.md
- [ ] package-name.doc/LICENSE-BOOST.txt  
- [ ] tlpkg contains license information
- [ ] Source repository linked in documentation

## Questions and Support

For questions about licensing compliance:
- LGPL/GPL inquiries: Refer to https://www.gnu.org/licenses/
- Boost license inquiries: Refer to https://www.boost.org/LICENSE_1_0.txt
- Specific Asymptote licensing: Contact the Asymptote project
- span-lite: https://github.com/martinmoene/span-lite

## Summary

The key principle is **transparency**: ensure end users and developers know what licenses apply and can access the full license text and source code. This guide helps achieve that regardless of how Asymptote is distributed.
