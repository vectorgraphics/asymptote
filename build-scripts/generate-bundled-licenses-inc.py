#!/usr/bin/env python3
"""generate-bundled-licenses-inc.py  <registry-tsv> <output-inc>

Turns the bundled-dylib license registry (the single source of truth) into a C
include file defining the bundledLicenses[] table that settings.cc walks under
#ifdef ASY_BUNDLED_DYLIBS.  Run at build time from the Makefile.

The registry fields are authored in-repo and known to be C-string-safe (no
embedded quotes, backslashes, or newlines), but we still reject any row that
violates that assumption rather than emit malformed C.
"""

import sys


HEADER = """\
// AUTO-GENERATED from {tsv}
// by build-scripts/generate-bundled-licenses-inc.py -- do not edit.
//
// One row per license file shipped for a dynamic library that the macOS
// bundling build may copy into ./lib.  settings.cc includes this inside
// the #ifdef ASY_BUNDLED_DYLIBS block.
static const struct bundledLicense {{
  const char *summary;    // short name shown by --licenses
  const char *spdx;       // SPDX-style expression
  const char *dylibGlobs; // space-separated fnmatch patterns; the library
                          // is "present" if any ./lib file matches one
  const char *filename;   // basename of the license text in doc/licenses/
  const char *url;        // upstream URL (shown if the license is missing)
}} bundledLicenses[] = {{
"""


def die(*lines):
    for line in lines:
        print(line, file=sys.stderr)
    sys.exit(1)


def read_rows(tsv_path):
    with open(tsv_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 7:
                die("ERROR: malformed registry row (need 7 tab-separated "
                    "fields):", "  " + line)
            yield fields[:7]


def c_safe(value, tsv_path):
    if '"' in value or "\\" in value:
        die("ERROR: registry field is not C-string-safe (contains a quote or "
            "backslash):", "  " + value, "in " + tsv_path)
    return value


def emit(tsv_path):
    out = [HEADER.format(tsv=tsv_path)]
    for keys, summary, spdx, globs, sources, installed, url in read_rows(tsv_path):
        globs_spaced = " ".join(globs.split(":"))
        # Emit one entry per installed license file (a library may ship several).
        for inst in installed.split(":"):
            if not inst:
                continue
            row = (c_safe(summary, tsv_path), c_safe(spdx, tsv_path),
                   c_safe(globs_spaced, tsv_path), c_safe(inst, tsv_path),
                   c_safe(url, tsv_path))
            out.append('  {{"{}", "{}", "{}", "{}", "{}"}},\n'.format(*row))
    out.append("};\n")
    return "".join(out)


def main(argv):
    if len(argv) != 3:
        die("Usage: {} <registry-tsv> <output-inc>".format(argv[0]))
    tsv, out_path = argv[1], argv[2]
    text = emit(tsv)
    if out_path == "-":
        sys.stdout.write(text)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Generated {} from {}.".format(out_path, tsv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
