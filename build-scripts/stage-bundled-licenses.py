#!/usr/bin/env python3
"""stage-bundled-licenses.py  <lib-dir> <dest-dir> <registry-tsv>

Copies the in-repo vendored license texts for the dynamic libraries that were
actually bundled into <lib-dir>/*.dylib into <dest-dir> (normally doc/licenses).
Driven by the registry TSV, which is the single source of truth
(build-scripts/bundled-dylib-licenses.tsv).

Called by the top-level Makefile when BUNDLED_DYLIBS_LICENSES=yes, right after
bundle-vulkan-macos.py has populated ./lib.  Idempotent.
"""

import os
import re
import shutil
import stat
import sys


def die(*lines):
    for line in lines:
        print(line, file=sys.stderr)
    sys.exit(1)


def dylib_key(basename):
    """Key-derivation rule (must match the runtime check and the registry):
    strip the leading "lib", the trailing ".dylib", and any trailing
    ".<version>" segments, then lowercase.  libMoltenVK.dylib -> moltenvk,
    libvulkan.1.4.350.dylib -> vulkan, libSPIRV.16.dylib -> spirv.
    """
    name = re.sub(r"\.dylib$", "", basename)
    name = re.sub(r"(\.[0-9]+)*$", "", name)
    name = re.sub(r"^lib", "", name)
    return name.lower()


def read_registry(tsv_path):
    """Yield (keys, summary, spdx, globs, sources, installed, url) tuples,
    skipping blank lines and comment rows."""
    with open(tsv_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 7:
                die("ERROR: malformed registry row (need 7 tab-separated "
                    "fields):", "  " + line)
            yield tuple(fields[:7])


def main(argv):
    if len(argv) != 4:
        die("Usage: {} <lib-dir> <dest-dir> <registry-tsv>".format(argv[0]))
    libdir, destdir, tsv = argv[1], argv[2], argv[3]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(argv[0])))

    if not os.path.isfile(tsv):
        die("ERROR: registry not found: " + tsv)

    # 1. Compute the set of library keys present in LIBDIR.
    present_keys = set()
    if os.path.isdir(libdir):
        for entry in os.listdir(libdir):
            if entry.endswith(".dylib"):
                present_keys.add(dylib_key(entry))

    if not present_keys:
        print("No bundled dylibs found in {}; nothing to stage.".format(libdir))
        return 0

    os.makedirs(destdir, exist_ok=True)

    # 2. Walk the registry.  A row applies if any of its (colon-separated) keys
    #    is present in LIBDIR; if so, stage its license file(s).
    matched_keys = set()
    for keys, summary, spdx, globs, sources, installed, url in read_registry(tsv):
        row_keys = [k for k in keys.split(":") if k]
        hit = [k for k in row_keys if k in present_keys]
        if not hit:
            continue
        matched_keys.update(hit)

        # 3. Stage source_files -> installed_names, pairwise.
        src_list = [s for s in sources.split(":") if s]
        dst_list = [d for d in installed.split(":") if d]
        for i, src in enumerate(src_list):
            dst = dst_list[i] if i < len(dst_list) else None
            if dst is None:
                die("ERROR: registry entry '{}' has more source files than "
                    "installed names.".format(summary))
            srcpath = os.path.join(repo_root, src)
            if not os.path.isfile(srcpath):
                die("ERROR: a bundled dylib matched registry entry '{}',"
                    .format(summary),
                    "       but its vendored license file is missing:",
                    "         " + src,
                    "       This text must be committed before release.")
            destpath = os.path.join(destdir, dst)
            shutil.copyfile(srcpath, destpath)
            # Ensure the staged copy is writable (vendored sources may be ro).
            mode = os.stat(destpath).st_mode
            os.chmod(destpath, mode | stat.S_IWUSR)
            print("Staged license: {} ({})".format(dst, summary))

    # 4. Warn (non-fatal) about any bundled dylib with no registry entry, so the
    #    maintainer adds its license before release.
    for p in sorted(present_keys - matched_keys):
        print("WARNING: bundled dynamic library with key '{}' has no entry in"
              .format(p), file=sys.stderr)
        print("         {}; no license was staged for it. Please add a"
              .format(tsv), file=sys.stderr)
        print("         registry row (see MAINTAINER-LICENSE-GUIDE.md) so its",
              file=sys.stderr)
        print("         license ships alongside the binary.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
