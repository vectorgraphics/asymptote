#!/usr/bin/env python3
"""Fetch a GNU source archive from a mirror: gnu-mirror.py URL DST SHA512

Used as a vcpkg asset-caching script (x-script). vcpkg runs it before trying
a port's own URLs and falls back to those when it exits nonzero. Ports fetch
GNU sources from ftpmirror.gnu.org and ftp.gnu.org, which are at times
unreachable from the CI runners for hours.

vcpkg passes only the first of a port's URLs, so a port that lists the GNU
servers after a site of its own needs that site in ORIGINS.

The mirrors are trusted only to be available: a download is kept only if it
has the SHA512 the port expects. vcpkg does not run a script whose template
names {sha512} for a download without one, so those never come from a mirror.
"""

import contextlib
import hashlib
import http.client
import os
import re
import sys
import time
import urllib.request

MIRRORS = [
    "https://mirrors.kernel.org/gnu/",
    "https://mirrors.ocf.berkeley.edu/gnu/",
    "https://www.mirrorservice.org/sites/ftp.gnu.org/gnu/",
]

# First URLs of downloads that the mirrors carry too. The group is the path
# of the file below a mirror's gnu/ directory.
ORIGINS = [
    r"https?://(?:ftpmirror|ftp)\.gnu\.org/(?:gnu/)?(.+)",
    r"https?://invisible-mirror\.net/archives/(ncurses/.+)",
]

# Per mirror. The largest GNU archives are tens of megabytes.
DEADLINE = 120  # seconds
MAX_SIZE = 1 << 30  # bytes


def fetch(source: str, dst: str, sha512: str) -> None:
    """Download source to dst; raise OSError unless it has the given hash."""
    deadline = time.monotonic() + DEADLINE
    digest = hashlib.sha512()
    size = 0
    with urllib.request.urlopen(source, timeout=30) as src:
        with open(dst, "wb") as out:
            # The timeout above restarts with every byte received. read1
            # returns after a single socket read, so that a mirror trickling
            # its data cannot keep the loop from checking the deadline.
            while True:
                chunk = src.read1(1 << 16)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_SIZE:
                    raise OSError(f"more than {MAX_SIZE} bytes")
                if time.monotonic() > deadline:
                    raise OSError(f"not complete after {DEADLINE} seconds")
                digest.update(chunk)
                out.write(chunk)
    if digest.hexdigest() != sha512.lower():
        raise OSError(f"SHA512 mismatch: got {digest.hexdigest()}")


def main() -> int:
    url, dst, sha512 = sys.argv[1:4]
    matches = (re.match(origin, url) for origin in ORIGINS)
    path = next((match[1] for match in matches if match), None)
    if path is None:
        return 1
    for mirror in MIRRORS:
        try:
            fetch(mirror + path, dst, sha512)
            print(f"Fetched {url} from {mirror}")
            return 0
        except (OSError, http.client.HTTPException) as error:
            print(f"{mirror}: {error!r}", file=sys.stderr)
            with contextlib.suppress(FileNotFoundError):
                os.remove(dst)
    return 1


if __name__ == "__main__":
    sys.exit(main())
