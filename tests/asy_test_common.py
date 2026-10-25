"""Shared helpers for the standalone Asymptote test scripts in tests/.

Those scripts are run directly (``python3 tests/test_foo.py``) with this
directory on ``sys.path``, so they import this module to share the small amount
of boilerplate each one would otherwise repeat: locating the asy executable and
base directory, running a snippet of Asymptote code in a subprocess, and the
matching ``--asy`` / ``--asy-base-dir`` command-line options.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import textwrap
from collections.abc import Sequence

# One directory up from this file is the asymptote source root.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ASY = os.path.join(SCRIPT_DIR, "asy")
DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "base")


def add_asy_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the --asy and --asy-base-dir options used by the test scripts."""
    parser.add_argument(
        "--asy",
        default=DEFAULT_ASY,
        help="path to the asy executable (default: %(default)s)",
    )
    parser.add_argument(
        "--asy-base-dir",
        default=DEFAULT_BASE_DIR,
        help="path to the asy base/sysdir (default: %(default)s)",
    )


def run_asy(
    asy: str,
    base_dir: str,
    code: str,
    extra_args: Sequence[str] = (),
) -> tuple[str, int]:
    """Write *code* to a temporary .asy file and run asy on it.

    Returns a (stderr, returncode) pair.  Any *extra_args* are inserted into
    the command line between ``-q`` and ``-sysdir``.
    """
    fd, tmpfile = tempfile.mkstemp(suffix=".asy")
    try:
        # The Asy snippets are embedded in Python triple-quoted strings, which
        # inherit Python indentation; dedent them so the .asy files read cleanly.
        with os.fdopen(fd, "w") as f:
            f.write(textwrap.dedent(code).lstrip("\n"))
        result = subprocess.run(
            [asy, "-q", *extra_args, "-sysdir", base_dir, tmpfile],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stderr, result.returncode
    finally:
        os.unlink(tmpfile)
