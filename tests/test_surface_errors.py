#!/usr/bin/env python3
"""Runtime error tests for surface evaluation and surface coloring.

An error ends the asy process, so each case runs a snippet of Asymptote code
in a process of its own and checks stderr against a regex pattern.  The cases
run in parallel and are reported together, in order, once all have finished.
The values returned when there is no error are tested in tests/three/*.asy.

Usage:
    python3 tests/test_surface_errors.py           # run all tests
    python3 tests/test_surface_errors.py -v        # verbose
    python3 tests/test_surface_errors.py -k PAT    # filter tests by name
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile
import textwrap

# One directory up from this file is the asymptote source root.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ASY = os.path.join(SCRIPT_DIR, "asy")
_DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "base")

# Prepended to the snippets that use the surface named in their first line.

# A surface over box((1,2),(5,8)) with 4 x 3 mesh cells, cyclic in neither
# direction.
_PLANE = """
import graph3;
triple f(pair z) {return (z.x,z.y,0);}
surface s=surface(f,(1,2),(5,8),4,3);
"""

# A cylinder: cyclic in u (the angle, in radians), with 2 mesh cells in v.
_CYLINDER = """
import graph3;
surface s=surface(O,(1,0,0)--(1,0,1)--(1,0,3),Z,4);
"""

# A 3 x 3 grid without the four cells that touch the vertex (2,2).
_HOLE = """
import graph3;
triple f(pair z) {return (z.x,z.y,0);}
bool cond(pair z) {return z != (4,4);}
surface s=surface(f,(0,0),(6,6),3,3,cond);
"""

_COLORS = """
import three;
import palette;
"""

_OUTSIDE = r"lies outside the domain \[{},{}\] of the surface"
_UNSTRUCTURED = r"surface is unstructured \(it has no parametric coordinates\)"

# (name, code, pattern that stderr must match).  Every case costs a process, so
# there is one per kind of error rather than one per direction and method.
CASES: list[tuple[str, str, str]] = [
    # --- evaluation outside the domain of a noncyclic direction ---
    (
        "point just beyond the rounding tolerance",
        _PLANE + "s.point(4+1e-6,1);",
        "u=4.000001 " + _OUTSIDE.format(0, 4),
    ),
    ("point below v", _PLANE + "s.point(1,-2);", "v=-2 " + _OUTSIDE.format(0, 3)),
    ("normal above u", _PLANE + "s.normal(5,1);", "u=5 " + _OUTSIDE.format(0, 4)),
    # paramPoint and paramNormal report parametric coordinates.
    (
        "paramPoint above u",
        _PLANE + "s.paramPoint(6,3);",
        "u=6 " + _OUTSIDE.format(1, 5),
    ),
    (
        "paramNormal below v",
        _PLANE + "s.paramNormal(2,1);",
        "v=1 " + _OUTSIDE.format(2, 8),
    ),
    (
        "noncyclic direction of a cylinder",
        _CYLINDER + "s.point(9.5,2.5);",
        "v=2.5 " + _OUTSIDE.format(0, 2),
    ),
    # --- evaluation where a cell was omitted ---
    (
        "paramPoint in a missing cell",
        _HOLE + "s.paramPoint(3,3);",
        r"no patch at parametric coordinates \(3,3\)",
    ),
    # --- surfaces without parametric coordinates ---
    (
        "paramPoint on an unstructured surface",
        "import three; unitsphere.paramPoint(0,0);",
        "paramPoint: " + _UNSTRUCTURED,
    ),
    (
        "paramNormal on an unstructured surface",
        "import three; unitsphere.paramNormal(0,0);",
        "paramNormal: " + _UNSTRUCTURED,
    ),
    # --- bounds of an empty surface ---
    (
        "palette of a null surface",
        _COLORS + "surface s; s.palette(zpart,Rainbow());",
        "null surface",
    ),
    # --- cornerPen ---
    (
        "cornerPen with no rows",
        _COLORS + "cornerPen(new pen[][]);",
        "cornerPen: no pens specified$",
    ),
    (
        "cornerPen with an empty row",
        _COLORS + "cornerPen(new pen[][] {{red},{}});",
        "cornerPen: no pens specified for patch 1",
    ),
    (
        "cornerPen row shorter than the patch",
        _COLORS
        + """
        picture pic;
        draw(pic,surface(unitsquare3),cornerPen(red,green,blue));
        pic.fit3(currentprojection);
        """,
        "reading array of length 3 with out-of-bounds index 3",
    ),
    # --- fitColors ---
    (
        "fitColors with arrays of different length",
        _COLORS + "fitColors(new triple[] {O,X},new pen[] {red});",
        "coords and colors arrays must have the same length",
    ),
    (
        "fitColors with no data",
        _COLORS + "fitColors(new triple[],new pen[]);",
        "fitColors requires at least one data point",
    ),
]


def run_asy(asy: str, base_dir: str, code: str) -> tuple[str, int]:
    """Write *code* to a temp file, run asy on it, return stderr and return code."""
    fd, tmpfile = tempfile.mkstemp(suffix=".asy")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(textwrap.dedent(code).lstrip("\n"))
        result = subprocess.run(
            [asy, "-q", "-sysdir", base_dir, tmpfile],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stderr, result.returncode
    finally:
        os.unlink(tmpfile)


def _available_cpus() -> int:
    """The number of CPUs this process may use (not the number in the machine)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Runtime error tests for Asymptote surfaces."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument(
        "-k",
        "--filter",
        metavar="PATTERN",
        help="only run tests whose name matches PATTERN (case-insensitive regex)",
    )
    parser.add_argument(
        "--asy",
        default=_DEFAULT_ASY,
        help="path to the asy executable (default: %(default)s)",
    )
    parser.add_argument(
        "--asy-base-dir",
        default=_DEFAULT_BASE_DIR,
        help="path to the asy base/sysdir (default: %(default)s)",
    )
    args = parser.parse_args()

    cases = [
        case
        for case in CASES
        if not args.filter or re.search(args.filter, case[0], re.IGNORECASE)
    ]

    # The cases are independent processes, so run one per available core.  The
    # results come back in the order of CASES whichever finishes first, and
    # nothing is printed until then, so the report reads the same as a serial
    # run.
    with concurrent.futures.ThreadPoolExecutor(max_workers=_available_cpus()) as pool:
        results = list(
            pool.map(lambda case: run_asy(args.asy, args.asy_base_dir, case[1]), cases)
        )

    print(f"Testing runtime errors (surfaces, asy = {args.asy})")
    passed = failed = 0
    for (name, _, pattern), (stderr, returncode) in zip(cases, results):
        ok = returncode != 0 and bool(re.search(pattern, stderr, re.MULTILINE))
        if ok:
            passed += 1
        else:
            failed += 1
        if args.verbose or not ok:
            print(f"  {name} ... {'PASSED' if ok else 'FAILED'}")
        if not ok:
            print(f"    Expected pattern: {pattern!r}")
            print(f"    Got:              {stderr!r}")

    print(f"{passed}/{passed + failed} passed")
    if not failed:
        print("PASSED.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
