#!/usr/bin/env python3
"""Test that mesh shadings keep their coordinates when converted to PDF.

Asymptote writes Gouraud and tensor-product shadings (PostScript shading types
4 and 7) as PostScript, and Ghostscript's pdfwrite device converts them to PDF.
pdfwrite stores the coordinates of such a shading as fixed-point numbers whose
resolution and range depend on the Ghostscript version, so coordinates that
are written carelessly come out rounded to whole points (leaving gaps between
adjacent patches) or clamped (collapsing the patches).

This test draws a few shadings, reads the coordinates Asymptote wrote from the
EPS output and the coordinates Ghostscript stored from the PDF output, and
requires them to agree.  Nothing is rasterized.  The two files differ by the
position of the page origin, so the comparison is made up to one translation
common to all the shadings.

The test exercises whichever Ghostscript is used, so to cover several versions
run it once for each:

    python3 testsExtra/test_mesh_shading.py              # default Ghostscript
    python3 testsExtra/test_mesh_shading.py --gs PATH    # a particular one
    python3 testsExtra/test_mesh_shading.py -v           # list every shading

Without --gs, asy chooses Ghostscript as usual: the environment variable
ASYMPTOTE_GS if it is set, and otherwise gs on the PATH.

The test depends on how pdfwrite lays out a shading in the PDF file.  If a
future version lays it out differently, the test reports that it could not
read the file rather than passing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zlib
from typing import NamedTuple

# One directory up from this file is the asymptote source root.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ASY = os.path.join(SCRIPT_DIR, "asy")
_DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "base")

# The coordinates are deliberately not whole points, and the shadings differ
# in shape so that each one in the PDF file can be matched to its source.
_SOURCE = """
unitsize(1bp);
pen[] p={red,green,blue,yellow};
path g=(0.3,0.7){dir(45)}..(101.2,3.9)..(97.6,88.1)..(2.4,93.3)..cycle;

// A Coons patch, whose interior control points asy computes.
tensorshade(g,p);

// A tensor-product patch with interior control points specified.
tensorshade(shift(140.37,20.61)*scale(0.4)*g,p,
            new pair[] {(160.1,40.2),(170.3,41.7),(171.9,50.4),(158.8,52.6)});

// Two patches in one shading.
path h=(0.2,250.6)--(30.4,251.3)--(31.7,280.9)--(0.9,282.2)--cycle;
tensorshade(new path[] {h,shift(31.13,0.37)*h},new pen[][] {p,p});

// A Gouraud-shaded triangle.
gouraudshade((0,150)--(60,150)--(0,210)--cycle,new pen[] {red,green,blue},
             new pair[] {(0.25,150.3),(60.7,151.9),(1.1,209.6)},
             new int[] {0,0,0});

// A small patch far from the others: the precision must not depend on the
// position on the page.
tensorshade(shift(2987.43,3141.59)*scale(0.25)*g,p);

// A large patch, which leaves less room for scaling the coordinates.
tensorshade(shift(300.3,0.6)*scale(6.5)*g,p);
"""

# The coordinates of a shading must agree to within the larger of
# _TOLERANCE and its size divided by _RELATIVE_TOLERANCE, in PostScript
# points.  The second term allows for the resolution that remains when a large
# shading has to fit the coordinate range of Ghostscript 10.02 (16384).
_TOLERANCE = 0.01
_RELATIVE_TOLERANCE = 8192

# The number of control points of a patch with edge flag 0.
_PATCH_POINTS = {6: 12, 7: 16}

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


class TestError(Exception):
    """The test could not be carried out."""


class Shading(NamedTuple):
    """A mesh shading: its PostScript type and its points in page units."""

    kind: int
    points: "list[tuple[float, float]]"

    def size(self) -> float:
        """The larger dimension of the bounding box."""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def centroid(self) -> "tuple[float, float]":
        """The mean of the points."""
        n = len(self.points)
        return (
            sum(p[0] for p in self.points) / n,
            sum(p[1] for p in self.points) / n,
        )


def run_asy(args: argparse.Namespace, source: str, fmt: str, prefix: str) -> str:
    """Run asy on *source*, writing *prefix*.*fmt*; return the output file."""
    command = [
        args.asy,
        "-q",
        "-config",
        "",
        "-sysdir",
        args.asy_base_dir,
        "-noV",
        "-f",
        fmt,
        "-o",
        prefix,
    ]
    if args.gs:
        command += ["-gs", args.gs]
    command.append(source)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
    except OSError as e:
        raise TestError(f"cannot run {args.asy}: {e}") from e
    output = f"{prefix}.{fmt}"
    if result.returncode != 0 or not os.path.exists(output):
        raise TestError(
            f"asy did not produce {fmt} output:\n{result.stdout}{result.stderr}"
        )
    return output


def _layout(kind: int) -> "tuple[int, int]":
    """Return how many points and how many colors a record of a shading of
    the given type has: a vertex of a triangle mesh, or a whole patch."""
    if kind == 4:
        return 1, 1
    if kind in _PATCH_POINTS:
        return _PATCH_POINTS[kind], 4
    raise TestError(f"unexpected shading type {kind}")


# The shift and scaling that asy writes on the line before a shading, and the
# shading itself.
_EPS_FRAME = re.compile(
    rf"^(?:({_NUMBER}) ({_NUMBER}) translate )?1 ({_NUMBER}) div dup scale\n$"
)
_EPS_SHADING = re.compile(
    r"<< /ShadingType (\d+)\n/ColorSpace /Device(\w+)\n/DataSource \[\n"
    r"(.*?)\n\]\n>>\nshfill\n",
    re.DOTALL,
)
_COMPONENTS = {"Gray": 1, "RGB": 3, "CMYK": 4}


def _eps_points(match: "re.Match[str]", before: str) -> Shading:
    """Return the shading of a match of _EPS_SHADING, given the line of the
    file that precedes it."""
    kind = int(match.group(1))
    npoints, ncorners = _layout(kind)
    length = 1 + 2 * npoints + ncorners * _COMPONENTS[match.group(2)]
    framed = _EPS_FRAME.match(before)
    ox, oy, scale = (float(v or 0) for v in framed.groups()) if framed else (0, 0, 1)

    points = []
    for line in match.group(3).split("\n"):
        values = [float(v) for v in line.split()]
        if len(values) != length or values[0] != 0:
            raise TestError(f"cannot read shading data: {line}")
        for i in range(1, 1 + 2 * npoints, 2):
            points.append((ox + values[i] / scale, oy + values[i + 1] / scale))
    return Shading(kind, points)


def read_eps(filename: str) -> "list[Shading]":
    """Return the mesh shadings in an EPS file written by asy.

    The coordinates are those of the user space outside the shift and scaling
    that asy puts around a shading.
    """
    with open(filename, encoding="latin-1") as f:
        text = f.read()

    shadings = []
    for match in _EPS_SHADING.finditer(text):
        start = text.rfind("\n", 0, match.start() - 1) + 1
        try:
            shadings.append(_eps_points(match, text[start : match.start()]))
        except TestError as e:
            raise TestError(f"{filename}: {e}") from e
    return shadings


def _pdf_objects(data: bytes) -> "dict[int, tuple[bytes, bytes | None]]":
    """Return the dictionary and the stream of each top-level object."""
    objects = {}
    pattern = re.compile(
        rb"(\d+) 0 obj\s*(<<.*?>>)\s*(?:stream\r?\n(.*?)endstream\s*)?endobj",
        re.DOTALL,
    )
    for match in pattern.finditer(data):
        objects[int(match.group(1))] = (match.group(2), match.group(3))
    return objects


def _numbers(dictionary: bytes, key: bytes) -> "list[float]":
    match = re.search(rb"/" + key + rb"\s*\[([^\]]*)\]", dictionary)
    if not match:
        raise TestError(f"no /{key.decode()} array in {dictionary!r}")
    return [float(v) for v in match.group(1).split()]


def _integer(dictionary: bytes, key: bytes) -> int:
    match = re.search(rb"/" + key + rb"\s+(\d+)", dictionary)
    if not match:
        raise TestError(f"no /{key.decode()} in {dictionary!r}")
    return int(match.group(1))


def _mesh_points(mesh: bytes, stream: bytes) -> Shading:
    """Decode a mesh shading from its dictionary and its data, with the
    coordinates in the space of the shading."""
    # The end-of-line before endstream is not part of the data.
    stream = stream[: _integer(mesh, b"Length")]
    if b"/FlateDecode" in mesh:
        stream = zlib.decompress(stream)
    elif b"/Filter" in mesh:
        raise TestError(f"unsupported filter: {mesh!r}")

    kind = _integer(mesh, b"ShadingType")
    npoints, ncorners = _layout(kind)
    decode = _numbers(mesh, b"Decode")
    bits = [
        _integer(mesh, key)
        for key in (b"BitsPerCoordinate", b"BitsPerComponent", b"BitsPerFlag")
    ]
    if any(n % 8 for n in bits):
        raise TestError(f"fields are not whole bytes: {mesh!r}")
    size, color, flag = (n // 8 for n in bits)
    record = flag + 2 * npoints * size + ncorners * (len(decode) // 2 - 2) * color
    if not stream or len(stream) % record:
        raise TestError(
            f"shading data of {len(stream)} bytes is not made of "
            f"{record}-byte records: {mesh!r}"
        )

    def coordinate(at: int, low: float, high: float) -> float:
        raw = int.from_bytes(stream[at : at + size], "big")
        return low + raw * (high - low) / ((1 << 8 * size) - 1)

    points = []
    for start in range(0, len(stream), record):
        if any(stream[start : start + flag]):
            raise TestError(f"unexpected edge flag: {mesh!r}")
        for at in range(start + flag, start + flag + 2 * npoints * size, 2 * size):
            points.append(
                (
                    coordinate(at, decode[0], decode[1]),
                    coordinate(at + size, decode[2], decode[3]),
                )
            )
    return Shading(kind, points)


def read_pdf(filename: str) -> "list[Shading]":
    """Return the mesh shadings in a PDF file written by pdfwrite.

    The coordinates are those of the page.
    """
    with open(filename, "rb") as f:
        objects = _pdf_objects(f.read())

    # pdfwrite draws each shading as a shading pattern, whose matrix maps the
    # space of the shading to the page.
    shadings = []
    for dictionary, _ in objects.values():
        if not re.search(rb"/PatternType\s+2\b", dictionary):
            continue
        reference = re.search(rb"/Shading\s+(\d+)\s+0\s+R", dictionary)
        mesh, stream = objects.get(int(reference.group(1)) if reference else -1) or (
            b"",
            None,
        )
        if stream is None:
            raise TestError(f"{filename}: cannot find the shading of {dictionary!r}")
        m = _numbers(dictionary, b"Matrix")
        try:
            shading = _mesh_points(mesh, stream)
        except TestError as e:
            raise TestError(f"{filename}: {e}") from e
        shadings.append(
            Shading(
                shading.kind,
                [
                    (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
                    for x, y in shading.points
                ],
            )
        )
    return shadings


def producer(filename: str) -> str:
    """Return the program that wrote a PDF file, as the file records it."""
    with open(filename, "rb") as f:
        data = f.read()
    # The document information may be inside a compressed object stream.
    chunks = [data]
    for _, stream in _pdf_objects(data).values():
        if stream is not None:
            try:
                chunks.append(zlib.decompress(stream))
            except zlib.error:
                pass
    for chunk in chunks:
        for pattern in (rb"/Producer\s*\(([^)]*)\)", rb"<pdf:Producer>([^<]*)<"):
            match = re.search(pattern, chunk)
            if match:
                return match.group(1).decode("latin-1")
    return "unknown Ghostscript version"


def deviation(source: Shading, stored: Shading) -> float:
    """The largest distance between corresponding points of two shadings,
    once their centroids have been brought together."""
    sx, sy = source.centroid()
    tx, ty = stored.centroid()
    return max(
        max(abs((q[0] - tx) - (p[0] - sx)), abs((q[1] - ty) - (p[1] - sy)))
        for p, q in zip(source.points, stored.points)
    )


def _closest(shading: Shading, candidates: "list[Shading]") -> "Shading | None":
    """Return the candidate of the same structure whose shape is closest."""
    best = None
    for candidate in candidates:
        if candidate.kind != shading.kind:
            continue
        if len(candidate.points) != len(shading.points):
            continue
        if best is None or deviation(shading, candidate) < deviation(shading, best):
            best = candidate
    return best


def check(
    source: "list[Shading]", stored: "list[Shading]", verbose: bool
) -> "list[str]":
    """Compare the shadings written with those stored; return the failures."""
    failures = []
    remaining = list(stored)
    # For each shading its name, its tolerance, and where the PDF file has it
    # relative to the EPS file.
    offsets = []
    for index, shading in enumerate(source):
        name = f"shading {index} (type {shading.kind}, {shading.size():.0f}bp)"
        best = _closest(shading, remaining)
        if best is None:
            failures.append(f"{name}: not found in the PDF file")
            continue
        remaining.remove(best)

        tolerance = max(_TOLERANCE, shading.size() / _RELATIVE_TOLERANCE)
        error = deviation(shading, best)
        if error > tolerance:
            failures.append(
                f"{name}: points are off by up to {error:.4g}bp "
                f"(tolerance {tolerance:.3g}bp)"
            )
        if verbose:
            print(
                f"  {name}: shape off by {error:.2g}bp (tolerance {tolerance:.2g}bp)"
                f" ... {'PASSED' if error <= tolerance else 'FAILED'}"
            )
        offsets.append(
            (
                name,
                tolerance,
                best.centroid()[0] - shading.centroid()[0],
                best.centroid()[1] - shading.centroid()[1],
            )
        )

    if remaining:
        failures.append(f"{len(remaining)} shadings in the PDF file have no source")

    # The shadings must also be in the right place relative to one another:
    # the page origin is the only shift that the two files may differ by.
    for name, tolerance, x, y in offsets[1:]:
        error = max(abs(x - offsets[0][2]), abs(y - offsets[0][3]))
        if error > tolerance + offsets[0][1]:
            failures.append(
                f"{name}: displaced by {error:.4g}bp relative to shading 0 "
                f"(tolerance {tolerance + offsets[0][1]:.3g}bp)"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test the coordinates of mesh shadings in PDF output."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
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
    parser.add_argument(
        "--gs",
        metavar="PATH",
        help="the Ghostscript executable for asy to use (default: asy's own "
        "choice, which honors the environment variable ASYMPTOTE_GS)",
    )
    parser.add_argument(
        "--keep",
        metavar="DIR",
        help="write the test files to DIR and keep them, instead of using a "
        "temporary directory",
    )
    args = parser.parse_args()

    try:
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
            return run(args, os.path.abspath(args.keep))
        with tempfile.TemporaryDirectory() as directory:
            return run(args, directory)
    except TestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace, directory: str) -> int:
    """Run the test with its files in *directory*; return the exit status."""
    source = os.path.join(directory, "meshshading.asy")
    with open(source, "w", encoding="utf-8") as f:
        f.write(_SOURCE)
    prefix = os.path.join(directory, "meshshading")
    written = read_eps(run_asy(args, source, "eps", prefix))
    pdf = run_asy(args, source, "pdf", prefix)
    stored = read_pdf(pdf)

    print(f"Testing mesh shading coordinates (asy = {args.asy}, {producer(pdf)})")
    if not written:
        raise TestError("no shadings found in the EPS output")
    if not stored:
        raise TestError(
            "no shadings found in the PDF output; pdfwrite may lay them out "
            "differently in this version of Ghostscript"
        )

    failures = check(written, stored, args.verbose)
    for failure in failures:
        print(f"  {failure}")
    print(f"{len(written)} shadings compared")
    print("FAILED." if failures else "PASSED.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
