#!/usr/bin/env python3
"""Smoke test for getExecutablePath() (locate.cc) on the host it runs on.

    python3 test_executable_path.py --asy ../asy --asy-base-dir ../base

getExecutablePath() is the one part of sysdir resolution written three times --
GetModuleFileNameA on Windows, _NSGetExecutablePath + realpath on macOS,
/proc/self/exe elsewhere -- so it is the part that can be wrong on a platform
nobody has run the suite on.  It is static, and nothing prints it, so it is
checked here through its only observable: resolveSysdir() offers <exedir>/base
as its first candidate, and reports the result as settings.sysdir.

So: copy asy and a base/ into a fresh temporary directory and ask the copy where
its sysdir is.  Nothing could have compiled that path in and the answer does not
come from the cwd, the environment or an installed Asymptote, so getting it back
means the executable's own directory was computed at run time, and correctly.

That candidate is deliberately the one that is live in *every* build (it is
outside the IS_RELOCATABLE guard, locate.cc:117), so this script needs no
knowledge of how asy was configured -- unlike a full enumeration of the
resolution matrix, which has to be told, or to detect, the build mode.  Run this
one first: a wider sysdir test built on a broken exedir reads it as a wrong
sysdir in every case, and every conclusion it draws is void.

It is written as straight-line code -- no functions, no branches -- so that what
it asserts can be read off in one pass.  The two failure modes are an exception:
a non-zero exit from asy raises CalledProcessError (its stderr is inherited, so
the real complaint appears above the traceback), and a wrong sysdir trips the
assert.  Either one leaves the staged tree behind for inspection; a passing run
removes it.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--asy", required=True, help="path to the asy binary under test")
ap.add_argument(
    "--asy-base-dir",
    required=True,
    dest="base_dir",
    help="the build tree's base/ directory, staged next to the copied binary",
)
args = ap.parse_args()

asy = os.path.abspath(args.asy)
base_dir = os.path.abspath(args.base_dir)

# <work>/bin/asy with <work>/bin/base beside it: the build-tree shape, in a
# directory whose name did not exist when asy was compiled.
work = tempfile.mkdtemp(prefix="asy-exedir-")
bindir = os.path.join(work, "bin")
expected = os.path.join(bindir, "base")
shutil.copytree(base_dir, expected)  # creates bindir on the way
staged_asy = os.path.join(bindir, os.path.basename(asy))
shutil.copy2(asy, staged_asy)

# The cwd is <work>, one level above the staged binary, so a resolver that used
# the working directory instead of the executable's would answer differently.
# stdout is captured (it carries the answer); stderr is left attached to ours.
run = subprocess.run(
    [staged_asy, "-c", "write(settings.sysdir);"],
    stdout=subprocess.PIPE,
    text=True,
    cwd=work,
    timeout=120,
    check=True,
)
resolved = run.stdout.strip()

print(f"staged executable: {staged_asy}")
print(f"expected sysdir:   {expected}")
print(f"resolved sysdir:   {resolved!r}")

# realpath so that a symlinked temporary directory (macOS: /tmp -> /private/tmp)
# does not read as a mismatch, normcase for the case-insensitive filesystems.
assert os.path.normcase(os.path.realpath(resolved)) == os.path.normcase(
    os.path.realpath(expected)
), f"asy resolved sysdir to {resolved!r}, not the staged {expected!r}"

shutil.rmtree(work, ignore_errors=True)
print("PASS: getExecutablePath() found the executable's own directory")
