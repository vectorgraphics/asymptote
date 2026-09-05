#!/usr/bin/env python3
"""Smoke test for executablePath() (locate.cc) on the host it runs on.

    python3 test_executable_path.py --asy ../asy --asy-base-dir ../base

executablePath() is the one part of sysdir resolution written once per OS --
GetModuleFileNameW + canonical() on Windows, _NSGetExecutablePath + realpath on
macOS, the kern.proc.pathname sysctl on FreeBSD, /proc/self/exe elsewhere -- so
it is the part that can be wrong on a platform nobody has run the suite on.  Not
hypothetically: the FreeBSD branch exists because /proc is not mounted there, so
the "elsewhere" case returned nothing at all.  Nothing prints the value, so it
is checked through its only observable: resolveSysdir() takes <exedir>/base as
its one candidate and reports the result as settings.sysdir.

So: copy asy and a base/ into a fresh temporary directory and ask the copy where
its sysdir is.  Nothing could have compiled that path in, the cwd is elsewhere,
and the probe's environment is stripped of the variables that would answer for
it, so getting it back means the executable's own directory was computed at run
time, and correctly.

Run this before any wider sysdir test, which a broken exedir would invalidate
throughout.  It is straight-line code so that what it asserts reads off in one
pass; ``-v`` adds the three paths the assertion is made of.  A failing run
leaves the staged tree behind for inspection.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# tests-asy.cmake passes these two flags to both scripts the same way, so they
# are spelled as test_relocatable.py spells them; pylint sees duplicate code.
# pylint: disable=duplicate-code
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
# pylint: enable=duplicate-code
ap.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="print the staged, expected and resolved paths, not just the verdict",
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

# The shared libraries a deployment bundles beside the executable travel with
# it, or the staged copy is not the shape under test -- and on Windows does not
# reach main() at all (0xC0000135).  None of them is plain.asy, so none can
# change which directory resolveSysdir() picks.  macOS needs both suffixes,
# since dyld loads either; everything else takes the ELF .so default, FreeBSD
# included (its sys.platform carries the major version, so it could not be a key
# here anyway).
suffixes = {"win32": (".dll",), "darwin": (".dylib", ".so")}.get(sys.platform, (".so",))
srcdir = os.path.dirname(asy)
for name in os.listdir(srcdir):
    src = os.path.join(srcdir, name)
    if (name.lower().endswith(suffixes) or ".so." in name.lower()) and os.path.isfile(
        src
    ):
        shutil.copy2(src, os.path.join(bindir, name))

# A macOS bundle collects them into lib/ beside the binary instead, with the
# references rewritten to @executable_path/lib/, so that directory travels
# whole (MoltenVK_icd.json included) or the loader aborts before main().
libdir = os.path.join(srcdir, "lib")
if os.path.isdir(libdir):
    shutil.copytree(libdir, os.path.join(bindir, "lib"))

# The answer has to come from executablePath(), so the probe does not inherit
# the variables that would answer for it.  ASYMPTOTE_SYSDIR is an envSetting
# (settings.cc:1933) and replaces the resolved value outright; ASYMPTOTE_DIR is
# the same hazard one step removed.  Both are dropped from a copy of os.environ,
# never from os.environ itself.  ASYMPTOTE_HOME is redirected rather than
# dropped: it cannot reach sysdir, but it names the directory config.asy comes
# from, and unset it would fall back to the likelier $HOME/.asy.
child_env = {
    k: v
    for k, v in os.environ.items()
    if k not in ("ASYMPTOTE_SYSDIR", "ASYMPTOTE_DIR")
}
child_env["ASYMPTOTE_HOME"] = work

# The cwd is <work>, one level above the staged binary, so a resolver that used
# the working directory instead of the executable's would answer differently.
# stdout is captured (it carries the answer); stderr is left attached to ours.
run = subprocess.run(
    [staged_asy, "-c", "write(settings.sysdir);"],
    stdout=subprocess.PIPE,
    text=True,
    cwd=work,
    env=child_env,
    timeout=120,
    check=True,
)
resolved = run.stdout.strip()

if args.verbose:
    print(f"staged executable: {staged_asy}")
    print(f"expected sysdir:   {expected}")
    print(f"resolved sysdir:   {resolved!r}")

# realpath so that a symlinked temporary directory (macOS: /tmp -> /private/tmp)
# does not read as a mismatch, normcase for the case-insensitive filesystems.
assert os.path.normcase(os.path.realpath(resolved)) == os.path.normcase(
    os.path.realpath(expected)
), (
    f"asy resolved sysdir to {resolved!r}, not the staged {expected!r} "
    f"(staged executable: {staged_asy})"
)

shutil.rmtree(work, ignore_errors=True)
print("PASS: executablePath() found the executable's own directory")
