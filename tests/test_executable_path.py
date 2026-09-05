#!/usr/bin/env python3
"""Smoke test for executablePath() (locate.cc) on the host it runs on.

    python3 test_executable_path.py --asy ../asy --asy-base-dir ../base

executablePath() is the one part of sysdir resolution written once per OS --
GetModuleFileNameW + canonical() on Windows, _NSGetExecutablePath + realpath on
macOS, the kern.proc.pathname sysctl on FreeBSD, /proc/self/exe elsewhere -- so
it is the part that can be wrong on a platform nobody has run the suite on.
That is not hypothetical: the FreeBSD branch exists because /proc is not mounted
on a stock FreeBSD, where the "elsewhere" case therefore returned nothing at
all.  Nothing prints the value, so it is checked here through its only
observable: resolveSysdir() takes <exedir>/base as its one candidate, and
reports the result as settings.sysdir.

So: copy asy and a base/ into a fresh temporary directory and ask the copy where
its sysdir is.  Nothing could have compiled that path in, the cwd is elsewhere,
and the probe runs in an environment stripped of the variables that would answer
for it, so getting it back means the executable's own directory was computed at
run time, and correctly.

Run this one first: a wider sysdir test built on a broken exedir reads it as a
wrong sysdir in every case, and every conclusion it draws is void.

It is written as straight-line code -- no functions, very little branching -- so
that what it asserts can be read off in one pass.  A passing run says so in a
line; ``-v`` adds the three paths the assertion is made of, which are what you
want when it is the staging rather than the result that is in question.  The two
failure modes are an exception: a non-zero exit from asy raises
CalledProcessError (its stderr is inherited, so the real complaint appears above
the traceback), and a wrong sysdir trips the assert.  Either one leaves the
staged tree behind for inspection; a passing run removes it.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# The two flags below are spelled exactly as test_relocatable.py spells them,
# because tests-asy.cmake passes them to both scripts the same way -- eight
# identical lines, which pylint reads as duplicate code.
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
# it: .dll on Windows (what the NSIS install ships next to asy.exe), .dylib or
# .so in a macOS bundle, .so under an $ORIGIN rpath.  Carrying its own runtime
# this way is part of what relocatable means, so the staged copy is only the
# shape under test if they come along -- and on Windows a copy without them does
# not reach main() at all (STATUS_DLL_NOT_FOUND, 0xC0000135).  None of them is
# plain.asy, so none can change which directory resolveSysdir() picks.
#
# macOS needs both suffixes: dyld loads either, and ports of libraries whose
# build systems assume ELF (and Python extension modules) install .so there.
#
# Everything else takes the .so default, which is what the ELF platforms want.
# That includes FreeBSD, whose sys.platform carries the major version
# ("freebsd14"), so it could not be a key here even if it needed a different
# answer.
suffixes = {"win32": (".dll",), "darwin": (".dylib", ".so")}.get(sys.platform, (".so",))
srcdir = os.path.dirname(asy)
for name in os.listdir(srcdir):
    src = os.path.join(srcdir, name)
    if (name.lower().endswith(suffixes) or ".so." in name.lower()) and os.path.isfile(
        src
    ):
        shutil.copy2(src, os.path.join(bindir, name))

# A macOS bundle does not lay them flat: --enable-macos-bundling collects them
# into lib/ beside the binary and rewrites the references to @executable_path/
# lib/, so that whole directory has to travel too -- the loader aborts before
# main() without it.  It carries MoltenVK_icd.json along with the libraries,
# which is the shape the bundle is meant to have.
libdir = os.path.join(srcdir, "lib")
if os.path.isdir(libdir):
    shutil.copytree(libdir, os.path.join(bindir, "lib"))

# The answer has to come from executablePath(), so the probe does not inherit
# the variables that would answer for it.  ASYMPTOTE_SYSDIR is an envSetting
# (settings.cc:1933): it replaces the resolved value outright, so a developer
# or CI shell that exports it -- at a working base/, which is how anyone who
# exports it sets it -- fails this test without any regression to find.
# ASYMPTOTE_DIR is the same hazard one step removed.  Both are dropped from a
# copy of os.environ, never from os.environ itself, so this process's own
# environment is left as the caller wrote it.
#
# ASYMPTOTE_HOME cannot reach sysdir -- setOptions() restores the value it read
# before the configuration file (settings.cc:2325) -- but it names the
# directory config.asy is read from, and whatever that file writes arrives in
# the captured stdout beside the answer.  Unset, it falls back to $HOME/.asy
# (initDir()), which is a config file the caller did not write either, so it is
# pointed at the staging tree rather than dropped.
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
# The message repeats all three paths rather than relying on the block above,
# which a quiet run did not print.
assert os.path.normcase(os.path.realpath(resolved)) == os.path.normcase(
    os.path.realpath(expected)
), (
    f"asy resolved sysdir to {resolved!r}, not the staged {expected!r} "
    f"(staged executable: {staged_asy})"
)

shutil.rmtree(work, ignore_errors=True)
print("PASS: executablePath() found the executable's own directory")
