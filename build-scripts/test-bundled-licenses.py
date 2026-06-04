#!/usr/bin/env python3
"""Self-contained tests for the bundled-dylib license machinery.

Covers the key-derivation rule and the registry-driven staging performed by
stage-bundled-licenses.py.  No build is required; everything runs against a
throwaway temp tree.  Run from anywhere:

    python3 build-scripts/test-bundled-licenses.py

Exits 0 if every assertion passes, non-zero otherwise.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TSV = os.path.join(SCRIPT_DIR, "bundled-dylib-licenses.tsv")
STAGER = os.path.join(SCRIPT_DIR, "stage-bundled-licenses.py")


def load_stager():
    """Import stage-bundled-licenses.py as a module to reuse dylib_key()."""
    spec = importlib.util.spec_from_file_location("stage_bundled_licenses",
                                                  STAGER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Checker:
    def __init__(self):
        self.failed = 0

    def check(self, desc, expected, actual):
        if expected == actual:
            print("ok   - " + desc)
        else:
            print("FAIL - {} (expected {!r}, got {!r})".format(
                desc, expected, actual))
            self.failed += 1

    def ok(self, cond, desc):
        if cond:
            print("ok   - " + desc)
        else:
            print("FAIL - " + desc)
            self.failed += 1


def run_stager(libdir, destdir, tsv):
    """Run the stager as a subprocess; return (returncode, combined output)."""
    proc = subprocess.run([sys.executable, STAGER, libdir, destdir, tsv],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def make_libs(libdir, names):
    os.makedirs(libdir, exist_ok=True)
    for n in names:
        open(os.path.join(libdir, n), "w").close()


def main():
    c = Checker()
    stager = load_stager()

    # --- Part A: key-derivation rule.
    cases = [
        ("libMoltenVK.dylib", "moltenvk"),
        ("libvulkan.1.4.350.dylib", "vulkan"),
        ("libSPIRV.15.dylib", "spirv"),
        ("libglfw.3.dylib", "glfw"),
        ("libglslang.15.dylib", "glslang"),
        ("libMachineIndependent.dylib", "machineindependent"),
    ]
    for basename, expected in cases:
        c.check("{} -> {}".format(basename, expected),
                expected, stager.dylib_key(basename))

    with tempfile.TemporaryDirectory(prefix="bundled-lic-test.") as work:
        # --- Part B: end-to-end staging into a temp tree.
        libdir = os.path.join(work, "lib")
        destdir = os.path.join(work, "doc", "licenses")
        os.makedirs(destdir, exist_ok=True)
        make_libs(libdir, [
            "libglfw.3.dylib", "libvulkan.1.4.350.dylib", "libMoltenVK.dylib",
            "libglslang.15.dylib", "libSPIRV.15.dylib",
            "libsomethingexperimental.dylib",
        ])

        rc, out = run_stager(libdir, destdir, TSV)
        c.ok(rc == 0, "staging a realistic bundled set succeeds")
        for want in ("glfw-LICENSE.md", "vulkan-LICENSE.txt",
                     "moltenvk-LICENSE.txt", "glslang-LICENSE.txt"):
            c.ok(os.path.isfile(os.path.join(destdir, want)),
                 "staged " + want)
        # The unregistered dylib must produce a non-fatal warning naming its key.
        c.ok("somethingexperimental" in out,
             "warned about unregistered dylib")

        # Staging must be idempotent.
        rc2, _ = run_stager(libdir, destdir, TSV)
        c.ok(rc2 == 0, "staging is idempotent")

        # --- Part C: a matched key whose vendored source is missing is fatal.
        lib2 = os.path.join(work, "lib2")
        make_libs(lib2, ["libvulkan.1.dylib"])
        bad_tsv = os.path.join(work, "bad.tsv")
        with open(bad_tsv, "w") as f:
            f.write("\t".join([
                "vulkan", "Vulkan-Loader", "Apache-2.0", "libvulkan*.dylib",
                "licenses/bundled/does-not-exist.txt", "vulkan-LICENSE.txt",
                "https://example.invalid",
            ]) + "\n")
        rc3, _ = run_stager(lib2, destdir, bad_tsv)
        c.ok(rc3 != 0, "missing vendored source is fatal")

    print()
    if c.failed == 0:
        print("All bundled-license tests passed.")
        return 0
    print("Some bundled-license tests FAILED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
