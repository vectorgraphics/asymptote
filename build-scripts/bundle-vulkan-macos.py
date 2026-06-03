#!/usr/bin/env python3
"""bundle-vulkan-macos.py  <binary-name> <codesign-identity>

Bundles Vulkan-related dylibs next to the named binary so the result is
self-contained and portable.  Run from the build directory (the directory
that contains the binary).

Called by the top-level Makefile when BUNDLE_VULKAN=yes.
"""

import os
import re
import shutil
import stat
import subprocess
import sys


def die(*lines):
    for line in lines:
        print(line, file=sys.stderr)
    sys.exit(1)


def run(cmd):
    """Run a command, returning its stdout (text).  Never raises on a non-zero
    exit; returns whatever was produced on stdout."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout


def run_quiet(cmd):
    """Run a command, ignoring its exit status and output (mirrors the shell
    '... 2>/dev/null || true' idiom)."""
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def baked_rpaths(binary):
    """Collect the rpaths baked into the given binary via LC_RPATH commands."""
    rpaths = set()
    pending = False
    for line in run(["otool", "-l", binary]).splitlines():
        if "LC_RPATH" in line:
            pending = True
        elif pending and " path " in line:
            rpaths.add(line.split()[1])
            pending = False
    return sorted(rpaths)


def library_refs(binaries, pattern):
    """Collect first-column otool -L references across BINARIES whose path
    matches PATTERN (a compiled regex)."""
    refs = set()
    for line in run(["otool", "-L"] + binaries).splitlines():
        fields = line.split()
        if fields and pattern.match(fields[0]):
            refs.add(fields[0])
    return refs


def main(argv):
    if len(argv) < 2:
        die("Usage: {} <binary-name> <codesign-identity>".format(argv[0]))
    name = argv[1]
    codesign_identity = argv[2] if len(argv) > 2 and argv[2] else "-"

    print("Bundling Vulkan libraries for macOS...")
    os.makedirs("lib", exist_ok=True)

    # Collect the set of binaries to patch (main binary + any shim libs present).
    all_bins = [name]
    for shim in ("libasyvulkan.so", "libasyopengl.so"):
        if os.path.isfile(shim):
            all_bins.append(shim)

    # Collect the rpaths baked into the main binary.
    rpaths = baked_rpaths(name)

    # Collect @rpath-relative and absolute Vulkan/GLFW library references.
    rpath_re = re.compile(r"@rpath/lib(glfw|vulkan|SPIRV|glslang)")
    abs_re = re.compile(r"/.*/lib(glfw|vulkan|SPIRV|glslang)")
    rpath_refs = library_refs(all_bins, rpath_re)
    abs_refs = library_refs(all_bins, abs_re)

    # Resolve @rpath references to absolute paths via the recorded rpaths.
    vulkan_libs = set(abs_refs)
    for ref in rpath_refs:
        libname = ref[len("@rpath/"):]
        for rpath in rpaths:
            candidate = os.path.join(rpath, libname)
            if os.path.isfile(candidate):
                vulkan_libs.add(candidate)
                break

    vulkan_libs = sorted(p for p in vulkan_libs if p)

    if not vulkan_libs:
        die("ERROR: Could not locate Vulkan-related dynamic libraries "
            "referenced by {}.".format(name),
            "Fix: ensure Vulkan/GLFW are installed and linked, then run "
            "'make clean && make asy'.")

    moltenvk_dir = None
    for lib in vulkan_libs:
        if "libvulkan" in os.path.basename(lib):
            moltenvk_dir = os.path.dirname(lib)
            break

    # Copy each library and retarget references in all binaries.
    for lib in vulkan_libs:
        libname = os.path.basename(lib)
        dest = os.path.join("lib", libname)
        shutil.copyfile(lib, dest)
        mode = os.stat(dest).st_mode
        os.chmod(dest, mode | stat.S_IWUSR)
        for binary in all_bins:
            run_quiet(["install_name_tool", "-change", "@rpath/" + libname,
                       "@executable_path/lib/" + libname, binary])
            run_quiet(["install_name_tool", "-change", lib,
                       "@executable_path/lib/" + libname, binary])

    # Copy MoltenVK and generate the ICD JSON.
    moltenvk_src = (os.path.join(moltenvk_dir, "libMoltenVK.dylib")
                    if moltenvk_dir else None)
    if moltenvk_src and os.path.isfile(moltenvk_src):
        dest = os.path.join("lib", "libMoltenVK.dylib")
        shutil.copyfile(moltenvk_src, dest)
        mode = os.stat(dest).st_mode
        os.chmod(dest, mode | stat.S_IWUSR)
        sys_icd = os.path.join(os.path.dirname(moltenvk_dir),
                               "share", "vulkan", "icd.d", "MoltenVK_icd.json")
        icd_path = os.path.join("lib", "MoltenVK_icd.json")
        if os.path.isfile(sys_icd):
            with open(sys_icd, encoding="utf-8") as f:
                contents = f.read()
            contents = re.sub(
                r'("library_path"\s*:\s*)("[^"]*")(,?)',
                r'\1"./libMoltenVK.dylib"\3', contents)
            with open(icd_path, "w", encoding="utf-8") as f:
                f.write(contents)
        else:
            with open(icd_path, "w", encoding="utf-8") as f:
                f.write('{\n'
                        '    "file_format_version": "1.0.0",\n'
                        '    "ICD": {\n'
                        '        "library_path": "./libMoltenVK.dylib",\n'
                        '        "api_version": "1.0.0"\n'
                        '    }\n'
                        '}\n')
    else:
        die("ERROR: libMoltenVK.dylib was not found next to libvulkan.",
            "Fix: ensure Vulkan SDK with MoltenVK is installed, then rebuild.")

    # Fix the install names of the bundled libraries themselves.
    for bundled in sorted(p for p in os.listdir("lib") if p.endswith(".dylib")):
        bundled_path = os.path.join("lib", bundled)
        run_quiet(["install_name_tool", "-id",
                   "@executable_path/lib/" + bundled, bundled_path])
        for lib in vulkan_libs:
            libname = os.path.basename(lib)
            run_quiet(["install_name_tool", "-change", "@rpath/" + libname,
                       "@executable_path/lib/" + libname, bundled_path])
            run_quiet(["install_name_tool", "-change", lib,
                       "@executable_path/lib/" + libname, bundled_path])

    # Re-sign everything.
    for bundled in sorted(p for p in os.listdir("lib") if p.endswith(".dylib")):
        run_quiet(["codesign", "--sign", codesign_identity, "--force",
                   os.path.join("lib", bundled)])
    run_quiet(["codesign", "--sign", codesign_identity, "--force", name])

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))