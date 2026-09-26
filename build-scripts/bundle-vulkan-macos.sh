#!/bin/sh
# bundle-vulkan-macos.sh  <binary-name> <codesign-identity>
#
# Bundles Vulkan-related dylibs next to the named binary so the result is
# self-contained and portable.  Run from the build directory (the directory
# that contains the binary).
#
# Called by the top-level Makefile when BUNDLE_VULKAN=yes.

set -eu

NAME="${1:?Usage: $0 <binary-name> <codesign-identity>}"
CODESIGN_IDENTITY="${2:--}"

echo "Bundling Vulkan libraries for macOS..."
mkdir -p lib

# Collect the set of binaries to patch (main binary + any shim libs present).
all_bins="$NAME"
for s in libasyvulkan.so libasyopengl.so; do
    [ -f "$s" ] && all_bins="$all_bins $s" || true
done

# Collect the rpaths baked into the main binary.
rpaths=$(otool -l "$NAME" \
    | awk '/LC_RPATH/{f=1} f && /[[:space:]]path /{print $2; f=0}' \
    | sort -u)

# Collect @rpath-relative and absolute Vulkan/GLFW library references.
rpath_refs=$(otool -L $all_bins 2>/dev/null \
    | awk '{print $1}' \
    | grep -E '^@rpath/lib(glfw|vulkan|SPIRV|glslang)' \
    | sort -u || true)

abs_refs=$(otool -L $all_bins 2>/dev/null \
    | awk '{print $1}' \
    | grep -E '^/.*/lib(glfw|vulkan|SPIRV|glslang)' \
    | sort -u || true)

# Resolve @rpath references to absolute paths via the recorded rpaths.
vulkan_libs="$abs_refs"
for ref in $rpath_refs; do
    libname="${ref#@rpath/}"
    for rpath in $rpaths; do
        if [ -f "$rpath/$libname" ]; then
            vulkan_libs="$vulkan_libs $rpath/$libname"
            break
        fi
    done
done

vulkan_libs=$(printf '%s\n' $vulkan_libs | sort -u | grep -v '^$' || true)

if [ -z "$vulkan_libs" ]; then
    echo "ERROR: Could not locate Vulkan-related dynamic libraries referenced by $NAME."
    echo "Fix: ensure Vulkan/GLFW are installed and linked, then run 'make clean && make asy'."
    exit 1
fi

moltenvk_dir=$(echo "$vulkan_libs" | grep libvulkan | head -1 | xargs dirname 2>/dev/null || true)

# Copy each library and retarget references in all binaries.
for lib in $vulkan_libs; do
    libname=$(basename "$lib")
    cp "$lib" lib/"$libname"
    chmod u+w lib/"$libname"
    for bin in $all_bins; do
        install_name_tool -change "@rpath/$libname" "@executable_path/lib/$libname" "$bin" 2>/dev/null || true
        install_name_tool -change "$lib" "@executable_path/lib/$libname" "$bin" 2>/dev/null || true
    done
done

# Copy MoltenVK and generate the ICD JSON.
if [ -n "$moltenvk_dir" ] && [ -f "$moltenvk_dir/libMoltenVK.dylib" ]; then
    cp "$moltenvk_dir/libMoltenVK.dylib" lib/
    chmod u+w lib/libMoltenVK.dylib
    icd_prefix=$(cd "$moltenvk_dir/.." 2>/dev/null && pwd)
    sys_icd="$icd_prefix/share/vulkan/icd.d/MoltenVK_icd.json"
    if [ -f "$sys_icd" ]; then
        sed -E 's|("library_path"[[:space:]]*:[[:space:]]*)("[^"]*")(,?)|\1"./libMoltenVK.dylib"\3|' \
            "$sys_icd" > lib/MoltenVK_icd.json
    else
        printf '{\n    "file_format_version": "1.0.0",\n    "ICD": {\n        "library_path": "./libMoltenVK.dylib",\n        "api_version": "1.0.0"\n    }\n}\n' \
            > lib/MoltenVK_icd.json
    fi
else
    echo "ERROR: libMoltenVK.dylib was not found next to libvulkan."
    echo "Fix: ensure Vulkan SDK with MoltenVK is installed, then rebuild."
    exit 1
fi

# Fix the install names of the bundled libraries themselves.
for bundled in lib/*.dylib; do
    bname=$(basename "$bundled")
    install_name_tool -id "@executable_path/lib/$bname" "$bundled" 2>/dev/null || true
    for lib in $vulkan_libs; do
        libname=$(basename "$lib")
        install_name_tool -change "@rpath/$libname" "@executable_path/lib/$libname" "$bundled" 2>/dev/null || true
        install_name_tool -change "$lib" "@executable_path/lib/$libname" "$bundled" 2>/dev/null || true
    done
done

# Every reference now points at @executable_path/lib/, so nothing consults
# LC_RPATH any more.  Strip the absolute rpaths the linker recorded -- the
# Vulkan SDK and GLFW build directories, in the *builder's* home -- from
# everything that ships: they leak the builder's username and layout to every
# user of the .dmg, and they are a search path that would come back to life if
# a later change reintroduced an unbundled @rpath reference.  Makefile.in's
# portability check does not catch them; it reads otool -L, not otool -l.
# @executable_path/ and @loader_path/ rpaths relocate with the bundle and are
# kept.
shipped_bins="$all_bins"
for bundled in lib/*.dylib; do
    shipped_bins="$shipped_bins $bundled"
done

# Refuse to strip while something still needs them: dyld resolves @rpath/ only
# against LC_RPATH, so removing them out from under a surviving reference yields
# a binary that cannot start.  Makefile.in rejects such a reference too, but
# only after this script has run.
stray_refs=$(otool -L $shipped_bins 2>/dev/null \
    | awk '{print $1}' \
    | grep '^@rpath/' \
    | sort -u || true)
if [ -n "$stray_refs" ]; then
    echo "ERROR: these @rpath references were not rewritten to @executable_path/lib/:"
    echo "$stray_refs"
    echo "Fix: extend the lib(glfw|vulkan|SPIRV|glslang) pattern near the top of"
    echo "this script so the library is bundled, or link against a system library."
    exit 1
fi

for bin in $shipped_bins; do
    # One call per LC_RPATH: install_name_tool removes a single load command at
    # a time, and a universal binary reports the same path once per slice.
    while :; do
        rpath=$(otool -l "$bin" \
            | awk '/LC_RPATH/{f=1} f && /[[:space:]]path /{print $2; f=0}' \
            | grep -v '^@' \
            | head -1)
        [ -n "$rpath" ] || break
        if ! install_name_tool -delete_rpath "$rpath" "$bin"; then
            echo "ERROR: could not remove the rpath $rpath from $bin."
            echo "Fix options:"
            echo "  - check that $bin is writable and was not already signed;"
            echo "  - a path containing a space arrives here truncated at the"
            echo "    space (this script reads otool output field by field, as"
            echo "    it does for the rpaths it searches above), so rebuild with"
            echo "    the Vulkan SDK and GLFW at paths without spaces."
            exit 1
        fi
    done
done

# Re-sign everything.  This must stay last: the rewrites and the rpath removal
# above both invalidate any existing signature.
for bundled in lib/*.dylib; do
    codesign --sign "$CODESIGN_IDENTITY" --force "$bundled" 2>/dev/null || true
done
codesign --sign "$CODESIGN_IDENTITY" --force "$NAME" 2>/dev/null || true
