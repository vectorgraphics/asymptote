#!/bin/sh
# Usage: portability-check-macos.sh <name> <min_version> <universal> <bundle_vulkan>
# Checks macOS portability: architectures, minimum OS version, library references.
set -eu

NAME="$1"
MACOS_MIN_VERSION="$2"
MACOS_UNIVERSAL="$3"
BUNDLE_VULKAN="$4"

if [ "$(uname -s)" != "Darwin" ]; then
	echo "Skipping portability checks: host is not macOS."
	exit 0
fi

if ! command -v lipo >/dev/null 2>&1 || ! command -v otool >/dev/null 2>&1; then
	echo "ERROR: lipo and/or otool not found."
	echo "Fix: install Apple command line tools with: xcode-select --install"
	exit 1
fi

min_required="$MACOS_MIN_VERSION"
if [ -z "$min_required" ]; then
	min_required="12.0"
fi

to_int() {
	echo "$1" | awk -F. '{a=$1+0; b=($2==""?0:$2+0); c=($3==""?0:$3+0); printf("%d%03d%03d\n",a,b,c)}'
}

check_binary() {
	f="$1"
	if [ ! -f "$f" ]; then
		echo "ERROR: Required output '$f' was not found."
		echo "Fix: build it first, then re-run: make asy"
		exit 1
	fi
	archs="$(lipo -archs "$f" 2>/dev/null || true)"
	if [ "$MACOS_UNIVERSAL" = "yes" ]; then
		for required in x86_64 arm64; do
			if ! echo " $archs " | grep -q " $required "; then
				echo "ERROR: $f is missing architecture '$required' (found: $archs)."
				echo "Fix: ensure all dependencies are universal and configure with --enable-macos-universal (default), then rebuild from clean."
				exit 1
			fi
		done
	fi
	minos="$(otool -l "$f" | awk 'BEGIN{want=0; legacy=0} $1=="cmd" && $2=="LC_BUILD_VERSION" {want=1; next} want && $1=="minos" {print $2; exit} $1=="cmd" && $2=="LC_VERSION_MIN_MACOSX" {legacy=1; next} legacy && $1=="version" {print $2; exit}')"
	if [ -z "$minos" ]; then
		echo "ERROR: Could not determine minimum macOS version for $f."
		echo "Fix: build with Apple's linker tools and retry."
		exit 1
	fi
	if [ "$(to_int "$minos")" -gt "$(to_int "$min_required")" ]; then
		echo "ERROR: $f requires macOS $minos, which is newer than target $min_required."
		echo "Fix: reconfigure with --with-macos-min-version=$min_required (or older), then run 'make clean && make asy'."
		exit 1
	fi
	own_id="$(otool -D "$f" 2>/dev/null | awk 'NR==2{print $1}' || true)"
	bad_refs="$(otool -L "$f" | awk -v own="$own_id" '/^\t/{if (own != "" && $1 == own) next; print $1}' | \
		grep -Ev '^(@executable_path/lib/|@loader_path/|/usr/lib/|/System/Library/)' || true)"
	if [ -n "$bad_refs" ]; then
		echo "ERROR: $f contains non-portable library references:"
		echo "$bad_refs"
		echo "Every linked library must be one of:"
		echo "  - a macOS system library  (/usr/lib/ or /System/Library/)"
		echo "  - a bundled dylib          (@executable_path/lib/)"
		echo "  - a co-located dylib       (@loader_path/)"
		echo "Fix options:"
		echo "  - @rpath refs: ensure the bundle-vulkan step rewrote all Vulkan/GLFW"
		echo "    references to @executable_path/lib/, then rebuild."
		echo "  - /opt/homebrew, /opt/local, /usr/local, /Users/ absolute paths:"
		echo "    link against system SDK versions, or build the library as a universal"
		echo "    dylib, copy it into ./lib/, and add it to SHIMLIBS in Makefile.in."
		exit 1
	fi
}

files="$NAME libasyvulkan.so libasyopengl.so"
if [ "$BUNDLE_VULKAN" = "yes" ]; then
	if ls lib/*.dylib >/dev/null 2>&1; then
		for f in lib/*.dylib; do files="$files $f"; done
	else
		echo "ERROR: BUNDLE_VULKAN=yes but no bundled dylibs were found in ./lib."
		echo "Fix: verify Vulkan SDK/MoltenVK installation and rerun 'make clean && make asy'."
		exit 1
	fi
fi

for f in $files; do
	check_binary "$f"
done

echo "macOS portability checks passed for $NAME, renderer shims, and bundled dylibs."
