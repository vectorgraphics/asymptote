# Directories for libraries that must be universal (x86_64 + arm64) binaries.
# These are used for both compile-time detection and runtime bundling.
# The Vulkan SDK and GLFW builds here must themselves be universal; if they
# are not, the post-build portability check will report a clear error.

# -----------------------------------------------------------------------------
# Building GLFW from source: After cloning the repository and navigating to the
# directory, run the following commands to build a universal binary:

# mkdir build && cd build
#
# cmake .. \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DBUILD_SHARED_LIBS=ON \
#   -DGLFW_BUILD_EXAMPLES=OFF \
#   -DGLFW_BUILD_TESTS=OFF \
#   -DGLFW_BUILD_DOCS=OFF \
#   -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
#   -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15
#
# make -j$(sysctl -n hw.logicalcpu)
#------------------------------------------------------------------------------

# NOTE: Set these paths for your own system! The Vulkan SDK can be obtained
# from https://vulkan.lunarg.com/sdk/home. GLFW can be built from source
# using the instructions at
# https://www.glfw.org/docs/latest/compile_guide.html#compile_macos, but make
# sure to build a universal binary (see instructions above).
# If you have multiple versions of the Vulkan SDK installed, make sure to use
# the same version for both the include and lib paths.
# NOTE: This is usually in a directory called "VulkanSDK", not "vulkan_sdk".
VULKAN_LIB_DIR="${HOME}/vulkan_sdk/1.4.350.0/macOS/lib"
VULKAN_INCLUDE_DIR="${HOME}/vulkan_sdk/1.4.350.0/macOS/include"
GLFW_LIB_DIR="${HOME}/glfw/build/src"
GLFW_INCLUDE_DIR="${HOME}/glfw/include"

# Prevent pkg-config from discovering MacPorts/Homebrew packages. Their
# libraries carry non-portable install names (e.g. /opt/local/lib/libcurl.4.dylib)
# that the portability check will reject. System libraries are found via the
# linker's default search path without any pkg-config help.
export PKG_CONFIG_LIBDIR=""



# Staging tree for the bundle.  Wipe it before installing: the configured prefix
# is baked into the binary as ASYMPTOTE_SYSDIR, so a stale $STAGE/share/asymptote
# left by an older build would be a base/ that resolves on this machine and
# nowhere else.  A freshly-staged bundle never creates that directory.
STAGE=${HOME}/asy_vulkan/tmp/staging

# The bundle is relocatable by its layout, not by a build flag: asy prefers a
# base/ sitting beside the executable over the compiled-in ASYMPTOTE_SYSDIR
# (resolveSysdir(), locate.cc), so installing the two side by side makes the
# bundle work wherever the user drags it out of the .dmg.  After building:
#
#   rm -rf $STAGE
#   make install-asy bindir=$STAGE/Asymptote asydir=$STAGE/Asymptote/base
#
# which gives $STAGE/Asymptote/{asy,base/,lib/} -- lib/ being the bundled
# dylibs, which asy finds through @executable_path/lib.  $STAGE/Asymptote is
# what goes into the .dmg.

# For a portable build:
./configure CC=clang CXX=clang++ \
   CPPFLAGS="-I${VULKAN_INCLUDE_DIR} -I${GLFW_INCLUDE_DIR}" \
   LDFLAGS="-L${VULKAN_LIB_DIR} -L${GLFW_LIB_DIR} -Wl,-rpath,${VULKAN_LIB_DIR} -Wl,-rpath,${GLFW_LIB_DIR}" \
   --enable-macos-universal \
   --enable-macos-bundling \
   --disable-lsp \
   --disable-readline \
   --disable-fftw \
   --disable-sigsegv \
   --disable-gsl \
   --disable-curl \
   --disable-xdr \
   --disable-eigen \
   --prefix=${STAGE} \
   --with-latex=${STAGE}/texmf/tex/latex \
   --with-context=${STAGE}/texmf/tex/context
  # TODO: build readline from source to get universal binary, then re-enable it
