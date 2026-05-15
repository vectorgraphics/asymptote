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
VULKAN_LIB_DIR="${HOME}/vulkan_sdk/1.4.350.0/macOS/lib"
VULKAN_INCLUDE_DIR="${HOME}/vulkan_sdk/1.4.350.0/macOS/include"
GLFW_LIB_DIR="${HOME}/glfw/build/src"
GLFW_INCLUDE_DIR="${HOME}/glfw/include"

# Prevent pkg-config from discovering MacPorts/Homebrew packages. Their
# libraries carry non-portable install names (e.g. /opt/local/lib/libcurl.4.dylib)
# that the portability check will reject. System libraries are found via the
# linker's default search path without any pkg-config help.
export PKG_CONFIG_LIBDIR=""



# For a portable build:
./configure CC=clang CXX=clang++ \
   CPPFLAGS="-I${VULKAN_INCLUDE_DIR} -I${GLFW_INCLUDE_DIR}" \
   LDFLAGS="-L${VULKAN_LIB_DIR} -L${GLFW_LIB_DIR} -Wl,-rpath,${VULKAN_LIB_DIR} -Wl,-rpath,${GLFW_LIB_DIR}" \
   --disable-lsp \
   --disable-readline \
   --disable-fftw \
   --disable-sigsegv \
   --disable-gsl \
   --disable-curl \
   --disable-xdr \
   --disable-eigen \
   --enable-relocatable \
   --prefix=${HOME}/asy_vulkan/tmp/staging \
   --with-latex=${HOME}/asy_vulkan/tmp/staging/texmf/tex/latex \
   --with-context=${HOME}/asy_vulkan/tmp/staging/texmf/tex/context
  # TODO: build readline from source to get universal binary, then re-enable it