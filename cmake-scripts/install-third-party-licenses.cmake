# install-third-party-licenses.cmake
#
# Install rules for all Asymptote license files (LICENSE, LICENSE.LESSER,
# and all third-party licenses).  Installs from the already-renamed copies
# in the build tree produced by copy-build-licenses.cmake, which is the
# single source of truth for the file list and must be included before this
# file (CMakeLists.txt guarantees this).
#
# Both linux-install.cmake and win32-pre-nsis-installer.cmake include this
# file after setting ASY_LICENSE_INSTALL_ARGS to the platform-specific
# install() arguments (DESTINATION, COMPONENT, PERMISSIONS, etc.).
#
# To add a new third-party license file, add a single asy_copy_build_license()
# call in copy-build-licenses.cmake; both build and install will pick it up
# automatically.

if (NOT DEFINED ASY_LICENSE_INSTALL_ARGS)
    message(FATAL_ERROR
        "ASY_LICENSE_INSTALL_ARGS must be set before including "
        "install-third-party-licenses.cmake")
endif()

if (NOT DEFINED ASY_BUILD_LICENSE_COPIES)
    message(FATAL_ERROR
        "copy-build-licenses.cmake must be included before "
        "install-third-party-licenses.cmake")
endif()

install(
        FILES ${ASY_BUILD_LICENSE_COPIES}
        ${ASY_LICENSE_INSTALL_ARGS}
)
