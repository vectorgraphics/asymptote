# copy-build-licenses.cmake
#
# Copies license files into <build-dir>/doc/licenses/ so that a locally built
# asy binary can find them at runtime via the argv[0]-relative search path
# (dirname(asy)/doc/licenses/).
#
# This file is the source of truth for the license file list.
# The install-licenses / uninstall-docdir targets in Makefile.in must be kept
# in sync with the list of files here.

set(ASY_BUILD_LICENSES_DIR ${CMAKE_BINARY_DIR}/doc/licenses)

set(ASY_BUILD_LICENSE_COPIES)

macro(asy_copy_build_license src dest)
    set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    set(_dst ${ASY_BUILD_LICENSES_DIR}/${dest})
    add_custom_command(
            OUTPUT ${_dst}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${ASY_BUILD_LICENSES_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_src} ${_dst}
            DEPENDS ${_src}
            COMMENT "Copying license: ${dest}"
    )
    list(APPEND ASY_BUILD_LICENSE_COPIES ${_dst})
endmacro()

asy_copy_build_license(LICENSE                      LICENSE)
asy_copy_build_license(LICENSE.LESSER               LICENSE.LESSER)
asy_copy_build_license(backports/span/LICENSE.txt   span-LICENSE.txt)
asy_copy_build_license(backports/glew/LICENSE.txt   glew-LICENSE.txt)
asy_copy_build_license(wyhash/UNLICENSE.txt         wyhash-UNLICENSE.txt)
asy_copy_build_license(LspCpp/LICENSE               LspCpp-LICENSE.txt)
asy_copy_build_license(libatomic_ops/LICENSE        libatomic_ops-LICENSE.txt)
asy_copy_build_license(libatomic_ops/COPYING        libatomic_ops-COPYING.txt)
asy_copy_build_license(tinyexr/LICENSE.txt          tinyexr-LICENSE.txt)
asy_copy_build_license(gc/LICENSE.txt               gc-LICENSE.txt)

add_custom_target(asy-licenses DEPENDS ${ASY_BUILD_LICENSE_COPIES})
