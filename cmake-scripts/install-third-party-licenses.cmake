# install-third-party-licenses.cmake
#
# Shared install rules for third-party license files.  Both
# linux-install.cmake and win32-pre-nsis-installer.cmake include this
# file after setting ASY_LICENSE_INSTALL_ARGS to the platform-specific
# install() arguments (DESTINATION, COMPONENT, PERMISSIONS, etc.).
#
# To add a new third-party license file, add a single call here; both
# platforms will pick it up automatically.

if (NOT DEFINED ASY_LICENSE_INSTALL_ARGS)
    message(FATAL_ERROR
        "ASY_LICENSE_INSTALL_ARGS must be set before including "
        "install-third-party-licenses.cmake")
endif()

macro(asy_install_third_party_license source_file rename_to)
    install(
            FILES ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
            RENAME ${rename_to}
            ${ASY_LICENSE_INSTALL_ARGS}
    )
endmacro()

asy_install_third_party_license(backports/span/LICENSE.txt    span-LICENSE.txt)
asy_install_third_party_license(backports/glew/LICENSE.txt    glew-LICENSE.txt)
asy_install_third_party_license(wyhash/UNLICENSE.txt          wyhash-UNLICENSE.txt)
asy_install_third_party_license(LspCpp/LICENSE                LspCpp-LICENSE.txt)
asy_install_third_party_license(libatomic_ops/LICENSE         libatomic_ops-LICENSE.txt)
asy_install_third_party_license(libatomic_ops/COPYING         libatomic_ops-COPYING.txt)
asy_install_third_party_license(tinyexr/LICENSE.txt           tinyexr-LICENSE.txt)
asy_install_third_party_license(gc/LICENSE.txt                gc-LICENSE.txt)
