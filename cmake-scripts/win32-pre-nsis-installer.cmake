if (NOT WIN32)
    message(FATAL_ERROR "Pre-NSIS installation is intended for windows only!")
endif()

if (NOT ASY_WIN_RESOURCE_DIR)
    message(FATAL_ERROR "ASY_WIN_RESOURCE_DIR is not defined.
Please ensure win32-specific.cmake is included before this file!")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${CMAKE_CURRENT_SOURCE_DIR}/cmake-install-win32")
endif()

# if a component is not buildable
macro(action_if_component_not_buildable message)
    if (ALLOW_PARTIAL_INSTALLATION)
        install(CODE "message(WARNING ${message})")
    else()
        install(CODE "message(FATAL_ERROR ${message})")
    endif()
endmacro()

if (NOT ASY_TEX_BUILD_ROOT)
    action_if_component_not_buildable("Documentation is not buildable")
endif()

set(ASYMPTOTE_NSI_CONFIGURATION_DIR ${CMAKE_CURRENT_BINARY_DIR}/nsifiles)
file(MAKE_DIRECTORY ${ASYMPTOTE_NSI_CONFIGURATION_DIR})

configure_file(
        ${ASY_WIN_RESOURCE_DIR}/AsymptoteInstallInfo.nsi.in
        ${ASYMPTOTE_NSI_CONFIGURATION_DIR}/AsymptoteInstallInfo.nsi
)

set(ASY_INSTALL_DIRECTORY build-${ASY_VERSION})

set(BUILD_ASY_INSTALLER_SCRIPT ${ASY_WIN_RESOURCE_DIR}/build-asymptote-installer.py)
configure_file(
        ${ASY_WIN_RESOURCE_DIR}/build-asy-installer.ps1.in
        ${ASYMPTOTE_NSI_CONFIGURATION_DIR}/build-asy-installer.ps1
)

set(ASY_PRE_NSIS_COMPONENT_NAME asy-pre-nsis)
set(ASY_NSIS_INSTALL_ARGUMENT
        COMPONENT ${ASY_PRE_NSIS_COMPONENT_NAME}
        DESTINATION ${ASY_INSTALL_DIRECTORY}
)

set(ASY_NSIS_INSTALL_RESOURCES_ARGUMENT
        COMPONENT ${ASY_PRE_NSIS_COMPONENT_NAME}
        DESTINATION .
)

# <build-root>/asy.exe -> <install-root>/asy.exe
install(TARGETS asy
        RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
        POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
        ${ASY_NSIS_INSTALL_ARGUMENT}
)

# <build-root>/base/*, <build-root>/examples -> <install-root>/
install(
        DIRECTORY ${ASY_BUILD_BASE_DIR}/ ${CMAKE_CURRENT_SOURCE_DIR}/examples
        ${ASY_NSIS_INSTALL_ARGUMENT}
)

# resources files for installer + nsi files

install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
        ${ASY_WIN_RESOURCE_DIR}/asy.ico
        ${ASY_WIN_RESOURCE_DIR}/asymptote.nsi
        ${ASYMPTOTE_NSI_CONFIGURATION_DIR}/AsymptoteInstallInfo.nsi
        ${ASYMPTOTE_NSI_CONFIGURATION_DIR}/build-asy-installer.ps1
        ${ASY_NSIS_INSTALL_RESOURCES_ARGUMENT}
)

install(
        DIRECTORY ${ASY_WIN_RESOURCE_DIR}/
        ${ASY_NSIS_INSTALL_RESOURCES_ARGUMENT}
        FILES_MATCHING PATTERN "*.nsh"
)

install(
        DIRECTORY ${ASY_TEX_BUILD_ROOT}/
        ${ASY_NSIS_INSTALL_ARGUMENT}
        FILES_MATCHING PATTERN "*.pdf"
)

if (ASY_TEX_BUILD_ROOT)
    install(
            DIRECTORY ${ASY_TEX_BUILD_ROOT}/
            ${ASY_NSIS_INSTALL_ARGUMENT}
            FILES_MATCHING PATTERN "*.pdf"
    )
endif()
