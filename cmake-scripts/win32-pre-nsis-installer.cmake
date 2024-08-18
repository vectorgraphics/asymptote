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
        message(WARNING ${message})
    else()
        message(FATAL_ERROR ${message})
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
set(ASY_PRE_NSIS_COMPONENT_NAME asy-pre-nsis)

set(ASY_NSIS_INSTALL_ARGUMENT
        COMPONENT ${ASY_PRE_NSIS_COMPONENT_NAME}
        DESTINATION ${ASY_INSTALL_DIRECTORY}
)

# <build-root>/asy.exe -> <install-root>/asy.exe
install(TARGETS asy
        ${ASY_NSIS_INSTALL_ARGUMENT}
        RUNTIME DESTINATION ${ASY_INSTALL_DIRECTORY}
        ARCHIVE EXCLUDE_FROM_ALL
        LIBRARY EXCLUDE_FROM_ALL
        PUBLIC_HEADER EXCLUDE_FROM_ALL
        PRIVATE_HEADER EXCLUDE_FROM_ALL
)
# <build-root>/*.dll -> <install-root>/
# an issue is that this command also include empty directories.
# this can be cleaned up by a python script that compiles the final installation file
install(
        DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
        ${ASY_NSIS_INSTALL_ARGUMENT}
        FILES_MATCHING PATTERN "*.dll"
)

# <build-root>/base -> <install-root>/
install(
        DIRECTORY ${ASY_BUILD_BASE_DIR}/
        ${ASY_NSIS_INSTALL_ARGUMENT}
)

install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples
        ${ASY_NSIS_INSTALL_ARGUMENT}
)

if (ASY_TEX_BUILD_ROOT)
install(
        DIRECTORY ${ASY_TEX_BUILD_ROOT}/
        ${ASY_NSIS_INSTALL_ARGUMENT}
        FILES_MATCHING PATTERN "*.pdf"
    )
endif()
