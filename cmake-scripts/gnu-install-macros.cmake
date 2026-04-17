# Only run on unix-like systems, windows does not use gnu docpath/syspath locations

if (UNIX)
    include(GNUInstallDirs)

if (CTAN_BUILD)
    set(ASYMPTOTE_SYSDIR_VALUE "")
else()
    set(ASYMPTOTE_SYSDIR_VALUE ${CMAKE_INSTALL_FULL_DATADIR}/asymptote)
endif()

    set(ASYMPTOTE_DOCDIR_VALUE ${CMAKE_INSTALL_FULL_DATADIR}/doc/asymptote)

    set(ASYMPTOTE_LICENSEDIR_DEFAULT ${ASYMPTOTE_DOCDIR_VALUE}/licenses)
    set(ASY_LICENSEDIR ${ASYMPTOTE_LICENSEDIR_DEFAULT} CACHE STRING
        "Absolute path to license files at runtime (baked into binary as ASYMPTOTE_LICENSEDIR)")
    # Compute install DESTINATION: relative when ASY_LICENSEDIR is under the prefix
    # (so cmake --install --prefix works), absolute otherwise (for distro overrides).
    cmake_path(IS_PREFIX CMAKE_INSTALL_PREFIX "${ASY_LICENSEDIR}" NORMALIZE _asy_licensedir_under_prefix)
    if(_asy_licensedir_under_prefix)
        file(RELATIVE_PATH _asy_licensedir_install_dest "${CMAKE_INSTALL_PREFIX}" "${ASY_LICENSEDIR}")
    else()
        set(_asy_licensedir_install_dest "${ASY_LICENSEDIR}")
    endif()
    set(ASY_LICENSEDIR_INSTALL_DEST "${_asy_licensedir_install_dest}" CACHE STRING
        "Install DESTINATION for license files (relative = respects CMAKE_INSTALL_PREFIX)")

    list(APPEND ASY_MACROS
            ASYMPTOTE_SYSDIR="${ASYMPTOTE_SYSDIR_VALUE}"
            ASYMPTOTE_DOCDIR="${ASYMPTOTE_DOCDIR_VALUE}"
            ASYMPTOTE_LICENSEDIR="${ASY_LICENSEDIR}"
    )
endif()
