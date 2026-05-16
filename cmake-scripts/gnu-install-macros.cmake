# Only run on unix-like systems, windows does not use gnu docpath/syspath locations

if (UNIX)
    include(GNUInstallDirs)

    # ASYMPTOTE_SYSDIR is always set globally for asycore and shared libraries.
    set(ASYMPTOTE_SYSDIR_VALUE ${CMAKE_INSTALL_FULL_DATADIR}/asymptote CACHE PATH
        "Path baked into the asy binary as the system base dir (ASYMPTOTE_SYSDIR). \
Override for out-of-install-tree builds, e.g. in the sandbox preset.")

    set(ASYMPTOTE_DOCDIR_VALUE ${CMAKE_INSTALL_FULL_DATADIR}/doc/asymptote)

    set(ASYMPTOTE_LICENSEDIR_DEFAULT ${ASYMPTOTE_DOCDIR_VALUE}/licenses)
    set(ASY_LICENSEDIR ${ASYMPTOTE_LICENSEDIR_DEFAULT} CACHE STRING
        "Absolute path to license files at runtime (baked into binary as ASYMPTOTE_LICENSEDIR)")
    # Compute install DESTINATION: relative when ASY_LICENSEDIR is under the prefix
    # (so cmake --install --prefix works), absolute otherwise (for distro overrides).
    cmake_path(IS_PREFIX CMAKE_INSTALL_PREFIX "${ASY_LICENSEDIR}" NORMALIZE _asy_licensedir_under_prefix)
    if(_asy_licensedir_under_prefix)
        file(RELATIVE_PATH ASY_LICENSEDIR_INSTALL_DEST "${CMAKE_INSTALL_PREFIX}" "${ASY_LICENSEDIR}")
    else()
        set(ASY_LICENSEDIR_INSTALL_DEST "${ASY_LICENSEDIR}")
    endif()

    list(APPEND ASY_MACROS
            ASYMPTOTE_SYSDIR="${ASYMPTOTE_SYSDIR_VALUE}"
            ASYMPTOTE_DOCDIR="${ASYMPTOTE_DOCDIR_VALUE}"
            ASYMPTOTE_LICENSEDIR="${ASY_LICENSEDIR}"
    )
endif()