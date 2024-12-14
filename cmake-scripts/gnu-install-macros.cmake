# Only run on unix-like systems, windows does not use gnu docpath/syspath locations

if (UNIX)
    include(GNUInstallDirs)

if (CTAN_BUILD)
    set(ASYMPTOTE_SYSDIR_VALUE "")
else()
    set(ASYMPTOTE_SYSDIR_VALUE ${DATADIR}/asymptote)
endif()

    list(APPEND ASY_MACROS
            ASYMPTOTE_SYSDIR="${DATADIR}/asymptote"
            ASYMPTOTE_DOCDIR="${DATADIR}/doc/asymptote"
    )
endif()
