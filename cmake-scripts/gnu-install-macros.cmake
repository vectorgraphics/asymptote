# Only run on unix-like systems, windows does not use gnu docpath/syspath locations

if (UNIX)
    include(GNUInstallDirs)
    list(APPEND ASY_MACROS
            ASYMPTOTE_SYSDIR="${DATADIR}/asymptote"
            ASYMPTOTE_DOCDIR="${DATADIR}/doc/asymptote"
    )
endif()
