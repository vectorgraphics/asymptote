if (NOT LINUX)
    if (UNIX)
        message(WARNING "This file has not been tested on non-linux unix systems. It may not work!")
        # TODO: Do more testing on non-linux UNIX systems.
    else()
        message(FATAL_ERROR "This file is only for use with unix systems")
    endif()
endif()

if (CTAN_BUILD)
    message(FATAL_ERROR "system install is not supported for CTAN builds.")
endif()

# Requires gnu-install-macros to be ran already


set(PERMISSION_755_LIST
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

set(PERMISSION_644_LIST
        OWNER_READ OWNER_WRITE
        GROUP_READ
        WORLD_READ
)

set(ASY_BASE_EXTRA_FILES_NAME
        asy-mode.el asy-init.el asy.vim
        asy_filetype.vim asy-kate.sh asymptote.py
        reload.js nopapersize.ps)

list(
        TRANSFORM ASY_BASE_EXTRA_FILES_NAME
        PREPEND ${ASY_SOURCE_BASE_DIR}/
        OUTPUT_VARIABLE ASY_BASE_EXTRA_FILES
)


set(ASY_INSTALL_SYSDIR_VALUE ${CMAKE_INSTALL_DATADIR}/asymptote)
set(ASY_BASE_INSTALL_COMPONENT asy)
# installing files

#region base files
install(
        TARGETS asy
        PERMISSIONS ${PERMISSION_755_LIST}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# base/* -> <sysdir>/
install(
        DIRECTORY ${ASY_BUILD_BASE_DIR}/
        DESTINATION ${ASY_INSTALL_SYSDIR_VALUE}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        FILE_PERMISSIONS ${PERMISSION_644_LIST}
)

# extra base files -> <sysdir>/
install(
        FILES ${ASY_MISC_FILES_OUT_DIR}/asy-keywords.el ${ASY_BASE_EXTRA_FILES}
        DESTINATION ${ASY_INSTALL_SYSDIR_VALUE}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        PERMISSIONS ${PERMISSION_644_LIST}
)
#endregion

#region example files
set(ASYMPTOTE_EXAMPLESDIR_VALUE ${ASY_INSTALL_SYSDIR_VALUE}/examples)

# example files
install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/
        DESTINATION ${ASYMPTOTE_EXAMPLESDIR_VALUE}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        FILE_PERMISSIONS ${PERMISSION_644_LIST}
)

# extra examples from docs -> examples
install(
        DIRECTORY ${ASY_DOC_DIR}/extra/
        DESTINATION ${ASYMPTOTE_EXAMPLESDIR_VALUE}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        FILES_MATCHING PATTERN "*.asy"
        PERMISSIONS ${PERMISSION_644_LIST}
)

# DOCEXTRA files -> examples
install(
        DIRECTORY ${ASY_DOC_DIR}/
        DESTINATION ${ASYMPTOTE_EXAMPLESDIR_VALUE}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        FILES_MATCHING
            PATTERN "*.asy"
            PATTERN "*.csv"
            PATTERN "*.dat"
        PERMISSIONS ${PERMISSION_644_LIST}
)

set(ASY_DOCEXTRA_FILE_NAMES latexusage.tex externalprc.tex pixel.pdf)
list(
        TRANSFORM ASY_DOCEXTRA_FILE_NAMES
        PREPEND ${ASY_DOC_DIR}/
        OUTPUT_VARIABLE ASY_DOCEXTRA_FILES
)

install(
        FILES ${ASY_DOCEXTRA_FILES}
        DESTINATION ${ASYMPTOTE_EXAMPLESDIR_VALUE}
        COMPONENT ${ASY_BASE_INSTALL_COMPONENT}
        PERMISSIONS ${PERMISSION_644_LIST}
)
#endregion

#region documentation files
if (ENABLE_DOCGEN)
    set(ASY_DOCS_INSTALL_COMPONENT asy-docs)
    set(ASY_INSTALL_DOCDIR_VALUE ${CMAKE_INSTALL_DATADIR}/asymptote/doc)

    set(ASY_DOCFILE_PDF_FILES ${BASE_ASYMPTOTE_DOC_AND_TEX_FILES})
    list(FILTER ASY_DOCFILE_PDF_FILES INCLUDE REGEX "^.*\.pdf$")
    list(APPEND ASY_DOCFILE_PDF_FILES ${ASY_TEX_BUILD_ROOT}/asymptote.pdf)

    # pdf files
    install(
            FILES ${ASY_DOCFILE_PDF_FILES}
            COMPONENT ${ASY_DOCS_INSTALL_COMPONENT}
            DESTINATION ${ASY_INSTALL_DOCDIR_VALUE}
            PERMISSIONS ${PERMISSION_644_LIST}
    )
endif()
#endregion
