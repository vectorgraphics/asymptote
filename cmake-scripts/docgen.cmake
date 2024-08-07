if (NOT ENABLE_DOCGEN_DEFAULT)
    message(FATAL_ERROR "Documentation generation is disabled")
endif()

set(ASY_DOC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/doc)
set(ASY_TEX_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/docbuild)
file(MAKE_DIRECTORY ${ASY_TEX_BUILD_ROOT})
configure_file(${ASY_RESOURCE_DIR}/version.texi.in ${ASY_TEX_BUILD_ROOT}/version.texi)

set(LATEX_ARTIFRACT_EXTENSIONS aux hd idx ins log out)

find_package(LATEX COMPONENTS PDFLATEX REQUIRED)
list(
        TRANSFORM LATEX_ARTIFRACT_EXTENSIONS
        PREPEND ${ASY_TEX_BUILD_ROOT}/asy-latex.
        OUTPUT_VARIABLE ASY_LATEX_DTX_ARTIFACTS
)

add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/asy-latex.pdf ${ASY_TEX_BUILD_ROOT}/asymptote.sty
        DEPENDS ${ASY_DOC_ROOT}/asy-latex.dtx
        COMMAND ${PDFLATEX_COMPILER}
        -include-directory=${ASY_TEX_BUILD_ROOT}
        -output-directory=${ASY_TEX_BUILD_ROOT}
        ${ASY_DOC_ROOT}/asy-latex.dtx
        WORKING_DIRECTORY ${ASY_DOC_ROOT}
        BYPRODUCTS ${ASY_LATEX_DTX_ARTIFACTS}
)

add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/latexusage.pdf
        DEPENDS
            ${ASY_DOC_ROOT}/latexusage.tex
            ${ASY_TEX_BUILD_ROOT}/asymptote.sty
            asy ${ASY_OUTPUT_BASE_FILES}
        COMMAND ${PY3_INTERPRETER} ${ASY_DOC_ROOT}/build-latexusage-pdf.py
            --build-dir=${ASY_TEX_BUILD_ROOT}
            --latexusage-source-dir=${ASY_DOC_ROOT}
            --pdflatex-executable=${PDFLATEX_COMPILER}
            --asy-executable=$<TARGET_FILE:asy>
            --asy-base-dir=${ASY_BUILD_BASE_DIR}
        WORKING_DIRECTORY ${ASY_DOC_ROOT}
        BYPRODUCTS ${ASY_TEX_BUILD_ROOT}/latexusage.log
)
