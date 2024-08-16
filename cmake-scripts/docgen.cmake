if (NOT ENABLE_DOCGEN)
    message(FATAL_ERROR "Documentation generation is disabled")
endif()

set(ASY_DOC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/doc)
set(ASY_TEX_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/docbuild)
file(MAKE_DIRECTORY ${ASY_TEX_BUILD_ROOT})
configure_file(${ASY_RESOURCE_DIR}/version.texi.in ${ASY_TEX_BUILD_ROOT}/version.texi)

set(LATEX_ARTIFRACT_EXTENSIONS aux hd idx ins log out toc)

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

# asy files
set(ASY_DOC_FILE_PREFIXES
        axis3 basealign bezier bezier2 beziercurve bigdiagonal binarytreetest Bode brokenaxis
        CAD1 colons colors cube cylinderskeleton datagraph diagonal dots
        eetomumu elliptic errorbars exp  fillcontour flow flowchartdemo
        GaussianSurface generalaxis generalaxis3 graphmarkers graphwithderiv grid3xyz
        hatch helix HermiteSpline histogram Hobbycontrol Hobbydir icon image imagecontour
        irregularcontour join join3 knots labelsquare legend lineargraph lineargraph0
        linetype log2graph loggraph loggrid logimage logticks makepen markers1 markers2 mexicanhat
        monthaxis multicontour onecontour parametricgraph penfunctionimage penimage planes quartercircle
        saddle scaledgraph shadedtiling slopefield1 square subpictures superpath tile
        triangulate unitcircle3 vectorfield
)

set(ASY_DOC_PDF_FILES "")

set(CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS ${CMAKE_COMMAND} -E copy -t ${ASY_TEX_BUILD_ROOT})
set(CMAKE_RM_BASE_ARGUMENTS ${CMAKE_COMMAND} -E rm)
set(ASY_BASE_ARGUMENTS asy -dir ${ASY_BUILD_BASE_DIR} -config '' -render=0 -noprc -noV)

# independent asymptote files that can be generated with any other files
foreach(ASY_DOC_FILE_PREFIX ${ASY_DOC_FILE_PREFIXES})
    set(ASY_DOC_FILE_OUTPUT ${ASY_TEX_BUILD_ROOT}/${ASY_DOC_FILE_PREFIX}.pdf)
    # asymptote has some problems (currently as writing this) with asy files involving tex
    # and output directory not matching, so a workaround is to copy to the doc build root
    add_custom_command(
            OUTPUT ${ASY_DOC_FILE_OUTPUT}
            DEPENDS ${ASY_DOC_ROOT}/${ASY_DOC_FILE_PREFIX}.asy asy ${ASY_OUTPUT_BASE_FILES}
            # copy <docroot>/file.asy -> <buildroot>/file.asy
            COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS} ${ASY_DOC_ROOT}/${ASY_DOC_FILE_PREFIX}.asy
            COMMAND ${ASY_BASE_ARGUMENTS} -fpdf ${ASY_DOC_FILE_PREFIX}.asy
            # cleanup <buildroot>/file.asy
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS}
                ${ASY_TEX_BUILD_ROOT}/${ASY_DOC_FILE_PREFIX}.asy
            # cleanup tex artifacts, if exist
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS} -f
                ${ASY_TEX_BUILD_ROOT}/${ASY_DOC_FILE_PREFIX}_.tex
            WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
    )
    list(APPEND ASY_DOC_PDF_FILES ${ASY_DOC_FILE_OUTPUT})
endforeach()

macro(add_asy_file_with_extension asy_file extra_ext)
    set(ASY_DOC_FILE_OUTPUT ${ASY_TEX_BUILD_ROOT}/${asy_file}.pdf)
    set(ASY_AUX_FILE_NAME ${asy_file}.${extra_ext})
    # asymptote has some problems (currently as writing this) with asy files involving tex
    # and output directory not matching, so a workaround is to copy to the doc build root
    add_custom_command(
            OUTPUT ${ASY_DOC_FILE_OUTPUT}
            DEPENDS
                ${ASY_DOC_ROOT}/${asy_file}.asy
                ${ASY_DOC_ROOT}/${ASY_AUX_FILE_NAME}
                asy ${ASY_OUTPUT_BASE_FILES}
            COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS}
                ${ASY_DOC_ROOT}/${asy_file}.asy
                ${ASY_DOC_ROOT}/${ASY_AUX_FILE_NAME}
            COMMAND ${ASY_BASE_ARGUMENTS} -fpdf ${ASY_DOC_FILE_PREFIX}.asy
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS}
                ${ASY_TEX_BUILD_ROOT}/${asy_file}.asy
                ${ASY_TEX_BUILD_ROOT}/${ASY_AUX_FILE_NAME}
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS} -f
                ${ASY_TEX_BUILD_ROOT}/${asy_file}_.tex
            WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
    )
    list(APPEND ASY_DOC_PDF_FILES ${ASY_DOC_FILE_OUTPUT})
endmacro()

# asy + csv Files
foreach(ASY_DOC_FILE_PREFIX diatom secondaryaxis westnile)
    add_asy_file_with_extension(${ASY_DOC_FILE_PREFIX} csv)
endforeach()

# asy + dat files
foreach(ASY_DOC_FILE_PREFIX filegraph leastsquares)
    add_asy_file_with_extension(${ASY_DOC_FILE_PREFIX} dat)
endforeach()


# handle CDlabel and logo separately

# <doc-root>/logo.asy -> <build-root>/logo.asy
add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/logo.asy
        COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS}
            ${ASY_DOC_ROOT}/logo.asy
        DEPENDS ${ASY_DOC_ROOT}/logo.asy
)

# <build-root>/logo.asy -> <build-root>/logo.pdf [+ artifacts]
add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/logo.pdf
        COMMAND ${ASY_BASE_ARGUMENTS} -fpdf logo.asy
        DEPENDS ${ASY_TEX_BUILD_ROOT}/logo.asy
        BYPRODUCTS ${ASY_TEX_BUILD_ROOT}/logo_.tex ${ASY_TEX_BUILD_ROOT}/logo_.eps
        WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
)

# <build-root>/logo.asy + <doc-root>/CDlabel.asy -> <build-root>/CDlabel.pdf
add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/CDlabel.pdf
        DEPENDS ${ASY_DOC_ROOT}/CDlabel.asy asy ${ASY_OUTPUT_BASE_FILES} ${ASY_TEX_BUILD_ROOT}/logo.asy
        # copy <docroot>/file.asy -> <buildroot>/file.asy
        COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS} ${ASY_DOC_ROOT}/CDlabel.asy
        COMMAND ${ASY_BASE_ARGUMENTS} -fpdf CDlabel.asy
        # cleanup <buildroot>/file.asy
        COMMAND ${CMAKE_RM_BASE_ARGUMENTS} ${ASY_TEX_BUILD_ROOT}/CDlabel.asy
        # cleanup tex artifacts, if exist
        COMMAND ${CMAKE_RM_BASE_ARGUMENTS} -f ${ASY_TEX_BUILD_ROOT}/CDlabel_.tex
        WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
)

list(APPEND ASY_DOC_PDF_FILES ${ASY_TEX_BUILD_ROOT}/logo.pdf ${ASY_TEX_BUILD_ROOT}/CDlabel.pdf )

# options file
add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/options
        DEPENDS asy ${ASY_DOC_ROOT}/gen-asy-options-file.py
        COMMAND ${PY3_INTERPRETER} ${ASY_DOC_ROOT}/gen-asy-options-file.py
        --asy-executable=$<TARGET_FILE:asy>
        --output-file=${ASY_TEX_BUILD_ROOT}/options
)

# asymptote.pdf

set(TEXI_ARTIFACT_EXTENSIONS log tmp cp toc cps aux)
list(
        TRANSFORM TEXI_ARTIFACT_EXTENSIONS
        PREPEND ${ASY_TEX_BUILD_ROOT}/asymptote.
        OUTPUT_VARIABLE ASYMPTOTE_PDF_EXTRA_ARTIFACTS
    )

if (WIN32)
if (WIN32_TEXINDEX STREQUAL WSL)
    set(TEXINDEX_WRAPPER ${CMAKE_CURRENT_SOURCE_DIR}/windows/texindex-wsl.cmd)
else()
    set(TEXINDEX_WRAPPER ${WIN32_TEXINDEX})
endif()
add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/asymptote.pdf
        DEPENDS
            ${ASY_TEX_BUILD_ROOT}/options
            ${ASY_TEX_BUILD_ROOT}/latexusage.pdf
            ${ASY_DOC_ROOT}/asymptote.texi
            ${ASY_DOC_PDF_FILES}
        COMMAND ${PY3_INTERPRETER}
            ${ASY_DOC_ROOT}/build-asymptote-pdf-win.py
            --texify-loc=${TEXIFY}
            --texindex-loc=${TEXINDEX_WRAPPER}
            --texi-file=${ASY_DOC_ROOT}/asymptote.texi
        WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
        BYPRODUCTS ${ASYMPTOTE_PDF_EXTRA_ARTIFACTS}
)

add_custom_target(docgen DEPENDS ${ASY_TEX_BUILD_ROOT}/asymptote.pdf)
else()
    # TODO: Add asymptote.pdf generation for linux
endif()
