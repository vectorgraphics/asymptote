if (NOT ENABLE_DOCGEN)
    message(FATAL_ERROR "Documentation generation is disabled")
endif()

set(ASY_DOC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/doc)
set(ASY_TEX_BUILD_ROOT ${CMAKE_CURRENT_BINARY_DIR}/docbuild)
file(MAKE_DIRECTORY ${ASY_TEX_BUILD_ROOT})
configure_file(${ASY_RESOURCE_DIR}/version.texi.in ${ASY_TEX_BUILD_ROOT}/version.texi)

set(LATEX_ARTIFRACT_EXTENSIONS aux hd idx ins log out toc)

set(PDFLATEX_BASE_ARGUMENTS ${PDFLATEX_COMPILER}
        -output-directory=${ASY_TEX_BUILD_ROOT}
)

find_package(LATEX COMPONENTS PDFLATEX REQUIRED)
list(
        TRANSFORM LATEX_ARTIFRACT_EXTENSIONS
        PREPEND ${ASY_TEX_BUILD_ROOT}/asy-latex.
        OUTPUT_VARIABLE ASY_LATEX_DTX_ARTIFACTS
)

add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/asy-latex.pdf ${ASY_TEX_BUILD_ROOT}/asymptote.sty
        DEPENDS ${ASY_DOC_ROOT}/asy-latex.dtx
        COMMAND ${PDFLATEX_BASE_ARGUMENTS} ${ASY_DOC_ROOT}/asy-latex.dtx
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

set(CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS ${CMAKE_COMMAND} -E copy -t ${ASY_TEX_BUILD_ROOT})
set(CMAKE_RM_BASE_ARGUMENTS ${CMAKE_COMMAND} -E rm)
set(ASY_BASE_ARGUMENTS asy -dir ${ASY_BUILD_BASE_DIR} -config '' -render=0 -noprc -noV)

macro(add_additional_pdf_outputs filename_base num_times_to_call_pdflatex compiler)
    # ARGN can be used to specify additional dependencies
    foreach (DUMMY_VAR RANGE 1 ${num_times_to_call_pdflatex})
        list(APPEND COMMAND_ARGS COMMAND ${compiler} ${ASY_DOC_ROOT}/${filename_base}.tex)
    endforeach()

    set(PDFLATEX_OUTPUT_PREFIX ${ASY_TEX_BUILD_ROOT}/${filename_base})

    # there are unfortunately still some issues with out-of-source builds
    # with pdflatex, hence we have to copy the files to the build root first
    add_custom_command(
            OUTPUT ${PDFLATEX_OUTPUT_PREFIX}.tex
            DEPENDS ${ASY_DOC_ROOT}/${filename_base}.tex
            COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS} ${ASY_DOC_ROOT}/${filename_base}.tex
    )

    add_custom_command(
            OUTPUT ${PDFLATEX_OUTPUT_PREFIX}.pdf
            DEPENDS ${PDFLATEX_OUTPUT_PREFIX}.tex
            ${ARGN}
            ${COMMAND_ARGS}
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS} -f ${PDFLATEX_OUTPUT_PREFIX}.dvi
            WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
            BYPRODUCTS
                ${PDFLATEX_OUTPUT_PREFIX}.aux
                ${PDFLATEX_OUTPUT_PREFIX}.log
                ${PDFLATEX_OUTPUT_PREFIX}.toc
    )
endmacro()

# asy -> pdf file generation

set(ASY_DOC_PDF_FILES "")

macro(add_asy_pdf_dependency_basic asyfile)
    set(ASY_DOC_FILE_OUTPUT ${ASY_TEX_BUILD_ROOT}/${asyfile}.pdf)
    # asymptote has some problems (currently as writing this) with asy files involving tex
    # and output directory not matching, so a workaround is to copy to the doc build root
    add_custom_command(
            OUTPUT ${ASY_DOC_FILE_OUTPUT}
            DEPENDS ${ASY_DOC_ROOT}/${asyfile}.asy asy ${ASY_OUTPUT_BASE_FILES}
            # copy <docroot>/file.asy -> <buildroot>/file.asy
            COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS} ${ASY_DOC_ROOT}/${asyfile}.asy
            COMMAND ${ASY_BASE_ARGUMENTS} -fpdf ${asyfile}.asy
            # cleanup <buildroot>/file.asy
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS}
            ${ASY_TEX_BUILD_ROOT}/${asyfile}.asy
            # cleanup tex artifacts, if exist
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS} -f
            ${ASY_TEX_BUILD_ROOT}/${asyfile}_.tex
            WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
    )
    list(APPEND ASY_DOC_PDF_FILES ${ASY_DOC_FILE_OUTPUT})
endmacro()

# CAD1 is here because it is needed for CAD.pdf.
add_asy_pdf_dependency_basic(CAD1)

# additional asymptote file outputs
add_additional_pdf_outputs(asyRefCard 1 ${PDFTEX_EXEC})
add_additional_pdf_outputs(TeXShopAndAsymptote 2 ${PDFLATEX_COMPILER})
add_additional_pdf_outputs(
        CAD
        3
        ${PDFLATEX_COMPILER}
        DEPENDS ${ASY_TEX_BUILD_ROOT}/CAD1.pdf
)

set(
        BASE_ASYMPTOTE_DOC_AND_TEX_FILES
        ${ASY_TEX_BUILD_ROOT}/asymptote.sty
        ${ASY_TEX_BUILD_ROOT}/asy-latex.pdf
        ${ASY_TEX_BUILD_ROOT}/CAD.pdf
        ${ASY_TEX_BUILD_ROOT}/TeXShopAndAsymptote.pdf
        ${ASY_TEX_BUILD_ROOT}/asyRefCard.pdf
)

add_custom_target(docgen DEPENDS ${BASE_ASYMPTOTE_DOC_AND_TEX_FILES})


# the following files are only used with docgen, hence they are not added if
# ENABLE_ASYMPTOTE_PDF_DOCGEN is not used
if (ENABLE_ASYMPTOTE_PDF_DOCGEN)
# asy files
set(ASY_DOC_FILE_PREFIXES
        axis3 basealign bezier bigdiagonal binarytreetest Bode brokenaxis
        colons colors cube cylinderskeleton datagraph diagonal dots
        eetomumu elliptic errorbars exp  fillcontour flow flowchartdemo
        GaussianSurface generalaxis generalaxis3 graphmarkers graphwithderiv grid3xyz
        hatch helix HermiteSpline histogram Hobbycontrol Hobbydir icon image imagecontour
        irregularcontour join join3 knots labelsquare legend lineargraph lineargraph0
        linetype log2graph loggraph loggrid logimage logticks makepen markers1 markers2 mexicanhat
        monthaxis multicontour onecontour parametricgraph penfunctionimage penimage planes quartercircle
        saddle scaledgraph shadedtiling slopefield1 square subpictures superpath tile
        triangulate unitcircle3 vectorfield
)

# independent asymptote files that can be generated with any other files
foreach(ASY_DOC_FILE_PREFIX ${ASY_DOC_FILE_PREFIXES})
    add_asy_pdf_dependency_basic(${ASY_DOC_FILE_PREFIX})
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
macro(copy_doc_asy_file_to_docbuild_root asyfile)
    add_custom_command(
            OUTPUT ${ASY_TEX_BUILD_ROOT}/${asyfile}.asy
            COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS}
            ${ASY_DOC_ROOT}/${asyfile}.asy
            DEPENDS ${ASY_DOC_ROOT}/${asyfile}.asy
    )
endmacro()

function(add_asy_file_with_asy_dependency asyfile) # <asydep1> [asydep2] ...
    list(
            TRANSFORM ARGN
            PREPEND ${ASY_TEX_BUILD_ROOT}/
            OUTPUT_VARIABLE ASY_REQUIRED_DEPS
    )
    list(
            TRANSFORM ASY_REQUIRED_DEPS
            APPEND .asy
    )

    add_custom_command(
            OUTPUT ${ASY_TEX_BUILD_ROOT}/${asyfile}.pdf
            DEPENDS ${ASY_DOC_ROOT}/${asyfile}.asy asy ${ASY_OUTPUT_BASE_FILES} ${ASY_REQUIRED_DEPS}
            # copy <docroot>/file.asy -> <buildroot>/file.asy
            COMMAND ${CMAKE_COPY_ASY_FILE_TO_DOCBUILD_BASE_ARGS} ${ASY_DOC_ROOT}/${asyfile}.asy
            COMMAND ${ASY_BASE_ARGUMENTS} -fpdf ${asyfile}.asy
            # cleanup <buildroot>/file.asy
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS} ${ASY_TEX_BUILD_ROOT}/${asyfile}.asy
            # cleanup tex artifacts, if exist
            COMMAND ${CMAKE_RM_BASE_ARGUMENTS} -f ${ASY_TEX_BUILD_ROOT}/${asyfile}_.tex
            WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
    )
endfunction()

macro(add_asy_file_from_docbuild_root asyfile)
    # does not copy from doc root to docbuild root; have to do manually
    add_custom_command(
            OUTPUT ${ASY_TEX_BUILD_ROOT}/${asyfile}.pdf
            COMMAND ${ASY_BASE_ARGUMENTS} -fpdf ${asyfile}.asy
            DEPENDS ${ASY_TEX_BUILD_ROOT}/${asyfile}.asy
            BYPRODUCTS ${ASY_TEX_BUILD_ROOT}/${asyfile}_.tex ${ASY_TEX_BUILD_ROOT}/${asyfile}_.eps
            WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
    )
endmacro()

# CDlabel + logo
copy_doc_asy_file_to_docbuild_root(logo)
add_asy_file_from_docbuild_root(logo)
add_asy_file_with_asy_dependency(CDlabel logo)

# bezier2 & beziercurve
copy_doc_asy_file_to_docbuild_root(beziercurve)
add_asy_file_from_docbuild_root(beziercurve)
add_asy_file_with_asy_dependency(bezier2 beziercurve)

list(APPEND ASY_DOC_PDF_FILES
        ${ASY_TEX_BUILD_ROOT}/logo.pdf
        ${ASY_TEX_BUILD_ROOT}/CDlabel.pdf
        ${ASY_TEX_BUILD_ROOT}/beziercurve.pdf
        ${ASY_TEX_BUILD_ROOT}/bezier2.pdf
)

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
else()
add_custom_command(
        OUTPUT ${ASY_TEX_BUILD_ROOT}/asymptote.pdf
        DEPENDS
        ${ASY_TEX_BUILD_ROOT}/options
        ${ASY_TEX_BUILD_ROOT}/latexusage.pdf
        ${ASY_DOC_ROOT}/asymptote.texi
        ${ASY_DOC_PDF_FILES}
        COMMAND ${TEXI2DVI} --pdf ${ASY_DOC_ROOT}/asymptote.texi
        WORKING_DIRECTORY ${ASY_TEX_BUILD_ROOT}
        BYPRODUCTS ${ASYMPTOTE_PDF_EXTRA_ARTIFACTS}
)
endif()
add_custom_target(asymptote_pdf_file DEPENDS ${ASY_TEX_BUILD_ROOT}/asymptote.pdf)
add_dependencies(docgen asymptote_pdf_file)
endif() # ENABLE_ASYMPTOTE_PDF_DOCGEN
