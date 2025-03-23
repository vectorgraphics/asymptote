include(CMakeDependentOption)

# version override

set(
        ASY_VERSION_OVERRIDE "" CACHE STRING
        "Overriding asymptote version. If left blank, version is determined from configure.ac."
)

# Perl

set(PERL_INTERPRETER "" CACHE STRING "Perl interpreter. If left empty, will try to determine interpreter automatically")

if(NOT PERL_INTERPRETER)
    message(STATUS "No Perl interpreter specified, attempting to find perl")
    find_program(
            PERL_INTERPRETER_FOUND
            perl
            REQUIRED
    )
    message(STATUS "Found perl at ${PERL_INTERPRETER_FOUND}")
    set(PERL_INTERPRETER ${PERL_INTERPRETER_FOUND} CACHE STRING "" FORCE)
endif()

execute_process(COMMAND ${PERL_INTERPRETER} -e "print \"$]\"" OUTPUT_VARIABLE PERL_VERSION)
message(STATUS "Perl version: ${PERL_VERSION}")

# Python

set(PY3_INTERPRETER "" CACHE STRING "Python 3 interpreter. If left empty, will try to determine Python automatically")

function(verify_py3_interpreter_is_py3 validator_result_var py_interpreter)
    execute_process(
            COMMAND ${py_interpreter} -c "import sys; print(int(sys.version[0])>=3,end='')"
            OUTPUT_VARIABLE PY3_INTERPRETER_VERSION_RESULT)
    if (NOT PY3_INTERPRETER_VERSION_RESULT STREQUAL "True")
        set(${validator_result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

if(NOT PY3_INTERPRETER)
    message(STATUS "No Python3 interpreter specified, attempting to find python")
    find_program(
            PY3_INTERPRETER_FOUND
            NAMES python3 python
            VALIDATOR verify_py3_interpreter_is_py3
            REQUIRED
    )
    message(STATUS "Found python3 at ${PY3_INTERPRETER_FOUND}")
    set(PY3_INTERPRETER ${PY3_INTERPRETER_FOUND} CACHE STRING "" FORCE)
else()
    set(PY_INTERPRETER_IS_PY3 TRUE)
    set(VARIABLE_RESULT_VAR PY_INTERPRETER_IS_PY3)
    verify_py3_interpreter_is_py3(VARIABLE_RESULT_VAR ${PY3_INTERPRETER})

    if (NOT PY_INTERPRETER_IS_PY3)
        message(FATAL_ERROR "Specified python interpreter cannot be used as python3 interpreter!")
    endif()
endif()

execute_process(COMMAND ${PY3_INTERPRETER} --version OUTPUT_VARIABLE PY3_VERSION)
message(STATUS "Version: ${PY3_VERSION}")

# windows flex + bison
set(
        WIN32_FLEX_BINARY "" CACHE STRING
        "Flex binary for windows. If not specified, downloads from winflexibson. This option is inert on UNIX systems"
)
set(
        WIN32_BISON_BINARY "" CACHE STRING
        "Bison binary for windows. If not specified, downloads from winflexbison. This option is inert on UNIX systems"
)

# feature libraries

option(ENABLE_GC "enable boehm gc support" true)
option(ENABLE_CURL "enable curl support" true)
option(ENABLE_READLINE "libreadline" true)
option(ENABLE_THREADING "enable threading support" true)
option(ENABLE_GSL "Enable GSL support" true)
option(ENABLE_EIGEN3 "Enable eigen3 support" true)
option(ENABLE_FFTW3 "Enable fftw3 support" true)
option(ENABLE_OPENGL "Whether to enable opengl or not." true)
cmake_dependent_option(ENABLE_GL_COMPUTE_SHADERS
        "Whether to enable compute shaders for OpenGL. Requires OpenGL >= 4.3 and GL_ARB_compute_shader"
        true "ENABLE_OPENGL" false)
cmake_dependent_option(ENABLE_GL_SSBO
        "Whether to enable compute SSBO. Requires OpenGL >= 4.3 and GL_ARB_shader_storage_buffer_object"
        true "ENABLE_OPENGL" false)

option(
        ENABLE_RPC_FEATURES
        "Whether to enable XDR/RPC features. Also enables V3D. If compiling on UNIX systems, requires libtirpc to be installed."
        true)

# Additional options

option(DEBUG_GC_ENABLE "Enable debug mode for gc" false)
option(DEBUG_GC_BACKTRACE_ENABLE "Enable backtrace for gc" false)
option(CTAN_BUILD "Build for CTAN." false)

option(
        ENABLE_COMPACT_ZERO_BUILD "\
Set COMPACT flag to 0. \
Unless if building for debugging/testing with an explicit need for additional type verification, \
this option should be turned off."
        false)

# additional optimization options

if (CMAKE_BUILD_TYPE IN_LIST cmake_release_build_types)
    set(default_lto true)
else()
    set(default_lto false)
endif()

option(OPTIMIZE_LINK_TIME "Enable link-time optimization. Enabled by default in release build types" ${default_lto})

# testing
option(ENABLE_ASY_CXXTEST "Enable C++-side testing. This option is inert for final asy libraries and binaries" true)
option(
        DOWNLOAD_GTEST_FROM_SRC "Download google test from googletest's github repo. Otherwise use system libraries."
        true)

# msvc-specific
# The only reason this option is here is because msvc compiler (cl.exe) does not partial preprocessing
# (e.g. ignore missing headers and treat them as generated files or depfile generation with missing headers)
# We use MSVC compiler for all C++ compilation/linking
set(GCCCOMPAT_CXX_COMPILER_FOR_MSVC
        "" CACHE STRING
        "gcc-compatible C++ compiler for preprocessing with MSVC toolchain. This option is inert if not using MSVC.
This option is only used for preprocessing, it is not used for compilation."
)

# CUDA + asy cuda reflect
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    set(CAN_COMPILE_CUDA_REFLECT true)
endif()

cmake_dependent_option(
    ENABLE_CUDA_ASY_REFLECT
    "Enable target for reflect excutable for generating IBL lighting data.
Requires CUDA installed and a CUDA-compatible NVIDIA Graphics card"
    true "CAN_COMPILE_CUDA_REFLECT" false
)

# Language server protocol
option(
    ENABLE_LSP
    "Enable Language Server Protocol support."
    true
)

# documentation
set(WIN32_TEXINDEX "WSL" CACHE STRING
        "Location to texindex for windows, or WSL to use internal WSL wrapper.
Inert for non-windows systems.")

function(determine_asymptote_pdf_gen_possible_win32)
    # windows doesn't have an up-to-date
    # texi2dvi release in multiple years, so
    # we are using MikTeX's texify
    find_program(TEXIFY texify)
    if (NOT TEXIFY)
        message(STATUS "texify not found; will not enable docgen by default")
        set(ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE false PARENT_SCOPE)
        return()
    endif()

    if (NOT WIN32_TEXINDEX)
        message(STATUS "texindex for windows not given; will not enable docgen by default")
        set(ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE false PARENT_SCOPE)
        return()
    endif()

    # another issue is that
    if (WIN32_TEXINDEX STREQUAL WSL)
        execute_process(
                COMMAND wsl sh -c "which texindex >/dev/null 2>/dev/null && echo OK"
                OUTPUT_VARIABLE TEXINDEX_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (NOT TEXINDEX_RESULT STREQUAL "OK")
            message(STATUS "Cannot execute texindex on wsl; will not enable docgen by default")
            set(ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE false PARENT_SCOPE)
            return()
        endif()
    endif()
    set(ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE true PARENT_SCOPE)
endfunction()

set(ENABLE_BASE_DOCGEN_POSSIBLE false)
set(ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE false)

# finding latex and other programs needed
# pdflatex
find_package(LATEX COMPONENTS PDFLATEX)

# pdftex
set(PDFTEX_EXEC "" CACHE STRING "pdftex. If left empty, will try to determine interpreter automatically")
if (NOT PDFTEX)
    message(STATUS "No pdftex specified, attempting to find pdftex")
    find_program(
            PDFTEX_EXEC_FOUND
            pdftex
    )
    if (PDFTEX_EXEC_FOUND)
        message(STATUS "Found pdftex at ${PDFTEX_EXEC_FOUND}")
        set(PDFTEX_EXEC ${PDFTEX_EXEC_FOUND} CACHE STRING "" FORCE)
    endif()
endif()

if (LATEX_PDFLATEX_FOUND AND PDFTEX_EXEC)
    set(ENABLE_BASE_DOCGEN_POSSIBLE true)

    if (WIN32)
        determine_asymptote_pdf_gen_possible_win32()
    elseif(UNIX)
        find_program(TEXI2DVI texi2dvi)
        if (TEXI2DVI)
            set(ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE true)
        endif()
    endif()
endif()

set(
        EXTERNAL_DOCUMENTATION_DIR "" CACHE STRING
        "If specified, installation will use files from this directory as documentation.
In particular,

- if ENABLE_DOCGEN and ENABLE_ASYMPTOTE_PDF_DOCGEN is enabled and the system has the capability to build
all documentation files, this option is inert.
- if ENABLE_DOCGEN is enabled but ENABLE_ASYMPTOTE_PDF_DOCGEN is disabled or if the system cannot produce asymptote.pdf,
only asymptote.pdf will be copied from this directory.
- if ENABLE_DOCGEN is disabled, every documentation file will be copied from this directory.
"
)

cmake_dependent_option(
    ENABLE_DOCGEN
    "Enable basic document generation. Requires pdflatex"
    true
    "ENABLE_BASE_DOCGEN_POSSIBLE"
    false
)

cmake_dependent_option(
        ENABLE_ASYMPTOTE_PDF_DOCGEN
        "Enable asymptote.pdf document generation. Requires texinfo, and additionally WSL + texindex on windows."
        true
        "ENABLE_ASYMPTOTE_PDF_DOCGEN_POSSIBLE;ENABLE_DOCGEN"
        false
)

# misc files
option(
        ENABLE_MISCFILES_GEN
        "Enable generation of non-essential, non-documentation asymptote files (e.g. asy.list, asy-keywords.el) "
        true
)

# warnings if external docs dir is not given
if (NOT EXTERNAL_DOCUMENTATION_DIR)
    if (NOT ENABLE_DOCGEN)
        message(STATUS "Build is not generating documentation.
If you are planning on generating installation files, please make sure you have access to
documentation files in a directory and specify this directory in EXTERNAL_DOCUMENTATION_DIR cache variable.
")
    elseif(NOT ENABLE_ASYMPTOTE_PDF_DOCGEN)
        message(STATUS "Build is not generating asymptote.pdf.
If you are planning on generating installation files, please make sure you have access to asymptote.pdf
in a directory and specify this directory in EXTERNAL_DOCUMENTATION_DIR cache variable.
")
    endif()

    if (NOT ENABLE_MISCFILES_GEN)
        message(STATUS "Build is not generating non-essential, non-documentation asymptote files.
If you are planning on generating installation files, please make sure you have access to asy-keywords.el
in a directory and specify this directory in EXTERNAL_DOCUMENTATION_DIR cache variable.
")
    endif()
endif()

# windows-specific installation
option(
        ALLOW_PARTIAL_INSTALLATION
        "Allow installation to go through, even if not every component is buildable.
        CMake will produce a warning instead of a fatal error."
        false
)
