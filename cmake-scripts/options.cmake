# Perl

option(PERL_INTERPRETER "Perl interpreter")

if(NOT PERL_INTERPRETER)
    message(STATUS "No Perl interpreter specified, attempting to find perl")
    find_program(
            PERL_INTERPRETER_FOUND
            perl
            REQUIRED
    )
    message(STATUS "Found perl at ${PERL_INTERPRETER_FOUND}")
    set(PERL_INTERPRETER ${PERL_INTERPRETER_FOUND})
endif()

execute_process(COMMAND ${PERL_INTERPRETER} -e "print \"$]\"" OUTPUT_VARIABLE PERL_VERSION)
message(STATUS "Perl version: ${PERL_VERSION}")


# Python

option(PY3_INTERPRETER "Python 3 interpreter")

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
    set(PY3_INTERPRETER ${PY3_INTERPRETER_FOUND})
else()
    set(PY_INTERPRETER_IS_PY3 TRUE)
    set(VARIABLE_RESULT_VAR PY_INTERPRETER_IS_PY3)
    verify_py3_interpreter_is_py3(VARIABLE_RESULT_VAR ${PY3_INTERPRETER})
endif()

execute_process(COMMAND ${PY3_INTERPRETER} --version OUTPUT_VARIABLE PY3_VERSION)
message(STATUS "Version: ${PY3_VERSION}")

# windows flex + bison
option(WIN32_FLEX_BINARY
        "Flex binary for windows. If not specified, downloads from winflexibson. This option is inert on UNIX systems")
option(WIN32_BISON_BINARY
        "Bison binary for windows. If not specified, downloads from winflexbison. This option is inert on UNIX systems")

# feature libraries

option(ENABLE_GC "enable boehm gc support" true)
option(ENABLE_CURL "enable curl support" true)
option(ENABLE_READLINE "libreadline" true)
option(ENABLE_THREADING "enable threading support" true)
option(ENABLE_GSL "Enable GSL support" true)
option(ENABLE_EIGEN3 "Enable eigen3 support" true)
option(ENABLE_FFTW3 "Enable fftw3 support" true)
option(ENABLE_OPENGL "Whether to enable opengl or not." true)
option(ENABLE_GL_COMPUTE_SHADERS
        "Whether to enable compute shaders for OpenGL. Requires OpenGL >= 4.3 and GL_ARB_compute_shader" true)
option(ENABLE_GL_SSBO
        "Whether to enable compute SSBO. Requires OpenGL >= 4.3 and GL_ARB_shader_storage_buffer_object" true)


# RPC.
if (UNIX)
    set(DEFAULT_ENABLE_RPC TRUE)
else()
    # Not sure if there's a way to get rpc lib working on windows, as of yet
    set(DEFAULT_ENABLE_RPC FALSE)
endif()

option(
        ENABLE_RPC_FEATURES
        "Whether to enable XDR/RPC features. Also enables V3D. For Unix systems only"
        ${DEFAULT_ENABLE_RPC})

# Additional options

option(DEBUG_GC_ENABLE "Enable debug mode for gc" false)
option(DEBUG_GC_BACKTRACE_ENABLE "Enable backtrace for gc" false)

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
option(GCCCOMPAT_CXX_COMPILER_FOR_MSVC
        "gcc-compatible C++ compiler for preprocessing with MSVC toolchain. This option is inert if not using MSVC.
This option is only used for preprocessing, it is not used for compilation."
)

# CUDA + asy cuda reflect
set(ENABLE_CUDA_ASY_REFLECT_DEFAULT false)
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    set(ENABLE_CUDA_ASY_REFLECT_DEFAULT true)
endif()

option(
    ENABLE_CUDA_ASY_REFLECT
    "Enable target for reflect excutable for generating IBL lighting data.
Requires CUDA installed and a CUDA-compatible NVIDIA Graphics card"
    ${ENABLE_CUDA_ASY_REFLECT_DEFAULT}
)

# Language server protocol
option(
    ENABLE_LSP
    "Enable Language Server Protocol support."
    true
)
