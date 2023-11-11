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

option(ENABLE_LIBREADLINE "libreadline" true)
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
