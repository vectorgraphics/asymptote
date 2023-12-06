if (NOT WIN32)
    message(FATAL_ERROR "This file is only for use with windows.")
endif()

# msvc compile options
if (MSVC)
    list(APPEND ASY_COMPILE_OPTS
            /Zc:__cplusplus /Zc:__STDC__
            /Zc:externC /Zc:preprocessor
            /Zc:hiddenFriend)
endif()

# alot of asymptote sources use __MSDOS__ macro for checking windows
list(APPEND ASY_MACROS NOMINMAX __MSDOS__=1)


# set ASYMPTOTE_SYSTEM_DIR to empty string
list(APPEND ASY_MACROS ASYMPTOTE_SYSDIR="")

set(BUILD_SHARED_LIBS OFF)

# gcc/clang++ compatible validators

function(validate_gcc_compat_cxx validator_result_var gcccompat_compiler)
    execute_process(
            COMMAND ${gcccompat_compiler} "--version"
            OUTPUT_VARIABLE COMPILER_RESULT)

    if (NOT COMPILER_RESULT MATCHES 
        "(clang version )|(g\\+\\+\\.exe \\(MinGW\\))|(g\\+\\+ \\(GCC\\))")
        set(${validator_result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()


macro(find_compatible_gcc_compilers)
    find_program(
        GCCCOMPAT_CXX_COMPILER_FOUND
        NAMES clang++ g++
        VALIDATOR validate_gcc_compat_cxx
    )

    if (GCCCOMPAT_CXX_COMPILER_FOUND)
        message(STATUS "Found clang++/g++ at ${GCCCOMPAT_CXX_COMPILER_FOUND}")
        set(GCCCOMPAT_CXX_COMPILER_FOR_MSVC ${GCCCOMPAT_CXX_COMPILER_FOUND})
    endif()
endmacro()

# attempting to find gcc-compatible C++ compiler

if (MSVC)
    if (NOT GCCCOMPAT_CXX_COMPILER_FOR_MSVC)
        message(STATUS "GCCCOMPAT_CXX_COMPILER_FOR_MSVC not given, will attempt to find g++.exe or clang++.exe")
        find_compatible_gcc_compilers()
    endif()

    if (GCCCOMPAT_CXX_COMPILER_FOR_MSVC)
        message(STATUS "Using GCC-compatible C++ compiler at \
${GCCCOMPAT_CXX_COMPILER_FOR_MSVC} for preprocessing.")
    endif()

    if (NOT GCCCOMPAT_CXX_COMPILER_FOR_MSVC)
        message(WARNING "\
GCC-compatible C++ compiler not specified, target dependency resolution for generated files may \
not work properly. If you are looking for a GCC-compatible C++ compiler on windows for preprocessing, \
we recommend the LLVM toolchain. You can find LLVM at \
\
https://releases.llvm.org/download.html \
\
or through msys2. After that, re-run cmake with either \
-DGCCCOMPAT_CXX_COMPILER_FOR_MSVC=<path to clang++.exe> \
or if running msvc/* presets, with GCCCOMPAT_CXX_COMPILER_FOR_MSVC \
environment variable set to the path \
to clang++.exe.")
    endif()
endif()


# additional win32 api libraries
list(APPEND ASY_STATIC_LIBARIES Shlwapi)

# RC file
set(ASY_WIN_RESOURCE_DIR ${ASY_RESOURCE_DIR}/windows)
list(APPEND ASY_WIN_RC_FILE ${ASY_WIN_RESOURCE_DIR}/asy.rc)