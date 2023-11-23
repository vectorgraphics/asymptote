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

if (MSVC)
    if (NOT GCCCOMPAT_CXX_COMPILER_FOR_MSVC)
        message(
                WARNING
                "\
GCC-compatible C++ compiler not specified, target dependency resolution for generated files may \
not work properly. If you are looking for a GCC-compatible C++ compiler on windows for preprocessing, \
we recommend the LLVM toolchain. You can find LLVM at \
\
https://releases.llvm.org/download.html\
\
or through msys2."
        )
    endif()
endif()


# additional win32 api libraries
list(APPEND ASY_STATIC_LIBARIES Shlwapi)
