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
