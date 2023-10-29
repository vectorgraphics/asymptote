if (NOT WIN32)
    message(FATAL_ERROR "This file is only for use with windows.")
endif()

# for getopt
add_subdirectory(backports/getopt)

list(APPEND ASY_STATIC_LIBARIES getopt)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:getopt,INCLUDE_DIRECTORIES>)

# msvc compile options
if (MSVC)
    list(APPEND ASY_COMPILE_OPTS /Zc:__cplusplus)
    list(APPEND ASY_MACROS NOMINMAX)
endif()

# alot of asymptote sources use __MSDOS__ macro for checking windows
list(APPEND ASY_MACROS __MSDOS__=1)
