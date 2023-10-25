
if (ENABLE_LSP)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/LspCpp)
list(APPEND ASY_STATIC_LIBARIES lspcpp)

else()
    # only include lsp libraries
    list(APPEND ASYMPTOTE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/LspCpp/include)
endif()

find_package(tinyexr CONFIG REQUIRED)
list(APPEND ASY_STATIC_LIBARIES unofficial::tinyexr::tinyexr)

# zlib
find_package(ZLIB REQUIRED)
list(APPEND ASY_STATIC_LIBARIES ZLIB::ZLIB)

# flex + bison
if (UNIX)
    include(FindFLEX)
    include(FindBISON)

    if (NOT FLEX_FOUND)
        message(FATAL_ERROR "FLEX is required for building")
    endif()

    if (NOT BISON_FOUND)
        message(FATAL_ERROR "Bison is required for building")
    endif()
elseif(WIN32)
    TODO_NOTIMPL("Download win-bison and use that")
endif()

# boehm gc

find_package(BDWgc CONFIG)
if (BDWgc_FOUND)
    list(APPEND ASY_STATIC_LIBARIES BDWgc::gc BDWgc::gccpp)
    list(APPEND ASY_MACROS USEGC)
else()
    message(WARNING "BDWgc not found; compiling without gc")
endif()

# curses
set(CURSES_NEED_NCURSES TRUE)
find_package(Curses)
if (Curses_FOUND)
    list(APPEND ASYMPTOTE_INCLUDES ${CURSES_INCLUDE_DIRS})
    list(APPEND ASY_COMPILE_OPTS ${CURSES_CFLAGS})
    list(APPEND ASY_STATIC_LIBRARIES ${CURSES_LIBRARIES})

    list(APPEND ASY_MACROS HAVE_NCURSES_CURSES_H HAVE_LIBCURSES)
else()
    message(WARNING "curses not found; will compile without curses")
endif()

# libreadline
if (UNIX)
    include(FindPkgConfig)
    pkg_check_modules(readline IMPORTED_TARGET readline)

    if (readline_FOUND)
        list(APPEND ASY_STATIC_LIBARIES PkgConfig::readline)
        list(APPEND ASY_MACROS HAVE_LIBREADLINE)
    else ()
        message(WARNING "readline not found; will compile without libreadline")
    endif()

elseif(WIN32)
    TODO_NOTIMPL("Implement readline for windows")
else()
    message(FATAL_ERROR "Only supported on Unix or Win32 systems")
endif()
