
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
    if ((NOT WIN32_FLEX_BINARY) OR (NOT WIN32_BISON_BINARY))
        # downlod winflexbison
        message(STATUS "Flex or bison not given; downloading winflexbison.")
        include(FetchContent)

        FetchContent_Declare(
                winflexbison
                URL https://github.com/lexxmark/winflexbison/releases/download/v2.5.25/win_flex_bison-2.5.25.zip
                URL_HASH SHA256=8D324B62BE33604B2C45AD1DD34AB93D722534448F55A16CA7292DE32B6AC135
        )
        FetchContent_MakeAvailable(winflexbison)
        message(STATUS "Downloaded winflexbison")

        if (NOT WIN32_FLEX_BINARY)
            set(FLEX_EXECUTABLE ${winflexbison_SOURCE_DIR}/win_flex.exe)
        endif()

        if (NOT WIN32_BISON_BINARY)
            set(BISON_EXECUTABLE ${winflexbison_SOURCE_DIR}/win_bison.exe)
        endif()
    else()
        set(FLEX_EXECUTABLE ${WIN32_FLEX_BINARY})
        set(BISON_EXECUTABLE ${WIN32_BISON_BINARY})
    endif()
endif()

# getopt (win32 only)

if (WIN32)
    find_package(unofficial-getopt-win32 REQUIRED)
    list(APPEND ASY_STATIC_LIBARIES unofficial::getopt-win32::getopt)
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
if (UNIX)
    # we know ncurses work on unix systems, however
    # not always supported on windows (esp. msvc)
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
elseif(WIN32)
    find_package(unofficial-pdcurses CONFIG)
    if (unofficial-pdcurses_FOUND)
        list(APPEND ASY_STATIC_LIBRARIES unofficial::pdcurses::pdcurses)
        list(APPEND ASY_MACROS HAVE_CURSES_H HAVE_LIBCURSES)
    else()
        message(WARNING "curses not found; will compile without curses")
    endif()
else()
    message(FATAL_ERROR "Only supported on Unix or Win32 systems")
endif()

# libreadline

if (ENABLE_LIBREADLINE)
    include(FindPkgConfig)
    pkg_check_modules(readline IMPORTED_TARGET readline)

    if (readline_FOUND)
        list(APPEND ASY_STATIC_LIBARIES PkgConfig::readline)
        list(APPEND ASY_MACROS HAVE_LIBREADLINE)
    else ()
        message(WARNING "readline not found; will compile without libreadline")
    endif()
else()
    message(STATUS "libreadline disabled; will not use libreadline")
endif()

# libcurl

find_package(CURL)
if (CURL_FOUND)
    list(APPEND ASY_STATIC_LIBARIES CURL::libcurl)
    list(APPEND ASY_MACROS HAVE_LIBCURL)
else()
    message(WARNING "curl not found; will compile without curl")
endif()


# pthreads
if (UNIX)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    include(FindThreads)

    if(CMAKE_USE_PTHREADS_INIT)
        list(APPEND ASY_STATIC_LIBARIES Threads::Threads)
        list(APPEND ASY_MACROS HAVE_PTHREAD=1)
    else()
        message(WARNING "No thread library specified; will not use threads")
    endif()
elseif(WIN32)
    find_package(PThreads4W)

    if(PThreads4W_FOUND)
        list(APPEND ASY_STATIC_LIBARIES PThreads4W::PThreads4W)
        list(APPEND ASY_MACROS HAVE_PTHREAD=1)
    else()
        message(WARNING "No thread library specified; will not use threads")
    endif()
else()
    message(FATAL_ERROR "Only supported on Unix or Win32 systems")
endif()

# gsl
find_package(GSL)
if (GSL_FOUND)
    list(APPEND ASY_STATIC_LIBARIES GSL::gsl)
    list(APPEND ASY_MACROS HAVE_LIBGSL)
else()
    message(WARNING "GSL not found; will compile without gsl")
endif()


# eigen
find_package(Eigen3 CONFIG)
if (Eigen3_FOUND)
    list(APPEND ASY_STATIC_LIBARIES Eigen3::Eigen)
    list(APPEND ASY_MACROS HAVE_EIGEN_DENSE)
else()
    message(WARNING "eigen3 not found; will compile without eigen")
endif()
