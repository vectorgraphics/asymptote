include(FindPkgConfig)
include(FetchContent)

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

# glm; mandatory for all builds
find_package(glm CONFIG)
if (glm_FOUND)
    list(APPEND ASY_STATIC_LIBARIES glm::glm)
    list(APPEND ASY_MACROS HAVE_LIBGLM)
else()
    message(FATAL_ERROR "glm not found; will not use glm")
endif()

if (ENABLE_READLINE)
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
            message(FATAL_ERROR "curses not found; will compile without curses")
        endif()

        pkg_check_modules(readline IMPORTED_TARGET readline)

        if (readline_FOUND)
            list(APPEND ASY_STATIC_LIBARIES PkgConfig::readline)
            list(APPEND ASY_MACROS HAVE_LIBREADLINE)
        else ()
            message(FATAL_ERROR "readline not found; will compile without libreadline")
        endif()
    elseif(WIN32)
        find_package(unofficial-pdcurses CONFIG)
        if (unofficial-pdcurses_FOUND)
            list(APPEND ASY_STATIC_LIBRARIES unofficial::pdcurses::pdcurses)
            list(APPEND ASY_MACROS HAVE_CURSES_H HAVE_LIBCURSES)
        else()
            message(FATAL_ERROR "curses not found; will compile without curses")
        endif()

        find_package(unofficial-readline-win32 CONFIG)
        if (unofficial-readline-win32_FOUND)
            list(APPEND ASY_STATIC_LIBARIES unofficial::readline-win32::readline)
            list(APPEND ASY_MACROS HAVE_LIBREADLINE)
        else ()
            message(FATAL_ERROR "readline not found; will compile without libreadline")
        endif()
    else()
        message(FATAL_ERROR "Only supported on Unix or Win32 systems")
    endif()
else()
    message(STATUS "libreadline disabled; will not use libreadline")
endif()

# libcurl
if (ENABLE_CURL)
    find_package(CURL)
    if (CURL_FOUND)
        list(APPEND ASY_STATIC_LIBARIES CURL::libcurl)
        list(APPEND ASY_MACROS HAVE_LIBCURL)
    else()
        message(FATAL_ERROR "curl not found")
    endif()
else()
    message(STATUS "Disabling curl support")
endif()

# pthreads
if (ENABLE_THREADING)
    if (UNIX)
        set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
        set(THREADS_PREFER_PTHREAD_FLAG TRUE)
        include(FindThreads)

        if(CMAKE_USE_PTHREADS_INIT)
            list(APPEND ASY_STATIC_LIBARIES Threads::Threads)
            list(APPEND ASY_MACROS HAVE_PTHREAD=1)
        else()
            message(FATAL_ERROR "No thread library specified")
        endif()
    elseif(WIN32)
        find_package(PThreads4W)

        if(PThreads4W_FOUND)
            list(APPEND ASY_STATIC_LIBARIES PThreads4W::PThreads4W)
            list(APPEND ASY_MACROS HAVE_PTHREAD=1)
        else()
            message(FATAL_ERROR "No thread library specified")
        endif()
    else()
        message(FATAL_ERROR "Only supported on Unix or Win32 systems")
    endif()
else()
    message(STATUS "Disabling threading support")
endif()

# gsl
if (ENABLE_GSL)
    find_package(GSL)
    if (GSL_FOUND)
        list(APPEND ASY_STATIC_LIBARIES GSL::gsl)
        list(APPEND ASY_MACROS HAVE_LIBGSL)
    else()
        message(FATAL_ERROR "GSL not found")
    endif()
else()
    message(STATUS "Disabling gsl support")
endif()


# eigen
if (ENABLE_EIGEN3)
find_package(Eigen3 CONFIG)
    if (Eigen3_FOUND)
        list(APPEND ASY_STATIC_LIBARIES Eigen3::Eigen)
        list(APPEND ASY_MACROS HAVE_EIGEN_DENSE)
    else()
        message(FATAL_ERROR "eigen3 not found")
    endif()
else()
    message(STATUS "Disabling eigen3 support")
endif()

# Vulkan stuff
if (ENABLE_VULKAN)
    message(STATUS "If a warning about Vulkan::glslang comes up about missing debug configuration,
that warning can be safely ignored. We are not using glslang from the vulkan package.
We are using a separate glslang package
    ")

    find_package(Vulkan COMPONENTS glslang)
    if (Vulkan_FOUND AND Vulkan_glslang_FOUND)
        list(APPEND ASY_STATIC_LIBARIES Vulkan::Vulkan Vulkan::glslang)
        list(APPEND ASY_MACROS HAVE_LIBVULKAN)
    else()
        message(FATAL_ERROR "Vulkan not found")
    endif()

    find_package(glfw3 CONFIG)
    if (glfw3_FOUND)
        list(APPEND ASY_STATIC_LIBARIES glfw)
    else()
        message(FATAL_ERROR "glfw3 not found")
    endif()

    if (ENABLE_VK_VALIDATION_LAYERS)
        list(APPEND ASY_MACROS ENABLE_VK_VALIDATION)
    endif()
else()
    message(STATUS "Disabling vulkan support")
endif()


if (ENABLE_RPC_FEATURES)
    if(UNIX)
        pkg_check_modules(TIRPC REQUIRED IMPORTED_TARGET libtirpc)
        list(APPEND ASY_STATIC_LIBARIES PkgConfig::TIRPC)
    endif()

    if (WIN32)
        # win32 does not have native open_memstream support
        set(OLD_BUILD_TESTING ${BUILD_TESTING})
        set(BUILD_TESTING OFF CACHE INTERNAL "build testing")
        FetchContent_Declare(
                fmem
                GIT_REPOSITORY https://github.com/Kreijstal/fmem.git
                GIT_TAG 5f79fef3606be5dac54d62c7d0e2123363afabd7
        )
        FetchContent_MakeAvailable(fmem)
        set(BUILD_TESTING ${OLD_BUILD_TESTING} CACHE INTERNAL "build testing")

        list(APPEND ASY_STATIC_LIBARIES fmem)
        list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:fmem,INCLUDE_DIRECTORIES>)
    endif()
    list(APPEND ASY_MACROS HAVE_LIBTIRPC)


else()
    message(STATUS "Disabling rpc and xdr/v3d support")
endif()

# fftw3

if (ENABLE_FFTW3)
    set(FFTW3_USABLE TRUE)
    find_package(FFTW3 CONFIG)
    if (NOT FFTW3_FOUND)
        message(WARNING "libfftw3 not found; will not use fftw3")
        set(FFTW3_USABLE FALSE)
    endif()

    if (FFTW3_USABLE)
        list(APPEND ASY_STATIC_LIBARIES FFTW3::fftw3)
        list(APPEND ASY_MACROS HAVE_LIBFFTW3 FFTWPP_SINGLE_THREAD)
    else()
        message(FATAL_ERROR "environment lacks needed fftw3 features")
    endif()
else()
    message(STATUS "Disabling fftw3 support")
endif()
