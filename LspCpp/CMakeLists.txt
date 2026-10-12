cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0025)
    # detect Apple's Clang
    cmake_policy(SET CMP0025 NEW)
endif()
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

set(LSPCPP_VERSION "1.0.3" CACHE STRING "LspCpp package version. Keep this in sync with vcpkg.json.")

# Without this, paths are not relative in the sources list
cmake_policy(SET CMP0076 NEW)
project(lspcpp VERSION "${LSPCPP_VERSION}" LANGUAGES CXX C)
set(LIB_VERSION_STRING "${PROJECT_VERSION}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
SET(GOOGLETEST_VERSION "0.00")

# compile in RelWithDebInfo  mode by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(LSPCPP_DEBUG_POSTFIX d CACHE STRING "Debug library postfix.")
# Set LSPCPP_LIB_NAME for pkg-config lspcpp.pc. We cannot use the OUTPUT_NAME target
# property because it's not set by default.
set(LSPCPP_LIB_NAME lspcpp)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LSPCPP_LIB_NAME ${LSPCPP_LIB_NAME}${LSPCPP_DEBUG_POSTFIX})
endif ()

include(CMakeParseArguments)
# Joins arguments and places the results in ${result_var}.
function(join result_var)
    set(result "")
    foreach (arg ${ARGN})
        set(result "${result}${arg}")
    endforeach ()
    set(${result_var} "${result}" PARENT_SCOPE)
endfunction()
# Sets a cache variable with a docstring joined from multiple arguments:
#   set(<variable> <value>... CACHE <type> <docstring>...)
# This allows splitting a long docstring for readability.
function(set_verbose)
    # cmake_parse_arguments is broken in CMake 3.4 (cannot parse CACHE) so use
    # list instead.
    list(GET ARGN 0 var)
    list(REMOVE_AT ARGN 0)
    list(GET ARGN 0 val)
    list(REMOVE_AT ARGN 0)
    list(REMOVE_AT ARGN 0)
    list(GET ARGN 0 type)
    list(REMOVE_AT ARGN 0)
    join(doc ${ARGN})
    set(${var} ${val} CACHE ${type} ${doc})
endfunction()
###########################################################
# Options
###########################################################
function (option_if_not_defined name description default)
    if(NOT DEFINED ${name})
        option(${name} ${description} ${default})
    endif()
endfunction()

option_if_not_defined(LSPCPP_BUILD_SHARED_LIBRARY "Build lspcpp as a shared library" OFF)
option_if_not_defined(USE_SYSTEM_RAPIDJSON "Use system RapidJSON instead of the git submodule if exists" OFF)
option_if_not_defined(USE_SYSTEM_UTFCPP "Use system utfcpp" OFF)
option_if_not_defined(LSPCPP_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option_if_not_defined(LSPCPP_BUILD_MINIMAL_EXAMPLE "Build the Boost-free minimal stdio example" OFF)
option_if_not_defined(LSPCPP_BUILD_EXAMPLES "Build example applications" OFF)
option_if_not_defined(LSPCPP_BUILD_TESTS "Build and register tests with CTest" OFF)
option_if_not_defined(LSPCPP_BUILD_PERF_SMOKE "Build optional performance smoke tests" OFF)
option_if_not_defined(LSPCPP_PERF_WARNINGS_AS_ERRORS "Treat performance smoke warnings as test failures" OFF)
option_if_not_defined(LSPCPP_BUILD_FUZZER "Build fuzzer" OFF)
option_if_not_defined(LSPCPP_BUILD_WEBSOCKETS "Build websocket server" ON)
option_if_not_defined(LSPCPP_ASAN "Build lsp with address sanitizer" OFF)
option_if_not_defined(LSPCPP_MSAN "Build lsp with memory sanitizer" OFF)
option_if_not_defined(LSPCPP_TSAN "Build lsp with thread sanitizer" OFF)
option_if_not_defined(LSPCPP_INSTALL "Create lsp install target" OFF)
option_if_not_defined(LSPCPP_STANDALONE_ASIO "Use standalone asio" ON)
option_if_not_defined(LSPCPP_SUPPORT_BOEHM_GC "\
Enable support for Boehm GC. GC must be specified LSPCPP_GC_DOWNLOADED_ROOT, if downloaded or by
find_package(BDWgc CONFIG REQUIRED) or pkg_config." OFF)
option_if_not_defined(LSPCPP_USE_CPP17 "Use C++17 for compilation. Setting this to off requires boost-optional." ON)
option_if_not_defined(LSPCPP_USE_STATIC_CRT "Link against the static MSVC runtime library (/MT, /MTd) when building with MSVC." OFF)

if(LSPCPP_BUILD_TESTS)
    include(CTest)
    enable_testing()
    set(LSPCPP_BUILD_EXAMPLES ON)
endif()

set(LSPCPP_GC_DOWNLOADED_ROOT "" CACHE STRING "\
If using downloaded gc root from https://www.hboehm.info/gc/, extract its contents and set this setting to
where the tar.gz file is extracted to. This directory must be an absolute path.
If this setting is set, LspCpp will use downloaded GC regardless of whether GC from find_package or
pkg_config is available or not.
")
option_if_not_defined(LSPCPP_GC_STATIC "Compiling with static gc library. Only used if a custom GC root is given" OFF)
set(LSPCPP_WIN32_WINNT_VALUE "0x0A00" CACHE STRING
        "Value to specify for _WIN32_WINNT macro when compiling on windows. See
https://learn.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt?view=msvc-170"
)

set(LSPCPP_EXTERNAL_LIBS)

option_if_not_defined(
        USE_EXTERNAL_IXWEBSOCKET
        "Instead of using git submodules for ixwebsocket, use system or vcpkg dependency."
        OFF
)
option_if_not_defined(
        USE_EXTERNAL_ASIO
        "Instead of using git submodules for asio, use system or vcpkg dependency."
        OFF
)

###########################################################
# Boehm GC
###########################################################

if (LSPCPP_SUPPORT_BOEHM_GC AND NOT LSPCPP_GC_DOWNLOADED_ROOT)
    message(STATUS "Attempting to find BDWgc by find_package")
    find_package(BDWgc CONFIG)
    if (NOT BDWgc_FOUND)
        set(GC_USE_PKGCONFIG 1)
        # fallback to pkg-config
        message(STATUS "CMake config for BDWgc not found; falling back to pkg-config")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(gc REQUIRED IMPORTED_TARGET bdw-gc)
    endif()
endif()

###########################################################
# Directories
###########################################################
function (set_if_not_defined name value)
    if(NOT DEFINED ${name})
        set(${name} ${value} PARENT_SCOPE)
    endif()
endfunction()

set(LSPCPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_if_not_defined(LSPCPP_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

macro(lspcpp_set_target_options_with_nuget_pkg target id version)
    if (CMAKE_GENERATOR MATCHES "Visual Studio.*")
        if(EXISTS ${CMAKE_BINARY_DIR}/packages/${id}.${version}/build/${id}.targets)
            target_link_libraries(${target} PRIVATE ${CMAKE_BINARY_DIR}/packages/${id}.${version}/build/${id}.targets)
        elseif(EXISTS ${CMAKE_BINARY_DIR}/packages/${id}.${version}/build/native/${id}.targets)
            target_link_libraries(${target} PRIVATE ${CMAKE_BINARY_DIR}/packages/${id}.${version}/build/native/${id}.targets)
        else()
            message(FATAL_ERROR "Can't find target of ${id}.${version}")
        endif()
    else()
        message(FATAL_ERROR "NUGET package only use in Visual Studio")
    endif()

endmacro()

macro(INSTALL_NUGET id version)
    if (CMAKE_GENERATOR MATCHES "Visual Studio.*")
        unset(nuget_cmd)
        list(APPEND nuget_cmd install ${id} -Prerelease -Version ${version} -OutputDirectory ${CMAKE_BINARY_DIR}/packages)
        message("excute nuget install:${nuget_cmd}")
        execute_process(COMMAND nuget ${nuget_cmd} ENCODING AUTO)
    else()
        message(FATAL_ERROR "INSTALL_NUGET only use in Visual Studio")
    endif()

endmacro()
###########################################################
# Functions
###########################################################
function(lspcpp_set_target_options target)

    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)

    # Enable C++14/17 (Required)
    if (LSPCPP_USE_CPP17)
        set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
    else()
        set_property(TARGET ${target} PROPERTY CXX_STANDARD 14)
    endif()

    set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)

    if (CMAKE_GENERATOR MATCHES "Visual Studio.*")
        lspcpp_set_target_options_with_nuget_pkg(${target} boost 1.76.0.0)
        lspcpp_set_target_options_with_nuget_pkg(${target} boost_chrono-vc141 1.76.0.0)
        lspcpp_set_target_options_with_nuget_pkg(${target} boost_date_time-vc141 1.76.0.0)
        lspcpp_set_target_options_with_nuget_pkg(${target} boost_filesystem-vc141 1.76.0.0)
        lspcpp_set_target_options_with_nuget_pkg(${target} boost_program_options-vc141 1.76.0.0)
        lspcpp_set_target_options_with_nuget_pkg(${target} boost_system-vc141 1.76.0.0)
        lspcpp_set_target_options_with_nuget_pkg(${target} boost_thread-vc141 1.76.0.0)
    endif()

    # Enable all warnings
    if(MSVC)
        target_compile_options(${target} PRIVATE "-W4")
    else()
        target_compile_options(${target} PRIVATE "-Wall")
    endif()

    # Disable specific, pedantic warnings
    if(MSVC)
        target_compile_options(${target} PRIVATE
                "-D_CRT_SECURE_NO_WARNINGS"

                # Warnings from nlohmann/json headers.
                "/wd4267" # 'argument': conversion from 'size_t' to 'int', possible loss of data
                "/bigobj" # for visual studio 2022 x64 or later.
                )
    endif()

    # Add define for JSON library in use
    set_target_properties(${target} PROPERTIES
            COMPILE_DEFINITIONS "LSPCPP_JSON_${LSPCPP_JSON_LIBRARY_UPPER}=1"
            )

    # Treat all warnings as errors
    if(LSPCPP_WARNINGS_AS_ERRORS)
        if(MSVC)
            target_compile_options(${target} PRIVATE "/WX")
        else()
            target_compile_options(${target} PRIVATE "-Werror")
        endif()
    endif(LSPCPP_WARNINGS_AS_ERRORS)

    if(LSPCPP_ASAN)
        target_compile_options(${target} PUBLIC "-fsanitize=address")
        target_link_libraries(${target} PUBLIC "-fsanitize=address")
    elseif(LSPCPP_MSAN)
        target_compile_options(${target} PUBLIC "-fsanitize=memory")
        target_link_libraries(${target} PUBLIC "-fsanitize=memory")
    elseif(LSPCPP_TSAN)
        target_compile_options(${target} PUBLIC "-fsanitize=thread")
        target_link_libraries(${target} PUBLIC "-fsanitize=thread")
    endif()

    # Error on undefined symbols
    # if(NOT MSVC)
    #     target_compile_options(${target} PRIVATE "-Wl,--no-undefined")
    # endif()

    if(LSPCPP_STANDALONE_ASIO)
        target_compile_definitions(${target} PUBLIC LSPCPP_USE_STANDALONE_ASIO)
    endif()

    if (LSPCPP_SUPPORT_BOEHM_GC)
        if (LSPCPP_GC_DOWNLOADED_ROOT)
            message(STATUS "Using manually downloaded GC")
            target_include_directories(${target} PUBLIC ${LSPCPP_GC_DOWNLOADED_ROOT}/include)

            if (LSPCPP_GC_STATIC)
                target_compile_definitions(${target} PUBLIC GC_NOT_DLL)
            endif()
        else()
            if (NOT GC_USE_PKGCONFIG)
                message(STATUS "Using cmake config for locating gc")
                target_link_libraries(${target} PUBLIC BDWgc::gc)
                get_target_property(GC_INCLUDE_DIRS_BASE BDWgc::gc INTERFACE_INCLUDE_DIRECTORIES)
            else()
                message(STATUS "Using pkg-config for locating gc")
                target_link_libraries(${target} PUBLIC PkgConfig::gc)
                get_target_property(GC_INCLUDE_DIRS_BASE PkgConfig::gc INTERFACE_INCLUDE_DIRECTORIES)
            endif()

            if (NOT GC_INCLUDE_DIRS_BASE STREQUAL "GC_INCLUDE_DIRS_BASE-NOTFOUND")
                list(TRANSFORM GC_INCLUDE_DIRS_BASE APPEND /gc)
                target_include_directories(${target} PUBLIC ${GC_INCLUDE_DIRS_BASE})
            else()
                message(WARNING "Cannot find GC include directories; compilation may not work")
            endif()
        endif()
        target_compile_definitions(${target} PUBLIC LSPCPP_USEGC)
    endif()

    if (WIN32)
        target_compile_definitions(${target} PRIVATE _WIN32_WINNT=${LSPCPP_WIN32_WINNT_VALUE})
    endif()
endfunction()


# Libraries

if (MSVC AND LSPCPP_USE_STATIC_CRT)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

function(check_submodule dir)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
        message(WARNING "Missing submodule: ${dir}. Run `git submodule update --init --recursive`.")
    endif()
endfunction()


################################################
# ——— Optional Library ———
################################################
# bring in the tl::optional interface library

if (NOT LSPCPP_USE_CPP17)
    set(OPTIONAL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    check_submodule(third_party/optional)
    add_subdirectory(third_party/optional)
    list(APPEND LSPCPP_EXTERNAL_LIBS tl::optional)
endif ()

################################################
# ——— Add filesystem as INTERFACE target ———
################################################
if (NOT LSPCPP_USE_CPP17)
    set(GHC_FS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/filesystem/include)
    add_library(ghc_filesystem INTERFACE)
    # attach that include‑dir to our INTERFACE target
    target_include_directories(ghc_filesystem
            INTERFACE
            $<BUILD_INTERFACE:${GHC_FS_INCLUDE_DIRS}>
            $<INSTALL_INTERFACE:include>
    )
    list(APPEND LSPCPP_EXTERNAL_LIBS ghc_filesystem)
endif ()

################################################
# ——— Add Asio as INTERFACE target ———
################################################

if (LSPCPP_STANDALONE_ASIO)
    if (USE_EXTERNAL_ASIO)
        find_package(asio CONFIG)
        if (NOT asio_FOUND)
            # fallback to pkg-config
            message(STATUS "CMake config for asio not found; falling back to pkg-config")
            find_package(PkgConfig REQUIRED)
            pkg_check_modules(asio REQUIRED IMPORTED_TARGET asio)
            list(APPEND LSPCPP_EXTERNAL_LIBS PkgConfig::asio)
        else ()
            list(APPEND LSPCPP_EXTERNAL_LIBS asio::asio)
        endif ()
    else ()
        # threading (Asio needs this)
        find_package(Threads REQUIRED)
        add_library(asio_lib INTERFACE)
        # point at the standalone Asio headers
        set(ASIO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/asio/asio/include)
        # attach that include‑dir to our INTERFACE target
        target_include_directories(asio_lib
                INTERFACE
                $<BUILD_INTERFACE:${ASIO_INCLUDE_DIRS}>
                $<INSTALL_INTERFACE:include>         # if you ever install
        )
        # define ASIO_STANDALONE for any consumer
        target_compile_definitions(asio_lib INTERFACE ASIO_STANDALONE)
        # link OS threads
        target_link_libraries(asio_lib INTERFACE Threads::Threads)
        list(APPEND LSPCPP_EXTERNAL_LIBS asio_lib)
    endif ()
endif ()

#################################################
# ——— Add ixwebsocket ———
#################################################

if (USE_EXTERNAL_IXWEBSOCKET)
    find_package(Threads REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(ixwebsocket CONFIG REQUIRED)
    list(APPEND LSPCPP_EXTERNAL_LIBS ixwebsocket::ixwebsocket)
else ()
    add_subdirectory(third_party/ixwebsocket)
    list(APPEND LSPCPP_EXTERNAL_LIBS ixwebsocket)
endif ()

###########################################################
# boost library
###########################################################
if (NOT LSPCPP_STANDALONE_ASIO)
    if (CMAKE_GENERATOR MATCHES "Visual Studio.*")
        INSTALL_NUGET(boost 1.76.0.0)
        INSTALL_NUGET(boost_chrono-vc141 1.76.0.0)
        INSTALL_NUGET(boost_date_time-vc141 1.76.0.0)
        INSTALL_NUGET(boost_filesystem-vc141 1.76.0.0)
        INSTALL_NUGET(boost_program_options-vc141 1.76.0.0)
        INSTALL_NUGET(boost_system-vc141 1.76.0.0)
        INSTALL_NUGET(boost_thread-vc141 1.76.0.0)
    else ()
        find_package(Boost CONFIG)
        if (NOT Boost_FOUND)
            if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
                message(FATAL_ERROR "Can't find boost,lease build boost and install it or install boost with : brew install boost")
            elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
                message(FATAL_ERROR "Can't find boost,please build boost and install it. or install boost with : sudo apt-get install libboost-dev")
            else ()
                message(FATAL_ERROR "Boost not found. Please ensure boost is available for CMake.")
            endif ()
        endif ()
    endif ()
endif()


###########################################################
# JSON library
###########################################################
if(USE_SYSTEM_RAPIDJSON)
    set(RapidJSON_MIN_VERSION "1.1.0")
    find_package(RapidJSON ${RapidJSON_MIN_VERSION} QUIET)
    if(NOT DEFINED RapidJSON_INCLUDE_DIRS AND DEFINED RAPIDJSON_INCLUDE_DIRS)
        set(RapidJSON_INCLUDE_DIRS "${RAPIDJSON_INCLUDE_DIRS}")
    endif()
    if(TARGET rapidjson)
        list(APPEND LSPCPP_EXTERNAL_LIBS rapidjson)
    elseif(TARGET RapidJSON::RapidJSON)
        list(APPEND LSPCPP_EXTERNAL_LIBS RapidJSON::RapidJSON)
    endif()
endif()
if(NOT RapidJSON_FOUND)
    if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/rapidjson/include")
        message(STATUS "Using local RapidJSON")
        set(RapidJSON_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidjson/include")
    else()
        message(STATUS "Please initialize rapidJSON git submodule as currently installed version is to old:")
        message(STATUS "git submodule init && git submodule update")
        message(FATAL_ERROR "RapidJSON version is likely too old.")
    endif()
endif()


###########################################################
# Targets
###########################################################

if(LSPCPP_BUILD_SHARED_LIBRARY)
    set(LSPCPP_LIB_TYPE SHARED)
else()
    set(LSPCPP_LIB_TYPE STATIC)
endif()
add_library(lspcpp ${LSPCPP_LIB_TYPE})

set (LSP_INCLUDE_LIST ${LSPCPP_INCLUDE_DIR}
        ${RapidJSON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${ASIO_INCLUDE_DIRS}
        ${GHC_FS_INCLUDE_DIRS})
### Includes
target_include_directories(lspcpp
        PUBLIC
        $<BUILD_INTERFACE:${LSPCPP_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_include_directories(lspcpp SYSTEM
        PUBLIC
        $<BUILD_INTERFACE:${RapidJSON_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${ASIO_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${GHC_FS_INCLUDE_DIRS}>
        PRIVATE
        ${Boost_INCLUDE_DIRS})

target_link_libraries(
        lspcpp
        PUBLIC ${LSPCPP_EXTERNAL_LIBS}
)

# utfcpp
if(USE_SYSTEM_UTFCPP)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(UTFCPP QUIET utf8cpp)
    endif()
    if(NOT UTFCPP_FOUND)
        find_path(UTFCPP_INCLUDE_DIR NAMES utf8.h PATH_SUFFIXES utf8cpp utfcpp)
    else()
        set(UTFCPP_INCLUDE_DIR ${UTFCPP_INCLUDE_DIRS})
    endif()
    if(NOT UTFCPP_INCLUDE_DIR)
        message(FATAL_ERROR "System utfcpp not found. Install libutfcpp-devel or set USE_SYSTEM_UTFCPP=OFF.")
    endif()
    set(UTFCPP_INCLUDE_DIRS ${UTFCPP_INCLUDE_DIR})
else()
    set(UTFCPP_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utfcpp/source)
endif()

if(UTFCPP_INCLUDE_DIRS)
    get_filename_component(utf_include_realpath ${UTFCPP_INCLUDE_DIRS} REALPATH)
    # Don't add as SYSTEM if they are in CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES.
    # It would reorder the system search paths and cause issues with libstdc++'s
    # use of #include_next.
    if(NOT "${utf_include_realpath}" IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        target_include_directories(lspcpp SYSTEM PRIVATE ${UTFCPP_INCLUDE_DIRS})
    endif()
endif()

### Sources
set(JSONRPC_LIST
        src/jsonrpc/Context.cpp
        src/jsonrpc/Endpoint.cpp
        src/jsonrpc/GCThreadContext.cpp
        src/jsonrpc/message.cpp
        src/jsonrpc/MessageJsonHandler.cpp
        src/jsonrpc/RemoteEndPoint.cpp
        src/jsonrpc/RequestCancellation.cpp
        src/jsonrpc/Process.cpp
        src/jsonrpc/serializer.cpp
        src/jsonrpc/StreamMessageProducer.cpp
        src/jsonrpc/TcpServer.cpp
        src/jsonrpc/threaded_queue.cpp
)
set(LSPCPP_LIST
        src/lsp/initialize.cpp
        src/lsp/lsp.cpp
        src/lsp/lsp_diagnostic.cpp
        src/lsp/Markup.cpp
        src/lsp/ParentProcessWatcher.cpp
        src/lsp/ProtocolJsonHandler.cpp
        src/lsp/textDocument.cpp
        src/lsp/utils.cpp
        src/lsp/working_files.cpp
        src/lsp/uri.cpp
        src/lsp/path_mapping.cpp
        src/lsp/offset_encoding.cpp
        src/lsp/request_context.cpp
        )

if(LSPCPP_BUILD_WEBSOCKETS)
    set(JSONRPC_LIST
        ${JSONRPC_LIST}
        src/jsonrpc/WebSocketServer.cpp
    )
endif()

target_sources(lspcpp PRIVATE
        ${JSONRPC_LIST}
        ${LSPCPP_LIST})

### Compile options

lspcpp_set_target_options(lspcpp)
if (LSPCPP_USE_CPP17)
    target_compile_features(lspcpp PUBLIC cxx_std_17)
else()
    target_compile_features(lspcpp PUBLIC cxx_std_14)
endif()

set_target_properties(lspcpp PROPERTIES POSITION_INDEPENDENT_CODE 1)
set_target_properties(lspcpp PROPERTIES
        DEBUG_POSTFIX "${LSPCPP_DEBUG_POSTFIX}"

        # Workaround for Visual Studio 2017:
        # Ensure the .pdb is created with the same name and in the same directory
        # as the .lib. Newer VS versions already do this by default, but there is no
        # harm in setting it for those too. Ignored by other generators.
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMPILE_PDB_NAME "lspcpp"
        COMPILE_PDB_NAME_DEBUG "lspcpp${LSPCPP_DEBUG_POSTFIX}")
if(LSPCPP_BUILD_SHARED_LIBRARY)
    set_target_properties(lspcpp PROPERTIES
            VERSION   "${PROJECT_VERSION}"
            SOVERSION "${PROJECT_VERSION_MAJOR}")
endif()

# install
if(LSPCPP_INSTALL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/support/cmake/JoinPaths.cmake)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    set(targets_export_name lspcpp-targets)
    set(project_config ${PROJECT_BINARY_DIR}/lspcpp-config.cmake)
    set(version_config ${PROJECT_BINARY_DIR}/lspcpp-config-version.cmake)
    set(pkgconfig ${PROJECT_BINARY_DIR}/lspcpp.pc)

    set_verbose(LSPCPP_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE STRING
            "Installation directory for libraries, a relative path that "
            "will be joined to ${CMAKE_INSTALL_PREFIX} or an absolute path.")

    set_verbose(LSPCPP_PKGCONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/pkgconfig CACHE STRING
            "Installation directory for pkgconfig (.pc) files, a relative "
            "path that will be joined with ${CMAKE_INSTALL_PREFIX} or an "
            "absolute path.")

    set_verbose(LSPCPP_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/lspcpp CACHE STRING
            "Installation directory for cmake files, a relative path that "
            "will be joined with ${CMAKE_INSTALL_PREFIX} or an absolute "
            "path.")

    configure_package_config_file(
            ${PROJECT_SOURCE_DIR}/support/cmake/lspcpp-config.cmake.in
            ${project_config}
            INSTALL_DESTINATION ${LSPCPP_CMAKE_DIR} )
    write_basic_package_version_file(
            ${version_config}
            VERSION ${LIB_VERSION_STRING}
            COMPATIBILITY SameMajorVersion
    )
    join_paths(libdir_for_pc_file "\${exec_prefix}" "${LSPCPP_LIB_DIR}")
    join_paths(includedir_for_pc_file "\${prefix}" "${CMAKE_INSTALL_INCLUDEDIR}")

    configure_file(
            "${PROJECT_SOURCE_DIR}/support/cmake/lspcpp.pc.in"
            "${pkgconfig}"
            @ONLY)

    if(TARGET asio_lib)
        install(TARGETS asio_lib
                EXPORT ${targets_export_name})
    endif()

    install(DIRECTORY ${LSPCPP_INCLUDE_DIR}/LibLsp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            USE_SOURCE_PERMISSIONS
            )

    install(TARGETS lspcpp
            EXPORT ${targets_export_name}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            )

    install(EXPORT ${targets_export_name}
            NAMESPACE lspcpp::
            DESTINATION ${LSPCPP_CMAKE_DIR}
            )
    install(
            FILES ${project_config} ${version_config}
            DESTINATION ${LSPCPP_CMAKE_DIR}
    )
    install(FILES "${pkgconfig}" DESTINATION "${LSPCPP_PKGCONFIG_DIR}")
endif()

# OS libraries
###########################################################
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(LSPCPP_OS_LIBS WS2_32)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(LSPCPP_OS_LIBS pthread)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(LSPCPP_OS_LIBS)
endif()

# tests
if(LSPCPP_BUILD_TESTS)
    function(lspcpp_add_test_executable target source)
        add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/tests/${source}")
        target_include_directories(${target} PRIVATE
                ${LSP_INCLUDE_LIST}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        lspcpp_set_target_options(${target})
        target_link_libraries(${target} PRIVATE lspcpp)
    endfunction()

    lspcpp_add_test_executable(lspcpp_jsonrpc_tests jsonrpc_tests.cpp)
    lspcpp_add_test_executable(lspcpp_stream_message_producer_tests stream_message_producer_tests.cpp)
    lspcpp_add_test_executable(lspcpp_remote_endpoint_tests remote_endpoint_tests.cpp)
    lspcpp_add_test_executable(lspcpp_endpoint_serializer_tests endpoint_serializer_tests.cpp)
    lspcpp_add_test_executable(lspcpp_infrastructure_tests infrastructure_tests.cpp)
    lspcpp_add_test_executable(lspcpp_working_files_tests working_files_tests.cpp)
    lspcpp_add_test_executable(lspcpp_text_document_sync_tests text_document_sync_tests.cpp)
    lspcpp_add_test_executable(lspcpp_lsp_types_tests lsp_types_roundtrip_tests.cpp)
    lspcpp_add_test_executable(lspcpp_protocol_json_handler_tests protocol_json_handler_tests.cpp)
    lspcpp_add_test_executable(lspcpp_clangd_fixture_tests clangd_fixture_tests.cpp)
    lspcpp_add_test_executable(lspcpp_uri_path_mapping_tests uri_path_mapping_tests.cpp)
    lspcpp_add_test_executable(lspcpp_api_compat_tests api_compat_tests.cpp)
    lspcpp_add_test_executable(lspcpp_language_session_tests language_session_tests.cpp)
    lspcpp_add_test_executable(lspcpp_lsp_3_18_tests lsp_3_18_tests.cpp)
    lspcpp_add_test_executable(lspcpp_lsp_3_16_17_tests lsp_3_16_17_tests.cpp)
    lspcpp_add_test_executable(lspcpp_tcp_write_queue_tests tcp_write_queue_tests.cpp)
    target_link_libraries(lspcpp_tcp_write_queue_tests PRIVATE ${LSPCPP_OS_LIBS})
    if(LSPCPP_BUILD_WEBSOCKETS)
        lspcpp_add_test_executable(lspcpp_websocket_write_queue_tests websocket_write_queue_tests.cpp)
        target_link_libraries(lspcpp_websocket_write_queue_tests PRIVATE ${LSPCPP_OS_LIBS})
    endif()
    if(LSPCPP_BUILD_PERF_SMOKE)
        lspcpp_add_test_executable(lspcpp_perf_smoke_tests perf_smoke_tests.cpp)
        if(LSPCPP_PERF_WARNINGS_AS_ERRORS)
            target_compile_definitions(lspcpp_perf_smoke_tests PRIVATE LSPCPP_PERF_WARNINGS_AS_ERRORS)
        endif()
    endif()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_test(NAME lspcpp.lsp_metamodel_coverage
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_lsp_metamodel_coverage.py
            --repo-root ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(lspcpp.lsp_metamodel_coverage PROPERTIES TIMEOUT 60)

    add_test(NAME lspcpp.lspgen_smoke
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/lspgen.py
            --repo-root ${CMAKE_CURRENT_SOURCE_DIR}
            --method window/showDocument)
    set_tests_properties(lspcpp.lspgen_smoke PROPERTIES TIMEOUT 60)

    add_test(NAME lspcpp.jsonrpc
            COMMAND $<TARGET_FILE:lspcpp_jsonrpc_tests>)
    add_test(NAME lspcpp.stream_message_producer
            COMMAND $<TARGET_FILE:lspcpp_stream_message_producer_tests>)
    add_test(NAME lspcpp.remote_endpoint
            COMMAND $<TARGET_FILE:lspcpp_remote_endpoint_tests>)
    add_test(NAME lspcpp.endpoint_serializer
            COMMAND $<TARGET_FILE:lspcpp_endpoint_serializer_tests>)
    add_test(NAME lspcpp.infrastructure
            COMMAND $<TARGET_FILE:lspcpp_infrastructure_tests>)
    add_test(NAME lspcpp.working_files
            COMMAND $<TARGET_FILE:lspcpp_working_files_tests>)
    add_test(NAME lspcpp.text_document_sync
            COMMAND $<TARGET_FILE:lspcpp_text_document_sync_tests>)
    add_test(NAME lspcpp.lsp_types
            COMMAND $<TARGET_FILE:lspcpp_lsp_types_tests>)
    add_test(NAME lspcpp.protocol_json_handler
            COMMAND $<TARGET_FILE:lspcpp_protocol_json_handler_tests>)
    add_test(NAME lspcpp.clangd_fixture
            COMMAND $<TARGET_FILE:lspcpp_clangd_fixture_tests>)
    add_test(NAME lspcpp.uri_path_mapping
            COMMAND $<TARGET_FILE:lspcpp_uri_path_mapping_tests>)
    add_test(NAME lspcpp.api_compat
            COMMAND $<TARGET_FILE:lspcpp_api_compat_tests>)
    add_test(NAME lspcpp.language_session
            COMMAND $<TARGET_FILE:lspcpp_language_session_tests>)
    add_test(NAME lspcpp.lsp_3_18
            COMMAND $<TARGET_FILE:lspcpp_lsp_3_18_tests>)
    add_test(NAME lspcpp.lsp_3_16_17
            COMMAND $<TARGET_FILE:lspcpp_lsp_3_16_17_tests>)
    add_test(NAME lspcpp.tcp_write_queue
            COMMAND $<TARGET_FILE:lspcpp_tcp_write_queue_tests>)
    set_tests_properties(lspcpp.tcp_write_queue PROPERTIES TIMEOUT 30)
    if(LSPCPP_BUILD_WEBSOCKETS)
        add_test(NAME lspcpp.websocket_write_queue
                COMMAND $<TARGET_FILE:lspcpp_websocket_write_queue_tests>)
        set_tests_properties(lspcpp.websocket_write_queue PROPERTIES TIMEOUT 30)
    endif()
    if(LSPCPP_BUILD_PERF_SMOKE)
        add_test(NAME lspcpp.perf_smoke
                COMMAND $<TARGET_FILE:lspcpp_perf_smoke_tests>)
        set_tests_properties(lspcpp.perf_smoke PROPERTIES TIMEOUT 120)
    endif()

    set(LSPCPP_CONSUMER_SMOKE_BINARY_DIR "${CMAKE_BINARY_DIR}/lspcpp_consumer_smoke")
    set(LSPCPP_CONSUMER_SMOKE_CONFIGURE_ARGS
            -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/consumer_project
            -B ${LSPCPP_CONSUMER_SMOKE_BINARY_DIR}
            -DLSPCPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DLSPCPP_USE_CPP17=${LSPCPP_USE_CPP17}
            -DLSPCPP_STANDALONE_ASIO=${LSPCPP_STANDALONE_ASIO}
            -DUSE_EXTERNAL_ASIO=${USE_EXTERNAL_ASIO}
            -DUSE_EXTERNAL_IXWEBSOCKET=${USE_EXTERNAL_IXWEBSOCKET}
            -DUSE_SYSTEM_RAPIDJSON=${USE_SYSTEM_RAPIDJSON}
            )
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND LSPCPP_CONSUMER_SMOKE_CONFIGURE_ARGS
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
        # The consumer project has no vcpkg manifest of its own, so point it at
        # the dependencies already provisioned for this build tree and disable
        # manifest install. Otherwise the vcpkg toolchain falls back to classic
        # mode and cannot locate manifest-installed packages (e.g. asio).
        if(DEFINED VCPKG_INSTALLED_DIR)
            set(LSPCPP_CONSUMER_VCPKG_INSTALLED_DIR "${VCPKG_INSTALLED_DIR}")
        else()
            set(LSPCPP_CONSUMER_VCPKG_INSTALLED_DIR "${CMAKE_BINARY_DIR}/vcpkg_installed")
        endif()
        list(APPEND LSPCPP_CONSUMER_SMOKE_CONFIGURE_ARGS
                -DVCPKG_INSTALLED_DIR=${LSPCPP_CONSUMER_VCPKG_INSTALLED_DIR}
                -DVCPKG_MANIFEST_INSTALL=OFF)
    endif()
    add_test(NAME lspcpp.consumer_cmake_configure
            COMMAND ${CMAKE_COMMAND} ${LSPCPP_CONSUMER_SMOKE_CONFIGURE_ARGS})
    set_tests_properties(lspcpp.consumer_cmake_configure PROPERTIES TIMEOUT 120)
    add_test(NAME lspcpp.consumer_cmake_build
            COMMAND ${CMAKE_COMMAND} --build ${LSPCPP_CONSUMER_SMOKE_BINARY_DIR}
                    --target consumer_smoke MinimalStdIOServerExample)
    set_tests_properties(lspcpp.consumer_cmake_build PROPERTIES
            DEPENDS lspcpp.consumer_cmake_configure
            TIMEOUT 120)
    add_test(NAME lspcpp.consumer_smoke_run
            COMMAND ${LSPCPP_CONSUMER_SMOKE_BINARY_DIR}/consumer_smoke)
    set_tests_properties(lspcpp.consumer_smoke_run PROPERTIES
            DEPENDS lspcpp.consumer_cmake_build
            TIMEOUT 30)
endif()

# examples
if(LSPCPP_BUILD_MINIMAL_EXAMPLE OR LSPCPP_BUILD_EXAMPLES)
    function(build_minimal_example target)
        add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/examples/${target}.cpp")
        target_include_directories(${target} PRIVATE ${LSP_INCLUDE_LIST})
        set_target_properties(${target} PROPERTIES
                FOLDER "Examples"
                )
        lspcpp_set_target_options(${target})
        target_link_libraries(${target} PRIVATE lspcpp ${LSPCPP_OS_LIBS})
    endfunction(build_minimal_example)

    build_minimal_example(MinimalStdIOServerExample)

    if(LSPCPP_BUILD_TESTS)
        add_test(NAME lspcpp.minimal_stdio_example_build
                COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target MinimalStdIOServerExample)
        set_tests_properties(lspcpp.minimal_stdio_example_build PROPERTIES TIMEOUT 60)
    endif()
endif()

if(LSPCPP_BUILD_EXAMPLES)

    find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
    set(LSPCPP_EXAMPLE_LIBS
            ${LSPCPP_OS_LIBS}
            Boost::filesystem
            Boost::program_options
            Boost::system
            )

    function(build_example target)
        add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/examples/${target}.cpp")
        target_include_directories(${target} PRIVATE ${LSP_INCLUDE_LIST})
        set_target_properties(${target} PROPERTIES
                FOLDER "Examples"
                )
        lspcpp_set_target_options(${target})
        target_link_libraries(${target} PRIVATE lspcpp ${LSPCPP_EXAMPLE_LIBS})
    endfunction(build_example)

    set(EXAMPLES
            ProcessClientExample
            StdIOClientExample
            StdIOServerExample
            TcpServerExample
            )
    if(LSPCPP_BUILD_WEBSOCKETS)
        list(APPEND EXAMPLES WebsocketExample)
    endif()

    foreach (example ${EXAMPLES})
        build_example(${example})
    endforeach()

    if(LSPCPP_BUILD_TESTS)
        add_test(NAME lspcpp.tcp_server_example
                COMMAND $<TARGET_FILE:TcpServerExample>)
        set_tests_properties(lspcpp.tcp_server_example PROPERTIES TIMEOUT 30)

        add_test(NAME lspcpp.stdio_client_server_example
                COMMAND $<TARGET_FILE:StdIOClientExample> --execPath $<TARGET_FILE:StdIOServerExample>)
        set_tests_properties(lspcpp.stdio_client_server_example PROPERTIES TIMEOUT 30)

        add_test(NAME lspcpp.process_client_example_build
                COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ProcessClientExample StdIOServerExample)
        set_tests_properties(lspcpp.process_client_example_build PROPERTIES TIMEOUT 60)

        add_test(NAME lspcpp.process_client_example
                COMMAND $<TARGET_FILE:ProcessClientExample> --execPath $<TARGET_FILE:StdIOServerExample>)
        set_tests_properties(lspcpp.process_client_example PROPERTIES
                DEPENDS lspcpp.process_client_example_build
                TIMEOUT 30)

        if(LSPCPP_BUILD_WEBSOCKETS)
            add_test(NAME lspcpp.websocket_example_build
                    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target WebsocketExample)
            set_tests_properties(lspcpp.websocket_example_build PROPERTIES TIMEOUT 60)

            add_test(NAME lspcpp.websocket_example
                    COMMAND $<TARGET_FILE:WebsocketExample>)
            set_tests_properties(lspcpp.websocket_example PROPERTIES
                    DEPENDS lspcpp.websocket_example_build
                    TIMEOUT 30)
        endif()
    endif()
endif()

# Add a distclean target to the Makefile
ADD_CUSTOM_TARGET(distclean COMMAND ${CMAKE_COMMAND} -P  ${PROJECT_SOURCE_DIR}/support/cmake/distclean.cmake)
