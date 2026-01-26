cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0025)
    # detect Apple's Clang
    cmake_policy(SET CMP0025 NEW)
endif()
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

set(LIB_MAJOR_VERSION "1")
set(LIB_MINOR_VERSION "0")
set(LIB_PATCH_VERSION "0")
set(LIB_TWEAK_VERSION "0")
set(LIB_VERSION_STRING "${LIB_MAJOR_VERSION}.${LIB_MINOR_VERSION}.${LIB_PATCH_VERSION}.${LIB_TWEAK_VERSION}")

# Without this, paths are not relative in the sources list
cmake_policy(SET CMP0076 NEW)
project(lspcpp VERSION "${LIB_VERSION_STRING}" LANGUAGES CXX C)

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

option_if_not_defined(USE_SYSTEM_RAPIDJSON "Use system RapidJSON instead of the git submodule if exists" OFF)
option_if_not_defined(LSPCPP_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option_if_not_defined(LSPCPP_BUILD_EXAMPLES "Build example applications" OFF)
option_if_not_defined(LSPCPP_BUILD_FUZZER "Build fuzzer" OFF)
option_if_not_defined(LSPCPP_BUILD_WEBSOCKETS "Build websocket server" ON)
option_if_not_defined(LSPCPP_ASAN "Build lsp with address sanitizer" OFF)
option_if_not_defined(LSPCPP_MSAN "Build lsp with memory sanitizer" OFF)
option_if_not_defined(LSPCPP_TSAN "Build lsp with thread sanitizer" OFF)
option_if_not_defined(LSPCPP_INSTALL "Create lsp install target" OFF)
option_if_not_defined(LSPCPP_STANDALONE_ASIO "Use standalone asio" OFF)
option_if_not_defined(LSPCPP_SUPPORT_BOEHM_GC "\
Enable support for Boehm GC. GC must be specified LSPCPP_GC_DOWNLOADED_ROOT, if downloaded or by
find_package(BDWgc CONFIG REQUIRED) or pkg_config." OFF)
option_if_not_defined(LSPCPP_USE_CPP17 "Use C++17 for compilation. Setting this to off requires boost-optional." ON)

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

if (MSVC)
    set(Uri_USE_STATIC_CRT OFF)
endif()
set(Uri_BUILD_TESTS OFF)
add_subdirectory(third_party/uri)

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
    check_submodule(third_party/optional)
    add_subdirectory(third_party/optional)
    list(APPEND LSPCPP_EXTERNAL_LIBS tl::optional)
endif ()

################################################
# ——— Add filesystem as INTERFACE target ———
################################################
if (NOT LSPCPP_USE_CPP17)
    add_library(ghc_filesystem INTERFACE)
    # point at the ghc filesystem headers
    set(GHC_FS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/filesystem/include)
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
        find_package(Boost REQUIRED)
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
endif()
if(NOT RapidJSON_FOUND)
    if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/rapidjson/include")
        message(STATUS "Using local RapidJSON")
        set(RapidJSON_INCLUDE_DIRS third_party/rapidjson/include)
    else()
        message(STATUS "Please initialize rapidJSON git submodule as currently installed version is to old:")
        message(STATUS "git submodule init && git submodule update")
        message(FATAL_ERROR "RapidJSON version is likely too old.")
    endif()
endif()


###########################################################
# Targets
###########################################################

# lsp
add_library(lspcpp STATIC)
set (LSP_INCLUDE_LIST ${LSPCPP_INCLUDE_DIR}
        ${RapidJSON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${ASIO_INCLUDE_DIRS}
        ${GHC_FS_INCLUDE_DIRS}
        ${Uri_SOURCE_DIR}/include)
### Includes
target_include_directories(lspcpp PRIVATE ${LSP_INCLUDE_LIST})

target_link_libraries(
        lspcpp
        PUBLIC ${LSPCPP_EXTERNAL_LIBS} network-uri
)

set(LSPCPP_THIRD_PARTY_DIR_LIST
        ${LSPCPP_THIRD_PARTY_DIR}/utfcpp/source
        )

foreach(include_dir  ${LSPCPP_THIRD_PARTY_DIR_LIST})
    get_filename_component(include_dir_realpath ${include_dir} REALPATH)
    # Don't add as SYSTEM if they are in CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES.
    # It would reorder the system search paths and cause issues with libstdc++'s
    # use of #include_next.
    if(NOT "${include_dir_realpath}" IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        target_include_directories(lspcpp SYSTEM PRIVATE ${include_dir})
    endif()
endforeach()

### Sources
set(JSONRPC_LIST
        src/jsonrpc/Context.cpp
        src/jsonrpc/Endpoint.cpp
        src/jsonrpc/GCThreadContext.cpp
        src/jsonrpc/message.cpp
        src/jsonrpc/MessageJsonHandler.cpp
        src/jsonrpc/RemoteEndPoint.cpp
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

    install(TARGETS network-uri
            EXPORT NetworkURITargets
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT NetworkURITargets
            FILE NetworkURITargets.cmake
            DESTINATION lib/cmake/NetworkURI)

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

# examples
if(LSPCPP_BUILD_EXAMPLES)

    ###########################################################
    # OS libraries
    ###########################################################
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(LSPCPP_OS_LIBS WS2_32)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(LSPCPP_OS_LIBS pthread)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(LSPCPP_OS_LIBS)
    endif()

    function(build_example target)
        add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/examples/${target}.cpp")
        target_include_directories(${target} PRIVATE ${LSP_INCLUDE_LIST})
        set_target_properties(${target} PROPERTIES
                FOLDER "Examples"
                )
        lspcpp_set_target_options(${target})
        target_link_libraries(${target}  PRIVATE lspcpp "${LSPCPP_OS_LIBS}")
    endfunction(build_example)

    set(EXAMPLES
            StdIOClientExample
            StdIOServerExample
            TcpServerExample
            WebsocketExample
            )

    foreach (example ${EXAMPLES})
        build_example(${example})
    endforeach()
endif()

# Add a distclean target to the Makefile
ADD_CUSTOM_TARGET(distclean COMMAND ${CMAKE_COMMAND} -P  ${PROJECT_SOURCE_DIR}/support/cmake/distclean.cmake)
