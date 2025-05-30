cmake_minimum_required(VERSION 3.27)

include(cmake-scripts/common.cmake)
include(asy-pkg-version-suffix.cmake OPTIONAL RESULT_VARIABLE ASY_ADDR_VERSION_SUFFIX_FILE)
include(cmake-scripts/options.cmake)
include(pkg-info.cmake)
include(cmake-scripts/vcpkg-features.cmake)

project(${ASY_PACKAGE_NAME})

include(cmake-scripts/compiler-config.cmake)
include(cmake-scripts/basic-parameters.cmake)
include(cmake-scripts/buildfiles-to-src.cmake)
include(cmake-scripts/asy-files.cmake)

# setting build files
if (WIN32)
    include(cmake-scripts/win32-specific.cmake)
endif()

build_files_to_src(CORE_BUILD_FILES CORE_SOURCE_FILES)
include(cmake-scripts/subrepo-projects.cmake)
include(cmake-scripts/external-libs.cmake)
include(cmake-scripts/thirdparty-impl.cmake)
include(cmake-scripts/backport-libs.cmake)

# PRC
add_subdirectory(prc/)
list(APPEND ASY_STATIC_LIBARIES PRC)

get_target_property(prc_incl_dir PRC INCLUDE_DIRECTORIES)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:PRC,INCLUDE_DIRECTORIES>)

list(APPEND ASYMPTOTE_INCLUDES "LspCpp/include")

include(cmake-scripts/gnu-install-macros.cmake)
include(cmake-scripts/asy-macro.cmake)

include(cmake-scripts/generated-files.cmake)

# asy-reflect
if (ENABLE_CUDA_ASY_REFLECT)
    message(STATUS "asy reflect target enabled")
    add_subdirectory(cudareflect/)
endif()

# defining asy target

add_library(
        asycore STATIC
        ${CORE_SOURCE_FILES}
        ${ASY_GENERATED_BUILD_SOURCES}
)

target_include_directories(asycore PUBLIC ${ASYMPTOTE_INCLUDES})
add_dependencies(asycore asy_gen_headers)
target_link_libraries(asycore PUBLIC ${ASY_STATIC_LIBARIES})
target_compile_definitions(asycore PUBLIC ${ASY_MACROS})
target_compile_options(asycore PUBLIC ${ASY_COMPILE_OPTS})

# asy executable

add_executable(
        asy ${ASY_SRC_DIR}/main.cc ${ASY_WIN_RC_FILE}
)

target_link_libraries(asy PUBLIC asycore)

# base files
include(cmake-scripts/asy-base-files.cmake)

# asy + base files
add_custom_target(asy-basefiles DEPENDS ${ASY_OUTPUT_BASE_FILES})
add_custom_target(asy-with-basefiles ALL DEPENDS asy asy-basefiles)

# asy misc files
if (ENABLE_MISCFILES_GEN)
    include(cmake-scripts/asy-misc-files.cmake)
endif()

# documentation generation

if (ENABLE_DOCGEN)
    include(cmake-scripts/docgen.cmake)
endif()

# testing

enable_testing()
include(cmake-scripts/tests-asy.cmake)
include(cmake-scripts/tests-wce.cmake)

if (ENABLE_ASY_CXXTEST)
    add_subdirectory(cxxtests)
endif()

# installation
if (WIN32)
    # on windows, pre-NSIS
    include(cmake-scripts/win32-pre-nsis-installer.cmake)
else()
    # on unix systems
    if (NOT CTAN_BUILD)
        include(cmake-scripts/linux-install.cmake)
    endif()
endif()
