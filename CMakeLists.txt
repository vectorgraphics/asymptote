cmake_minimum_required(VERSION 3.26)

include(cmake-scripts/common.cmake)
include(pkg-info.cmake)
include(cmake-scripts/options.cmake)

project(${ASY_PACKAGE_NAME})

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(cmake-scripts/basic-parameters.cmake)
include(cmake-scripts/buildfiles-to-src.cmake)

include(cmake-scripts/asy-files.cmake)

# setting build files
build_files_to_src(CORE_BUILD_FILES CORE_SOURCE_FILES)
include(cmake-scripts/external-libs.cmake)

# PRC
add_subdirectory(prc/)
list(APPEND ASY_STATIC_LIBARIES PRC)

get_target_property(prc_incl_dir PRC INCLUDE_DIRECTORIES)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:PRC,INCLUDE_DIRECTORIES>)

include(cmake-scripts/gnu-install-macros.cmake)
include(cmake-scripts/asy-macro.cmake)

include(cmake-scripts/generated-files.cmake)

if (WIN32)
    include(cmake-scripts/win32-specific.cmake)
endif()


# defining asy target

add_library(
        asycore
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
        asy ${ASY_SRC_DIR}/main.cc
)

target_link_libraries(asy PUBLIC asycore)

# base files
include(cmake-scripts/asy-base-files.cmake)

# asy + base files
add_custom_target(asy-basefiles DEPENDS ${ASY_OUTPUT_BASE_FILES})
add_custom_target(asy-with-basefiles ALL DEPENDS asy asy-basefiles)
