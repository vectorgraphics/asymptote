cmake_minimum_required(VERSION 3.26)

include(pkg-info.cmake)

project(${ASY_PACKAGE_NAME})

include(cmake-scripts/options.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED 17)


include(cmake-scripts/basic-parameters.cmake)
include(cmake-scripts/buildfiles-to-src.cmake)

include(cmake-scripts/asy-files.cmake)

# setting build files
build_files_to_src(CORE_BUILD_FILES CORE_SOURCE_FILES)

include(cmake-scripts/generated-files.cmake)
include(cmake-scripts/external-libs.cmake)

# PRC
add_subdirectory(prc/)
list(APPEND ASY_STATIC_LIBARIES PRC)


# defining asy target
add_executable(
        asy
        ${CORE_SOURCE_FILES}
        ${ASYMPTOTE_GENERATED_HEADERS}
        ${ASY_GENERATED_BUILD_SOURCES}
        ${ASY_SRC_DIR}/main.cc
)

include(cmake-scripts/gnu-install-macros.cmake)
include(cmake-scripts/asy-macro.cmake)

target_include_directories(asy
    PUBLIC ${ASYMPTOTE_INCLUDES}
)

target_link_libraries(
        asy
        PUBLIC ${ASY_STATIC_LIBARIES}
)

target_compile_definitions(
        asy
        PUBLIC ${ASY_MACROS}
)
