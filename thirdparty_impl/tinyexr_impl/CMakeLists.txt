# Implementation library for tinyexr

cmake_minimum_required(VERSION 3.26)
project(tinyexr-impl)

# fetch content
include(FetchContent)
FetchContent_Declare(
        tinyexr
        GIT_REPOSITORY https://github.com/syoyo/tinyexr
        GIT_TAG v1.0.7
)
FetchContent_Populate(tinyexr)

add_library(tinyexr-impl src/tinyexr_impl.cc)
target_include_directories(
        tinyexr-impl PUBLIC ${tinyexr_SOURCE_DIR}
)

# zlib
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    target_compile_definitions(tinyexr-impl PRIVATE HAVE_ZLIB)
else()
    message(WARNING "zlib not found; will use miniz")
endif()

target_link_libraries(tinyexr-impl PRIVATE ZLIB::ZLIB)
