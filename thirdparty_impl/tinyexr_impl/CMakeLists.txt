# Implementation library for tinyexr

cmake_minimum_required(VERSION 3.27)

# fetch content before project
# to not add the repo's CMakeLists.txt to the build system
# see https://cmake.org/cmake/help/latest/module/FetchContent.html#populating-content-without-adding-it-to-the-build
include(FetchContent)
FetchContent_Declare(
        tinyexr
        GIT_REPOSITORY https://github.com/syoyo/tinyexr
        GIT_TAG v1.0.7
)
FetchContent_MakeAvailable(tinyexr)

# set tinyexr-impl project
project(tinyexr-impl)
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
