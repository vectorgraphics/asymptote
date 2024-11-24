# Implementation library for tinyexr

cmake_minimum_required(VERSION 3.27)
project(tinyexr-impl)

# fetch content

set(TINYEXR_REPO_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../../tinyexr" CACHE PATH "Tinyexr repo location")
add_library(tinyexr-impl src/tinyexr_impl.cc)
target_include_directories(
        tinyexr-impl PUBLIC ${TINYEXR_REPO_LOCATION}
)

# zlib
find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
    target_compile_definitions(tinyexr-impl PRIVATE HAVE_ZLIB)
else()
    message(WARNING "zlib not found; will use miniz")
endif()

target_link_libraries(tinyexr-impl PRIVATE ZLIB::ZLIB)
