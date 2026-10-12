cmake_minimum_required(VERSION 3.15)
project(lspcpp_consumer_smoke LANGUAGES CXX)

if(NOT DEFINED LSPCPP_SOURCE_DIR)
    message(FATAL_ERROR "LSPCPP_SOURCE_DIR must point to the LspCpp source tree")
endif()

set(LSPCPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LSPCPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LSPCPP_BUILD_MINIMAL_EXAMPLE ON CACHE BOOL "" FORCE)
set(LSPCPP_USE_STATIC_CRT ON CACHE BOOL "" FORCE)

add_subdirectory("${LSPCPP_SOURCE_DIR}" lspcpp)

add_executable(consumer_smoke main.cpp)
target_link_libraries(consumer_smoke PRIVATE lspcpp)
