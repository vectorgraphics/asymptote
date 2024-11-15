cmake_minimum_required(VERSION 3.27)

include(CMakeDependentOption)

cmake_dependent_option(
        GLEW_ENABLE_OFFSCREEN
        "enable offscreen rendering for glew. Only applies to Unix systems."
        false "UNIX" false)

project(GLEW)

add_library(GLEW STATIC ${CMAKE_CURRENT_LIST_DIR}/src/glew.c)
target_include_directories(GLEW PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

if (WIN32)
    target_compile_definitions(GLEW PUBLIC GLEW_STATIC)
endif()

if (GLEW_ENABLE_OFFSCREEN)
    target_compile_definitions(GLEW PUBLIC GLEW_OSMESA)
endif()
