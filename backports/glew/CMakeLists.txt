CMAKE_MINIMUM_REQUIRED(VERSION 3.26)

project(GLEW)

add_library(GLEW STATIC ${CMAKE_CURRENT_LIST_DIR}/src/glew.c)
target_include_directories(GLEW PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

if (WIN32)
    target_compile_definitions(GLEW PUBLIC GLEW_STATIC)
endif()
