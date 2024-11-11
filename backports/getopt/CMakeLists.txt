cmake_minimum_required(VERSION 3.26)

project(getopt)

add_library(
        getopt
        ${CMAKE_CURRENT_LIST_DIR}/src/getopt.c
        ${CMAKE_CURRENT_LIST_DIR}/src/getopt1.c
)

target_include_directories(
        getopt
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)


# we are requring string.h here

include(CheckIncludeFile)

check_include_file("string.h" HAVE_STRING_HEADER_FILE)

if (HAVE_STRING_HEADER_FILE)
    target_compile_definitions(
            getopt
            PRIVATE HAVE_STRING_H=1
    )
endif()
