cmake_minimum_required(VERSION 3.26)
project(PRC)

add_library(PRC
        ${CMAKE_CURRENT_LIST_DIR}/oPRCFile.cc
        ${CMAKE_CURRENT_LIST_DIR}/PRCbitStream.cc
        ${CMAKE_CURRENT_LIST_DIR}/PRCdouble.cc
        ${CMAKE_CURRENT_LIST_DIR}/writePRC.cc
)

target_include_directories(
        PRC
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
)
