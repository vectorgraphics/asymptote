cmake_minimum_required(VERSION 3.27)
project(PRC)

add_library(PRC
        ${CMAKE_CURRENT_LIST_DIR}/oPRCFile.cc
        ${CMAKE_CURRENT_LIST_DIR}/PRCbitStream.cc
        ${CMAKE_CURRENT_LIST_DIR}/PRCdouble.cc
        ${CMAKE_CURRENT_LIST_DIR}/writePRC.cc
)
find_package(ZLIB REQUIRED)

if (CMAKE_CXX_BYTE_ORDER EQUAL BIG_ENDIAN)
        target_compile_definitions(PRC PUBLIC WORDS_BIGENDIAN=1)
elseif (NOT CMAKE_CXX_BYTE_ORDER)  # only on apple platforms
        message(STATUS "Targeting multiple platforms with different endianness")
        target_compile_definitions(PRC PUBLIC UNIVERSAL_ENDIAN_BUILD)
endif()

target_include_directories(
        PRC
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(
        PRC
        PRIVATE ZLIB::ZLIB
)
