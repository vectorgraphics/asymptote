cmake_minimum_required(VERSION 3.26)
project(Asymptote)

set(CMAKE_CXX_STANDARD_REQUIRED 17)

set(ASYMPTOTE_INCLUDES include)
set(ASYMPTOTE_FILES
        main

add_executable(
        asy
)

target_include_directories(asy
    PUBLIC ${ASYMPTOTE_INCLUDES})
