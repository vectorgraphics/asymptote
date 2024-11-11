cmake_minimum_required(VERSION 3.26)

project(OptionalBackport)

add_library(OptionalBackport INTERFACE)
target_include_directories(OptionalBackport INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include/)