cmake_minimum_required(VERSION 3.26)

project(asy-cudareflect CXX CUDA)

add_executable(
    reflect
    main.cc
    EXRFiles.cc
    kernel.cu ReflectanceMapper.cu
)

# C++17 standard is derived from CMakeLists.txt.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FindCUDAToolkit REQUIRED)

target_include_directories(reflect PRIVATE ${CMAKE_CURRENT_LIST_DIR})

set(ASY_REFLECT_LIBRARIES tinyexr-impl glm::glm CUDA::cudart)

if (WIN32)
    list(APPEND ASY_REFLECT_LIBRARIES unofficial::getopt-win32::getopt)
endif()

target_link_libraries(
    reflect
    PRIVATE ${ASY_REFLECT_LIBRARIES}
)