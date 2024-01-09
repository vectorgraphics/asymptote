# Implementation library for vulkan memory allocator

cmake_minimum_required(VERSION 3.26)
project(vk-mem-allocator-impl)

# fetch content
include(FetchContent)
FetchContent_Declare(
        vk-mem-allocator
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
        GIT_TAG master
)
FetchContent_Populate(vk-mem-allocator)

add_library(vk-mem-allocator-impl src/vma_impl.cc src/vma_cxx.cc)

target_include_directories(
        vk-mem-allocator-impl PUBLIC
            include
            ${vk-mem-allocator_SOURCE_DIR}/include
)

include(FindVulkan)
if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found")
endif()
target_link_libraries(vk-mem-allocator-impl PRIVATE Vulkan::Vulkan)
