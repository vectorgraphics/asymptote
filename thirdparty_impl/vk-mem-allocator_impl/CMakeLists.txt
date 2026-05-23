# Implementation library for vulkan memory allocator

cmake_minimum_required(VERSION 3.26)
project(vk-mem-allocator-impl)

# fetch content
set(
        VMA_REPO_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/../../VulkanMemoryAllocator" CACHE PATH
        "VulkanMemoryAllocator repo location"
)

add_library(vk-mem-allocator-impl src/vma_impl.cc src/vma_cxx.cc)

target_include_directories(
        vk-mem-allocator-impl PUBLIC
            include
            ${VMA_REPO_LOCATION}/include
)

include(FindVulkan)
if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan not found")
endif()
target_link_libraries(vk-mem-allocator-impl PRIVATE Vulkan::Vulkan)
