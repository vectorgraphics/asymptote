set(THIRDPARTY_IMPL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_impl)

# tinyexr
set(TINYEXR_REPO_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr CACHE INTERNAL "tinyexr repo location")
add_subdirectory(${THIRDPARTY_IMPL_ROOT}/tinyexr_impl/)

list(APPEND ASY_STATIC_LIBARIES tinyexr-impl)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:tinyexr-impl,INCLUDE_DIRECTORIES>)

if (ENABLE_VULKAN)
    set(VULKAN_MEM_ALLOC_REPO_LOCATION
            ${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator CACHE INTERNAL
            "VulkanMemoryAllocator repo location")
    add_subdirectory(${THIRDPARTY_IMPL_ROOT}/vk-mem-allocator_impl)
    list(APPEND ASY_STATIC_LIBARIES vk-mem-allocator-impl)
    list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:vk-mem-allocator-impl,INCLUDE_DIRECTORIES>)
endif()
