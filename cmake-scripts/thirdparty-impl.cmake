# tinyexr

set(TINYEXR_REPO_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr CACHE INTERNAL "tinyexr repo location")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_impl/tinyexr_impl/)

list(APPEND ASY_STATIC_LIBARIES tinyexr-impl)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:tinyexr-impl,INCLUDE_DIRECTORIES>)

if (ENABLE_VULKAN)
    add_subdirectory(${THIRDPARTY_IMPL_ROOT}/vk-mem-allocator_impl)
    list(APPEND ASY_STATIC_LIBARIES vk-mem-allocator-impl)
    list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:vk-mem-allocator-impl,INCLUDE_DIRECTORIES>)
endif()
