set(THIRDPARTY_IMPL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_impl)

# tinyexr
set(TINYEXR_REPO_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr CACHE INTERNAL "tinyexr repo location")
add_subdirectory(${THIRDPARTY_IMPL_ROOT}/tinyexr_impl/)

# LTO on all subdirectory targets (matches the autotools build, which passes
# -flto through CFLAGS to every subproject). ASY_LTO_COMPILE_OPTIONS is only
# set on the non-MSVC (-flto) path; MSVC already propagates LTO here via
# CMAKE_INTERPROCEDURAL_OPTIMIZATION.
if (ASY_LTO_COMPILE_OPTIONS)
    target_compile_options(tinyexr-impl PRIVATE ${ASY_LTO_COMPILE_OPTIONS})
endif()

list(APPEND ASY_STATIC_LIBRARIES tinyexr-impl)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:tinyexr-impl,INCLUDE_DIRECTORIES>)

if (ENABLE_VULKAN)
    set(VULKAN_MEM_ALLOC_REPO_LOCATION
            ${CMAKE_CURRENT_SOURCE_DIR}/VulkanMemoryAllocator CACHE INTERNAL
            "VulkanMemoryAllocator repo location")
    add_subdirectory(${THIRDPARTY_IMPL_ROOT}/vk-mem-allocator_impl)

    # LTO on all subdirectory targets (matches the autotools build, which passes
    # -flto through CFLAGS to every subproject). A diagnostic build with VMA
    # under LTO confirmed the forked 3D render child (-noV) still works, so the
    # original blanket LTO exclusion of subdirectory targets was a red herring.
    # ASY_LTO_COMPILE_OPTIONS is only set on the non-MSVC (-flto) path; MSVC
    # already propagates LTO here via CMAKE_INTERPROCEDURAL_OPTIMIZATION.
    if (ASY_LTO_COMPILE_OPTIONS)
        target_compile_options(vk-mem-allocator-impl PRIVATE ${ASY_LTO_COMPILE_OPTIONS})
    endif()

    if (WIN32)
        list(APPEND ASY_STATIC_LIBRARIES vk-mem-allocator-impl)
    endif()
    list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:vk-mem-allocator-impl,INCLUDE_DIRECTORIES>)
endif()
