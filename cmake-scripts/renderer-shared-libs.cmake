# renderer-shared-libs.cmake
#
# Builds libasyvulkan.so (and optionally libasyopengl.so) for Unix platforms.
# On Unix the main asy binary carries zero link-time Vulkan/OpenGL dependencies;
# instead it calls dlopen("libasyvulkan.so") at runtime via rendererloader.cc.
# This mirrors the autotools SHIMLIBS / "libasyvulkan.so" target.
#
# Included from linux-install.cmake, so ASY_BASE_INSTALL_COMPONENT and
# ASY_INSTALL_SYSDIR_VALUE are already defined.

if (NOT UNIX OR WIN32)
    return()
endif()

# ── libasyvulkan.so ──────────────────────────────────────────────────────────

if (ENABLE_VULKAN)
    # The renderer links the glslang:: imported targets (glslang::glslang,
    # glslang::SPIRV). The main asy binary pulls glslang in via
    # find_package(Vulkan COMPONENTS glslang) (-> Vulkan::glslang), which does
    # not define the glslang:: targets, so request the standalone glslang CONFIG
    # package here. It resolves from the same vcpkg tree already on the prefix
    # path.
    find_package(glslang CONFIG REQUIRED)

    # Source set mirrors the autotools VULKAN_OBJS. EXRFiles, renderBase and the
    # other shared renderer code are compiled into asycore (the main binary) and
    # resolved from there at dlopen time via the binary's exported symbols (see
    # ENABLE_EXPORTS on the asy target), so they are deliberately NOT compiled
    # into the library a second time.
    add_library(asyvulkan SHARED
        ${ASY_SRC_DIR}/vkrender.cc
        ${ASY_SRC_DIR}/vkdispatchstorage.cc
        ${ASY_SRC_DIR}/vkutils.cc
        ${ASY_SRC_DIR}/vulkanshim.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_impl/vk-mem-allocator_impl/src/vma_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty_impl/vk-mem-allocator_impl/src/vma_cxx.cc
    )

    # CMake automatically adds the "lib" prefix and ".so" suffix → libasyvulkan.so
    #
    # Build the library into the assembled base directory (the same place the
    # base/*.asy files are staged). At runtime the loader finds it through the
    # Asymptote search path via settings::locateFile; placing it alongside the
    # base files makes `asy -dir <base>` resolve it from any working directory,
    # mirroring the installed layout (where it lands in the sysdir next to base).
    # Otherwise it would only be found when the current directory happened to be
    # the build root.
    set_target_properties(asyvulkan PROPERTIES
        OUTPUT_NAME asyvulkan
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${ASY_BUILD_BASE_DIR}
    )

    target_include_directories(asyvulkan PRIVATE
        # Pull in asycore's full include set (includes GC headers from the gc subrepo target).
        $<TARGET_PROPERTY:asycore,INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:vk-mem-allocator-impl,INCLUDE_DIRECTORIES>
    )

    # Same macro set as asycore (includes HAVE_LIBVULKAN, HAVE_LIBGLFW, etc.)
    # No -DFOR_SHARED: the renderer is not a Python bindings library.
    target_compile_definitions(asyvulkan PRIVATE ${ASY_MACROS})
    target_compile_options(asyvulkan PRIVATE ${ASY_COMPILE_OPTS})

    target_link_libraries(asyvulkan PRIVATE
        Vulkan::Vulkan
        glslang::glslang
        glslang::SPIRV
        glfw
    )

    # Generated headers (e.g. asy-keywords.h) must exist before compilation.
    add_dependencies(asyvulkan asy_gen_headers)

    # Pull libasyvulkan.so into the default ALL build via asy-with-basefiles.
    add_dependencies(asy-with-basefiles asyvulkan)

    install(
        TARGETS asyvulkan
        LIBRARY
            DESTINATION ${ASY_INSTALL_SYSDIR_VALUE}
            COMPONENT   ${ASY_BASE_INSTALL_COMPONENT}
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
    )
endif()
