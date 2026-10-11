
# renderer-shared-libs.cmake
#
# Builds the renderer shared libraries (libasyvulkan and optionally
# libasyopengl) for both Unix (.so) and Windows (.dll). The main asy
# binary carries zero link-time Vulkan/OpenGL dependencies; instead it
# calls dlopen("libasyvulkan.so") on Unix or LoadLibrary("libasyvulkan.dll")
# on Windows at runtime via rendererloader.cc. This mirrors the autotools
# SHIMLIBS model.
#
# On Unix, included from linux-install.cmake (ASY_BASE_INSTALL_COMPONENT
# and ASY_INSTALL_SYSDIR_VALUE are already defined).
# On Windows, included from CMakeLists.txt directly (before the NSIS
# installer step); DLLs are collected from the build dir by the installer.

# ── libasyvulkan (libasyvulkan.so / libasyvulkan.dll) ───────────────────────

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

    set_target_properties(asyvulkan PROPERTIES
        OUTPUT_NAME asyvulkan
        PREFIX "lib"          # libasyvulkan.so / libasyvulkan.dll
        POSITION_INDEPENDENT_CODE ON
    )

    # On Unix, build into the assembled base directory so the runtime loader
    # finds it via the Asymptote search path. On Windows, the DLL is placed in
    # the build output dir and collected by the NSIS installer.
    if (UNIX)
        set_target_properties(asyvulkan PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${ASY_BUILD_BASE_DIR}
        )
    endif()

    target_include_directories(asyvulkan PRIVATE
        $<TARGET_PROPERTY:asycore,INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:vk-mem-allocator-impl,INCLUDE_DIRECTORIES>
    )

    target_compile_definitions(asyvulkan PRIVATE ${ASY_MACROS})
    target_compile_options(asyvulkan PRIVATE ${ASY_COMPILE_OPTS})

    target_link_libraries(asyvulkan PRIVATE
        Vulkan::Vulkan
        glslang::glslang
        glslang::SPIRV
        glfw
    )

    add_dependencies(asyvulkan asy_gen_headers)

    if (UNIX)
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
endif()

# ── libasyopengl (libasyopengl.so / libasyopengl.dll) ───────────────────────

if (ENABLE_OPENGL)
    add_library(asyopengl SHARED
        ${ASY_SRC_DIR}/glrender.cc
        ${ASY_SRC_DIR}/shaders.cc
        ${ASY_SRC_DIR}/GLTextures.cc
        ${ASY_SRC_DIR}/sortTriangles.cc
    )

    set_target_properties(asyopengl PROPERTIES
        OUTPUT_NAME asyopengl
        PREFIX "lib"
        POSITION_INDEPENDENT_CODE ON
    )

    if (UNIX)
        set_target_properties(asyopengl PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${ASY_BUILD_BASE_DIR}
        )
    endif()

    target_include_directories(asyopengl PRIVATE
        $<TARGET_PROPERTY:asycore,INCLUDE_DIRECTORIES>
    )

    target_compile_definitions(asyopengl PRIVATE ${ASY_MACROS})
    target_compile_options(asyopengl PRIVATE ${ASY_COMPILE_OPTS})

    if (WIN32)
        target_link_libraries(asyopengl PRIVATE
            OpenGL::GL
            GLEW
            glfw
        )
    else()
        target_link_libraries(asyopengl PRIVATE
            OpenGL::GL
            glfw
        )
    endif()

    add_dependencies(asyopengl asy_gen_headers)

    if (UNIX)
        add_dependencies(asy-with-basefiles asyopengl)

        install(
            TARGETS asyopengl
            LIBRARY
                DESTINATION ${ASY_INSTALL_SYSDIR_VALUE}
                COMPONENT   ${ASY_BASE_INSTALL_COMPONENT}
                PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_EXECUTE
                            WORLD_READ WORLD_EXECUTE
        )
    endif()
endif()

