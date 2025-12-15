set(ASY_SUBREPO_CLONE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(LSP_REPO_ROOT ${ASY_SUBREPO_CLONE_ROOT}/LspCpp)
set(TINYEXR_SUBREPO_ROOT ${ASY_SUBREPO_CLONE_ROOT}/tinyexr)
set(BOEHM_GC_ROOT ${ASY_SUBREPO_CLONE_ROOT}/gc)
set(LIBATOMIC_OPS_ROOT ${ASY_SUBREPO_CLONE_ROOT}/libatomic_ops)
set(HIGHWAYHASH_ROOT ${ASY_SUBREPO_CLONE_ROOT}/highwayhash)

# highwayhash
set(OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "highwayhash shared libs flag")
add_subdirectory(${HIGHWAYHASH_ROOT})
unset(BUILD_SHARED_LIBS CACHE)
set(BUILD_SHARED_LIBS ${OLD_BUILD_SHARED_LIBS})
list(APPEND ASY_STATIC_LIBARIES highwayhash)

# boehm gc
if (ENABLE_GC)
    set(enable_gpl OFF CACHE INTERNAL "libatomicops gpl libs option")
    add_subdirectory(${LIBATOMIC_OPS_ROOT})

    set(OLD_CFLAG_EXTRA ${CFLAG_EXTRA})
    set(CFLAGS_EXTRA -I${LIBATOMIC_OPS_ROOT}/src)  # for bdwgc

    set(OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "bdwgc shared libs flag")
    set(enable_cplusplus ON CACHE INTERNAL "bdwgc enable C++")
    set(without_libatomic_ops ON CACHE INTERNAL "bdwgc use libatomic ops")
    add_subdirectory(${BOEHM_GC_ROOT})

    set(CFLAG_EXTRA ${OLD_CFLAG_EXTRA})
    unset(BUILD_SHARED_LIBS CACHE)
    set(BUILD_SHARED_LIBS ${OLD_BUILD_SHARED_LIBS})

    list(APPEND ASY_STATIC_LIBARIES gc gccpp atomic_ops)

    if (WIN32)
        list(APPEND ASY_MACROS GC_NOT_DLL)
    endif()
    # We use #include <gc.h> as opposed to <gc/gc.h> (and also for other gc include files) to allow
    # linking directly to the compiled source for testing different GC versions.

    # In GC tarballs downloaded from https://www.hboehm.info/gc/, the header files are in include/gc.h, and not
    # include/gc/gc.h, hence we need a way to allow inclusion of "gc.h". In vcpkg gc distributions, the include
    # files are provided in include/gc/gc.h (and other files). Hence we append "/gc" to the include directories.

    if (WIN32)
        list(APPEND ASY_STATIC_LIBARIES gctba)
    endif()
    list(APPEND ASY_MACROS USEGC)
else()
    message(STATUS "Disabling gc support")
endif()

if (ENABLE_LSP)
    message(STATUS "LSP Enabled.")
    # disable New Boost version warning
    set(Boost_NO_WARN_NEW_VERSIONS 1)
    set(USE_SYSTEM_RAPIDJSON ON CACHE INTERNAL "Use system rapidjson")
    set(LSPCPP_USE_CPP17 ON CACHE INTERNAL "C++17 mode")
    # For transitive URI dependency
    set(Uri_BUILD_DOCS OFF CACHE INTERNAL "build docs for uri")
    set(Uri_BUILD_TESTS OFF CACHE INTERNAL "build tests for uri")

    if (WIN32)
        set(LSPCPP_WIN32_WINNT_VALUE ${ASY_WIN32_WINVER_VERSION} CACHE INTERNAL "lsp win32 winver value")
    endif()

    if (ENABLE_GC)
        set(LSPCPP_SUPPORT_BOEHM_GC ON CACHE INTERNAL "Use boehm GC")
        set(LSPCPP_GC_DOWNLOADED_ROOT ${BOEHM_GC_ROOT} CACHE INTERNAL "gc root for lsp")
        set(LSPCPP_GC_STATIC ON CACHE INTERNAL "lsp use static gc")
    endif()

    add_subdirectory(${LSP_REPO_ROOT})

    list(APPEND ASY_STATIC_LIBARIES lspcpp)
    list(APPEND ASY_MACROS HAVE_LSP=1)
else()
    # only include lsp libraries
    message(STATUS "LSP Disabled. Will not have language server protocol support.")
    list(APPEND ASYMPTOTE_INCLUDES ${LSP_REPO_ROOT}/include)
endif()
