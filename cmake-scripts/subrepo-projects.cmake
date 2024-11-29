set(LSP_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/LspCpp)

if (ENABLE_LSP)
    message(STATUS "LSP Enabled.")
    # disable New Boost version warning
    set(Boost_NO_WARN_NEW_VERSIONS 1)
    set(USE_SYSTEM_RAPIDJSON ON CACHE INTERNAL "Use system rapidjson")
    set(LSPCPP_USE_CPP17 ON CACHE INTERNAL "C++17 mode")
    set(LSPCPP_SUPPORT_BOEHM_GC ON CACHE INTERNAL "Use boehm GC")
    # For transitive URI dependency
    set(Uri_BUILD_DOCS OFF CACHE INTERNAL "build docs for uri")
    set(Uri_BUILD_TESTS OFF CACHE INTERNAL "build tests for uri")

    add_subdirectory(${LSP_REPO_ROOT})
    list(APPEND ASY_STATIC_LIBARIES lspcpp)
    list(APPEND ASY_MACROS HAVE_LSP=1)
else()
    # only include lsp libraries
    message(STATUS "LSP Disabled. Will not have language server protocol support.")
    list(APPEND ASYMPTOTE_INCLUDES ${LSP_REPO_ROOT}/include)
endif()

set(TINYEXR_SUBREPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tinyexr)
