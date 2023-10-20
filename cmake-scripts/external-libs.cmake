
if (ENABLE_LSP)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/LspCpp)
list(APPEND ASY_STATIC_LIBARIES lspcpp)

else()
    # only include lsp libraries
    list(APPEND ASYMPTOTE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/LspCpp/include)
endif()
