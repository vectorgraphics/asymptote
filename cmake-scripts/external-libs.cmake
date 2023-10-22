
if (ENABLE_LSP)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/LspCpp)
list(APPEND ASY_STATIC_LIBARIES lspcpp)

else()
    # only include lsp libraries
    list(APPEND ASYMPTOTE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/LspCpp/include)
endif()

find_package(tinyexr CONFIG REQUIRED)
list(APPEND ASY_STATIC_LIBARIES unofficial::tinyexr::tinyexr)

# zlib
find_package(ZLIB REQUIRED)
list(APPEND ASY_STATIC_LIBARIES ZLIB::ZLIB)
