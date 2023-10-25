
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

# flex + bison
if (UNIX)
    include(FindFLEX)
    include(FindBISON)

    if (NOT FLEX_FOUND)
        message(FATAL_ERROR "FLEX is required for building")
    endif()

    if (NOT BISON_FOUND)
        message(FATAL_ERROR "Bison is required for building")
    endif()
elseif(WIN32)
    TODO_NOTIMPL("Download win-bison and use that")
endif()

# boehm gc

find_package(BDWgc CONFIG)
if (BDWgc_FOUND)
    list(APPEND ASY_STATIC_LIBARIES BDWgc::gc BDWgc::gccpp)
    list(APPEND ASY_MACROS USEGC)
else()
    message(WARNING "BDWgc not found; compiling without gc")
endif()

# libreadline
