# asy-base-cpp-plugins.cmake
#
# Builds the C++ asybind plugins shipped under base/collections/ (e.g.
# hashset_core.cc -> libhashset_core.so). Each is a MODULE library that
# the asy interpreter loads at runtime via `from collections.X access ...`.
#
# Mirrors the autotools pattern in Makefile.in:
#   BASE_PLUGIN_CC_FILES = $(wildcard base/collections/*.cc)
#   base/collections/lib%.so: base/collections/%.cc
#       $(CXX) -std=$(CXX_STANDARD) -fPIC -shared -O2 -g \
#           -Iasybind/include -o $@ $<
#
# The compiled .so files are placed under ${ASY_BUILD_BASE_DIR}/collections/
# so an in-tree run (`./asy -dir build/base ...`) can load them, and are
# added to ASY_OUTPUT_BASE_FILES so asy-basefiles / asy-with-basefiles
# pick them up automatically. They are installed alongside the
# corresponding .asy wrappers.

set(ASY_BASE_COLLECTIONS_CPP_SRC_DIR ${ASY_SOURCE_BASE_DIR}/collections)
set(ASY_BASE_COLLECTIONS_BUILD_DIR ${ASY_BUILD_BASE_DIR}/collections)

file(GLOB ASY_BASE_COLLECTIONS_CC_FILES CONFIGURE_DEPENDS
    ${ASY_BASE_COLLECTIONS_CPP_SRC_DIR}/*.cc
)

set(ASY_BASE_CPP_PLUGIN_TARGETS "")

foreach(_plugin_cc ${ASY_BASE_COLLECTIONS_CC_FILES})
    get_filename_component(_plugin_name ${_plugin_cc} NAME_WE)
    set(_target asy-plugin-collections-${_plugin_name})

    add_library(${_target} MODULE ${_plugin_cc})

    set_target_properties(${_target} PROPERTIES
        OUTPUT_NAME    ${_plugin_name}
        PREFIX         "lib"
        SUFFIX         ".so"
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY  ${ASY_BASE_COLLECTIONS_BUILD_DIR}
    )

    target_include_directories(${_target} PRIVATE
        ${ASY_SRC_DIR}/asybind/include
    )

    # Keep dependencies minimal: the plugin only sees the asybind SDK.
    # No -DFOR_SHARED, no asycore link — the SDK is header-only and the
    # host API is supplied at load time by the asy interpreter.

    list(APPEND ASY_OUTPUT_BASE_FILES
        ${ASY_BASE_COLLECTIONS_BUILD_DIR}/lib${_plugin_name}.so
    )
    list(APPEND ASY_BASE_CPP_PLUGIN_TARGETS ${_target})
endforeach()

# This file is included() (not add_subdirectory'd) by CMakeLists.txt, so
# the variables above are already visible to the caller without
# PARENT_SCOPE.
