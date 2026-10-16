cmake_minimum_required(VERSION 3.27)


include(cmake-scripts/common.cmake)
include(asy-pkg-version-suffix.cmake OPTIONAL RESULT_VARIABLE ASY_ADDR_VERSION_SUFFIX_FILE)
include(cmake-scripts/options.cmake)
include(pkg-info.cmake)
include(cmake-scripts/vcpkg-features.cmake)

project(${ASY_PACKAGE_NAME})

include(cmake-scripts/check-host-processor-vars.cmake)

include(cmake-scripts/compiler-config.cmake)
include(cmake-scripts/basic-parameters.cmake)
include(cmake-scripts/buildfiles-to-src.cmake)
include(cmake-scripts/asy-files.cmake)

# setting build files
if (WIN32)
    include(cmake-scripts/win32-specific.cmake)
endif()

build_files_to_src(CORE_BUILD_FILES CORE_SOURCE_FILES)
include(cmake-scripts/subrepo-projects.cmake)
include(cmake-scripts/external-libs.cmake)
include(cmake-scripts/thirdparty-impl.cmake)

# PRC
add_subdirectory(prc/)

# LTO on all subdirectory targets (matches the autotools build, which passes
# -flto through CFLAGS to every subproject). ASY_LTO_COMPILE_OPTIONS is only
# set on the non-MSVC (-flto) path; MSVC already propagates LTO here via
# CMAKE_INTERPROCEDURAL_OPTIMIZATION.
if (ASY_LTO_COMPILE_OPTIONS)
    target_compile_options(PRC PRIVATE ${ASY_LTO_COMPILE_OPTIONS})
endif()

list(APPEND ASY_STATIC_LIBRARIES PRC)

get_target_property(prc_incl_dir PRC INCLUDE_DIRECTORIES)
list(APPEND ASYMPTOTE_INCLUDES $<TARGET_PROPERTY:PRC,INCLUDE_DIRECTORIES>)

list(APPEND ASYMPTOTE_INCLUDES "LspCpp/include")

include(cmake-scripts/gnu-install-macros.cmake)
include(cmake-scripts/asy-macro.cmake)

include(cmake-scripts/generated-files.cmake)

# asy-reflect
if (ENABLE_CUDA_ASY_REFLECT)
    message(STATUS "asy reflect target enabled")
    add_subdirectory(cudareflect/)
endif()

# defining asy target

add_library(
        asycore OBJECT
        ${CORE_SOURCE_FILES}
        ${ASY_GENERATED_BUILD_SOURCES}
)

target_include_directories(asycore PUBLIC ${ASYMPTOTE_INCLUDES})
add_dependencies(asycore asy_gen_headers)
target_link_libraries(asycore PUBLIC ${ASY_STATIC_LIBRARIES})
target_compile_definitions(asycore PUBLIC ${ASY_MACROS})
target_compile_options(asycore PUBLIC ${ASY_COMPILE_OPTS} ${ASY_LTO_COMPILE_OPTIONS})
target_link_options(asycore PUBLIC ${ASY_LINK_OPTS})

set_property(TARGET asycore APPEND PROPERTY LINK_DEPENDS ${ASY_ADDITIONAL_DEPENDENCIES})

# settings.cc bakes in ASYMPTOTE_SYSDIR, which differs per binary, so it is
# compiled per-executable as an OBJECT library rather than into the shared
# asycore (a single shared copy would collide with these objects). Each
# variant inherits asycore's include dirs / macros / options via the PRIVATE
# link; any extra args become build-specific compile definitions.
#
function(add_settings_obj target sysdir)
    add_library(${target} OBJECT ${ASY_SRC_DIR}/settings.cc)
    target_link_libraries(${target} PRIVATE asycore)
    target_compile_options(${target} PRIVATE ${ASY_COMPILE_OPTS} ${ASY_LTO_COMPILE_OPTIONS})
    target_compile_definitions(
            ${target} PRIVATE ASYMPTOTE_SYSDIR="${sysdir}" ${ARGN}
    )
endfunction()

#   settings_obj      -- normal binary (ASYMPTOTE_SYSDIR_VALUE, "" on Windows)
#   settings_obj_ctan -- CTAN/TeXLive binary (empty sysdir + KPSEWHICH: the
#                        sysdir comes from kpsewhich)
add_settings_obj(settings_obj "${ASYMPTOTE_SYSDIR_VALUE}")
add_settings_obj(settings_obj_ctan "" KPSEWHICH)

# asy executable (normal installer build)
add_executable(
        asy ${ASY_SRC_DIR}/main.cc ${ASY_WIN_RC_FILE} $<TARGET_OBJECTS:settings_obj>
)

# asy-ctan executable (CTAN/TexLive build with ASYMPTOTE_SYSDIR="", KPSEWHICH)
add_executable(asy-ctan ${ASY_SRC_DIR}/main.cc ${ASY_WIN_RC_FILE} $<TARGET_OBJECTS:settings_obj_ctan>)

# Link asycore into both executables. On Unix the renderer libraries
# (libasyvulkan.so / libasyopengl.so) are dlopened at runtime
# (rendererloader.cc) and resolve a large set of symbols -- camp::AsyRender,
# camp::IEXRFile, etc. -- from the loading binary rather than compiling them in.
# asycore is an OBJECT library (not STATIC), so all its objects are linked
# directly into the executable -- no WHOLE_ARCHIVE needed.  ENABLE_EXPORTS
# adds --export-dynamic (-rdynamic) so those symbols land in the executable's
# dynamic symbol table where the dlopened libraries can bind to them,
# mirroring the autotools build.
#
# Applied to asy-ctan as well so the CTAN/TeXLive binary can render too.
foreach(asy_exe asy asy-ctan)
    if (UNIX AND NOT WIN32)
        target_link_libraries(${asy_exe} PUBLIC asycore)
        set_target_properties(${asy_exe} PROPERTIES ENABLE_EXPORTS ON)
    else()
        target_link_libraries(${asy_exe} PUBLIC asycore)
    endif()
    target_link_options(${asy_exe} PRIVATE ${ASY_LTO_LINK_OPTIONS})
endforeach()

# base files
include(cmake-scripts/asy-base-files.cmake)

# asy + base files
add_custom_target(asy-basefiles DEPENDS ${ASY_OUTPUT_BASE_FILES})

# license files in build tree (for running asy locally without installing)
include(cmake-scripts/copy-build-licenses.cmake)

add_custom_target(asy-with-basefiles ALL DEPENDS asy asy-basefiles asy-licenses)

# Everything the `asy-check-tests` ctest label runs against.  ctest never builds
# anything, and asy-with-basefiles is not enough: the relocatable matrix also
# drives asy-ctan.  Build this before `ctest -L asy-check-tests`.
add_custom_target(asy-check-test-deps DEPENDS asy-with-basefiles asy-ctan)

# asy misc files
if (ENABLE_MISCFILES_GEN)
    include(cmake-scripts/asy-misc-files.cmake)
endif()

# documentation generation

if (ENABLE_DOCGEN)
    include(cmake-scripts/docgen.cmake)
endif()

# testing

enable_testing()
include(cmake-scripts/tests-asy.cmake)
include(cmake-scripts/tests-wce.cmake)

if (ENABLE_ASY_CXXTEST)
    add_subdirectory(cxxtests)
endif()

# installation
if (WIN32)
    # on windows, pre-NSIS
    include(cmake-scripts/win32-pre-nsis-installer.cmake)
else()
    # on unix systems
    include(cmake-scripts/linux-install.cmake)
endif()
