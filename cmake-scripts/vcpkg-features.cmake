# call by map_option_to_vcpkg_feat(OPTION_NAME feat1 ... featn) where
# feat1 to featn are vcpkg features in the manifest file

set(VCPKG_MANIFEST_FEATURES)

macro(map_option_to_vcpkg_feat option_name)
    if (${option_name})
        list(APPEND VCPKG_MANIFEST_FEATURES ${ARGN})
    endif()
endmacro()

map_option_to_vcpkg_feat(ENABLE_GC gc)
map_option_to_vcpkg_feat(ENABLE_READLINE readline)
map_option_to_vcpkg_feat(ENABLE_CURL curl)
map_option_to_vcpkg_feat(ENABLE_GSL gsl)
map_option_to_vcpkg_feat(ENABLE_EIGEN3 eigen3)
map_option_to_vcpkg_feat(ENABLE_FFTW3 fftw3)
map_option_to_vcpkg_feat(ENABLE_VULKAN vulkan)
map_option_to_vcpkg_feat(ENABLE_THREADING threading)
map_option_to_vcpkg_feat(ENABLE_ASY_CXXTEST build-cxx-testing)
map_option_to_vcpkg_feat(ENABLE_LSP lsp)
