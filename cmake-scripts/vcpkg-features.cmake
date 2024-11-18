if (ENABLE_GC)
    list(APPEND VCPKG_MANIFEST_FEATURES gc)
endif()

if (ENABLE_READLINE)
    list(APPEND VCPKG_MANIFEST_FEATURES readline)
endif()

if (ENABLE_CURL)
    list(APPEND VCPKG_MANIFEST_FEATURES curl)
endif()

if (ENABLE_GSL)
    list(APPEND VCPKG_MANIFEST_FEATURES gsl)
endif()

if (ENABLE_EIGEN3)
    list(APPEND VCPKG_MANIFEST_FEATURES eigen3)
endif()

if (ENABLE_FFTW3)
    list(APPEND VCPKG_MANIFEST_FEATURES fftw3)
endif()

if (ENABLE_OPENGL)
    list(APPEND VCPKG_MANIFEST_FEATURES opengl)

    if (ENABLE_GL_OFFSCREEN_RENDERING)
        # using mesa from vcpkg requires building LLVM, which takes
        # a disproportionally long time compared to other packages

        # For now, we can use the system's OSMesa
        # list(APPEND VCPKG_MANIFEST_FEATURES gl-offscreen)
    endif()
endif()

if (ENABLE_THREADING)
    list(APPEND VCPKG_MANIFEST_FEATURES threading)
endif()

if (ENABLE_ASY_CXXTEST)
    list(APPEND VCPKG_MANIFEST_FEATURES build-cxx-testing)
endif()

if (ENABLE_LSP)
    list(APPEND VCPKG_MANIFEST_FEATURES lsp)
endif()
