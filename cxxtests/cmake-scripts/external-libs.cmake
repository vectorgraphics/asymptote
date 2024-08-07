# Use directly downloaded library because vcpkg's version has some
# linking issues with windows + clang64-msys2

if (DOWNLOAD_GTEST_FROM_SRC)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest
            GIT_TAG v1.14.0
    )

    if (WIN32)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(googletest)
else()
    find_package(GTest REQUIRED)
endif()
