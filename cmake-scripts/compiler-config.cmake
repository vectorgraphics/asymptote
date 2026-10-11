set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (OPTIMIZE_LINK_TIME)
    include(CheckIPOSupported)

    check_ipo_supported(RESULT ipo_supported_result LANGUAGES C CXX)

    if (ipo_supported_result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "Using link-time optimization")
    else()
        message(FATAL_ERROR "Compiler does not support link-time optimization")
    endif()
endif()

if (TARGET_X86_64_V3_OR_HIGHER STREQUAL "ON")
    if (MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-march=x86-64-v3)
    endif()
    message(STATUS "x86-64-v3 targeting: forced ON")
elseif (TARGET_X86_64_V3_OR_HIGHER STREQUAL "OFF")
    message(STATUS "x86-64-v3 targeting: disabled")
else()
    # AUTO: detect AVX2 support on the build CPU
    set(ACCEPTABLE_X86_ARCHITECTURE_FLAGS AMD64 x86 x86_64 x86-64 i386 i686)
    if (CMAKE_SYSTEM_PROCESSOR IN_LIST ACCEPTABLE_X86_ARCHITECTURE_FLAGS)
        if (MSVC)
            # MSVC: check for AVX2 intrinsics header availability
            include(CheckCXXSourceCompiles)
            check_cxx_source_compiles("
                #include <immintrin.h>
                int main() {
                    __m256i v = _mm256_setzero_si256();
                    return _mm256_movemask_epi8(v);
                }
            " HAVE_BUILD_CPU_AVX2)
        else()
            include(CheckCSourceRuns)
            check_c_source_runs("
                #include <stdlib.h>
                int main() {
                    return __builtin_cpu_supports(\"avx2\") ? 0 : 1;
                }
            " HAVE_BUILD_CPU_AVX2)
        endif()

        if (HAVE_BUILD_CPU_AVX2)
            if (MSVC)
                add_compile_options(/arch:AVX2)
                message(STATUS "Build CPU supports AVX2, targeting x86-64-v3 (/arch:AVX2)")
            else()
                include(CheckCCompilerFlag)
                check_c_compiler_flag(-march=x86-64-v3 COMPILER_SUPPORTS_X86_64_V3)
                if (COMPILER_SUPPORTS_X86_64_V3)
                    add_compile_options(-march=x86-64-v3)
                    message(STATUS "Build CPU supports AVX2, targeting x86-64-v3")
                else()
                    message(STATUS "Build CPU supports AVX2 but compiler lacks -march=x86-64-v3, using baseline")
                endif()
            endif()
        else()
            message(STATUS "Build CPU does not support AVX2, using baseline x86-64")
        endif()
    else()
        message(STATUS "Not an x86 architecture (${CMAKE_SYSTEM_PROCESSOR}), x86-64-v3 targeting not applicable")
    endif()
endif()
