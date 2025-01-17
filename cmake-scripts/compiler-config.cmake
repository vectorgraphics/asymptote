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
