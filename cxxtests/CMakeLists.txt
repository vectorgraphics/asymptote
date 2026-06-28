if (NOT ENABLE_ASY_CXXTEST)
    message(FATAL_ERROR "cxxtests require asy-cxxtest enabled")
endif()

set(TEST_CXX_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})

include(${TEST_CXX_SRC_ROOT}/cmake-scripts/external-libs.cmake)
include(${TEST_CXX_SRC_ROOT}/cmake-scripts/tests.cmake)

# test target

# settings.cc is no longer part of asycore (it is a per-binary OBJECT library);
# pull in the normal settings_obj so the test binary resolves settings symbols.
add_executable(asyCxxTests ${ASY_TEST_SOURCES} ${TEST_CXX_SRC_ROOT}/src/testMain.cc
        $<TARGET_OBJECTS:settings_obj>)
target_include_directories(
        asyCxxTests
        PRIVATE ${TEST_CXX_SRC_ROOT}/include
)
target_link_libraries(asyCxxTests
        PRIVATE asycore GTest::gtest GTest::gmock)

include(GoogleTest)
gtest_discover_tests(asyCxxTests TEST_PREFIX cxxtests.)
