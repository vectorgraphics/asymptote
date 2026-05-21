# testing for wce. Equivalent to ./wce script

set(ERRORTESTS_DIR ${SOURCE_ROOT}/errortests)

# Collect all .asy test files, excluding helper files.
file(GLOB TEST_FILES "${ERRORTESTS_DIR}/*.asy")
set(HELPER_FILES
    "${ERRORTESTS_DIR}/errortestNonTemplate.asy"
    "${ERRORTESTS_DIR}/errortestBrokenTemplate.asy"
    "${ERRORTESTS_DIR}/errortestTemplate.asy"
)
list(REMOVE_ITEM TEST_FILES ${HELPER_FILES})
list(SORT TEST_FILES)

list(LENGTH TEST_FILES NUM_TESTS)
if (NUM_TESTS EQUAL 0)
    message(FATAL_ERROR "No error test files found in ${ERRORTESTS_DIR}")
endif()

set(ALL_PASSED TRUE)

foreach(TEST_FILE IN LISTS TEST_FILES)
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    set(MODULE_PATH "errortests/${TEST_NAME}")
    set(ERRORS_FILE "${ERRORTESTS_DIR}/${TEST_NAME}.errors")

    execute_process(
            COMMAND ${ASY_EXEC} -q -sysdir ${ASY_BASE_DIR} -noautoplain -debug ${MODULE_PATH}
            WORKING_DIRECTORY ${SOURCE_ROOT}
            ERROR_VARIABLE ASY_STDERR_OUTPUT
    )

    file(READ ${ERRORS_FILE} EXPECTED_ERROR_OUTPUT)

    if (NOT ASY_STDERR_OUTPUT STREQUAL EXPECTED_ERROR_OUTPUT)
        message(WARNING "Error test FAILED: ${TEST_NAME}")
        set(ALL_PASSED FALSE)
    else()
        message(STATUS "Error test passed: ${TEST_NAME}")
    endif()
endforeach()

if (NOT ALL_PASSED)
    message(FATAL_ERROR "One or more Asymptote error tests failed.")
else()
    message(STATUS "All Asymptote error tests passed.")
endif()
