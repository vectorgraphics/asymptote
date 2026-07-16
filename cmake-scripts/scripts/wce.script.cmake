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

    # Normalize line endings (CRLF -> LF, standalone CR -> LF).
    string(REPLACE "\r\n" "\n" ASY_STDERR_OUTPUT "${ASY_STDERR_OUTPUT}")
    string(REPLACE "\r\n" "\n" EXPECTED_ERROR_OUTPUT "${EXPECTED_ERROR_OUTPUT}")
    string(REPLACE "\r" "\n" ASY_STDERR_OUTPUT "${ASY_STDERR_OUTPUT}")
    string(REPLACE "\r" "\n" EXPECTED_ERROR_OUTPUT "${EXPECTED_ERROR_OUTPUT}")

    # Trim trailing whitespace and newlines so that harmless formatting
    # differences do not cause test failures.
    string(REGEX REPLACE "[ \t\n]+$" "" ASY_STDERR_OUTPUT "${ASY_STDERR_OUTPUT}")
    string(REGEX REPLACE "[ \t\n]+$" "" EXPECTED_ERROR_OUTPUT "${EXPECTED_ERROR_OUTPUT}")

    # Strip indented continuation lines (search paths, help text) from
    # multi-line error messages.  These contain environment-specific paths
    # that differ between build configurations.
    string(REGEX REPLACE "\n  [^\n]*" "" ASY_STDERR_OUTPUT "${ASY_STDERR_OUTPUT}")
    string(REGEX REPLACE "\n  [^\n]*" "" EXPECTED_ERROR_OUTPUT "${EXPECTED_ERROR_OUTPUT}")

    # Collapse redundant blank lines left after stripping continuation text.
    string(REGEX REPLACE "\n\n+" "\n" ASY_STDERR_OUTPUT "${ASY_STDERR_OUTPUT}")
    string(REGEX REPLACE "\n\n+" "\n" EXPECTED_ERROR_OUTPUT "${EXPECTED_ERROR_OUTPUT}")

    if (NOT ASY_STDERR_OUTPUT STREQUAL EXPECTED_ERROR_OUTPUT)
        message(WARNING "Error test FAILED: ${TEST_NAME}")
        message(STATUS "  Expected:\n${EXPECTED_ERROR_OUTPUT}")
        message(STATUS "  Actual:\n${ASY_STDERR_OUTPUT}")
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
