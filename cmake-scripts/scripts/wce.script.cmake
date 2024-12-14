# testing for wce. Equivalent to ./wce script

execute_process(
        COMMAND ${ASY_EXEC} -q -sysdir ${ASY_BASE_DIR} -noautoplain -debug errortest
        WORKING_DIRECTORY ${SOURCE_ROOT}
        ERROR_VARIABLE ASY_STDERR_OUTPUT
)

file(READ ${SOURCE_ROOT}/errors EXPECTED_ERROR_OUTPUT)

if (NOT ASY_STDERR_OUTPUT STREQUAL EXPECTED_ERROR_OUTPUT)
    message(FATAL_ERROR "Asymptote error test fails.")
else()
    message(STATUS "Asymptote error messages equal expected. Test passes")
endif()
