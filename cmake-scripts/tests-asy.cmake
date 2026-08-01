# ---- asy tests ------

set(ASY_ASYLANG_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(ASY_ASYLANG_TEST_SCRATCH_DIR ${ASY_ASYLANG_TEST_ROOT}/out/)

add_test(
        NAME bundled.asy.checktests
        COMMAND ${PY3_INTERPRETER} ${ASY_ASYLANG_TEST_ROOT}/run_asy_tests.py
            --asy $<TARGET_FILE:asy>
            --asy-base-dir=${ASY_BUILD_BASE_DIR}
        WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
)
set_property(
        TEST bundled.asy.checktests
        PROPERTY LABELS asy-check-tests
)

set_property(
        TARGET asy
        APPEND
        PROPERTY ADDITIONAL_CLEAN_FILES ${ASY_ASYLANG_TEST_SCRATCH_DIR}
)

add_test(
        NAME bundled.asy.collections_errors
        COMMAND ${PY3_INTERPRETER} ${ASY_ASYLANG_TEST_ROOT}/test_collections_errors.py
            --asy $<TARGET_FILE:asy>
            --asy-base-dir=${ASY_BUILD_BASE_DIR}
        WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
)
set_property(
        TEST bundled.asy.collections_errors
        PROPERTY LABELS asy-check-tests
)

# ---- getExecutablePath() smoke test ------
# Ungated: it exercises only the <exedir>/base candidate, which is live in every
# build rather than only in an ENABLE_RELOCATABLE one.  It is also the
# prerequisite for any wider sysdir test -- a wrong exedir would invalidate
# every case of one -- so it is worth running wherever asy is built.
add_test(
        NAME bundled.asy.executable_path
        COMMAND ${PY3_INTERPRETER} ${ASY_ASYLANG_TEST_ROOT}/test_executable_path.py
            --asy $<TARGET_FILE:asy>
            --asy-base-dir ${ASY_BUILD_BASE_DIR}
        WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
)
set_property(
        TEST bundled.asy.executable_path
        PROPERTY LABELS asy-check-tests
)
