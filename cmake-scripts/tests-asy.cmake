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

add_test(
        NAME bundled.asy.render3d
        COMMAND ${PY3_INTERPRETER} ${ASY_ASYLANG_TEST_ROOT}/test_render3d.py
            --asy $<TARGET_FILE:asy>
            --asy-base-dir=${ASY_BUILD_BASE_DIR}
        WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
)
set_tests_properties(
        bundled.asy.render3d
        PROPERTIES LABELS asy-check-tests SKIP_RETURN_CODE 77
)
