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
# It exercises the <exedir>/base candidate on its own.  It is also the
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

# ---- sysdir-resolution matrix ------
# Covers the whole resolver, not just the one candidate above: the decoy layouts
# that must *not* resolve, the launch routes that feed executablePath(), the
# -sysdir / -dir / ASYMPTOTE_SYSDIR overrides applied afterwards, and the
# compiled-in fallback (the C/* rows).  It also probes the CTAN binary (the
# ctan/* rows, via the asy-ctan target); asy-ctan is not part of
# asy-with-basefiles, so build asy-check-test-deps before running this label.
# Given only asy -- an artifact download, say -- the ctan/* rows report
# themselves skipped and the rest still run.
add_test(
        NAME bundled.asy.relocatable
        COMMAND ${PY3_INTERPRETER} ${ASY_ASYLANG_TEST_ROOT}/test_relocatable.py
            --asy $<TARGET_FILE:asy>
            --asy-base-dir ${ASY_BUILD_BASE_DIR}
            --asy-ctan $<TARGET_FILE:asy-ctan>
            --compiled-in ${ASYMPTOTE_SYSDIR_VALUE}
        WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
)
set_property(
        TEST bundled.asy.relocatable
        PROPERTY LABELS asy-check-tests
)
