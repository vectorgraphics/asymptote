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

# ---- relocatable sysdir-resolution matrix ------
# Runs in either build.  ENABLE_RELOCATABLE decides which half of the matrix
# asserts, not whether there is one to run: with it off, the rows where
# <exedir>/../share/asymptote or <exedir> holds a valid base/ become the
# regression that those candidates really are compiled out of locate.cc.  That
# is the configuration Asymptote ships by default, so it is the one that most
# needs a test.  The script also probes the CTAN binary (the ctan/* rows, via
# the asy-ctan target) and the compiled-in fallback (the C/* rows); asy-ctan is
# not part of asy-with-basefiles, so build asy-check-test-deps before running
# this label.  Given only asy -- an artifact download, say -- the ctan/* rows
# report themselves skipped and the rest still run.
#
# Pass --mode rather than letting the script auto-detect: detection infers the
# mode from whether K2 resolves, so a binary that lost K2/K3 entirely would be
# reclassified as non-relocatable and the matrix would still pass.  Here the
# build flag is known, so state it and let the K2/K3 rows actually assert.
if (ENABLE_RELOCATABLE)
    set(ASY_RELOCATABLE_MODE on)
else ()
    set(ASY_RELOCATABLE_MODE off)
endif ()

add_test(
        NAME bundled.asy.relocatable
        COMMAND ${PY3_INTERPRETER} ${ASY_ASYLANG_TEST_ROOT}/test_relocatable.py
            --asy $<TARGET_FILE:asy>
            --asy-base-dir ${ASY_BUILD_BASE_DIR}
            --asy-ctan $<TARGET_FILE:asy-ctan>
            --compiled-in ${ASYMPTOTE_SYSDIR_VALUE}
            --mode ${ASY_RELOCATABLE_MODE}
        WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
)
set_property(
        TEST bundled.asy.relocatable
        PROPERTY LABELS asy-check-tests
)
