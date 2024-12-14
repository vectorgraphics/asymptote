# add tests here

set(ASY_CXX_TESTS
        placeholder
)

# ----- transform tests --------
list(TRANSFORM ASY_CXX_TESTS APPEND .cc)
list(TRANSFORM ASY_CXX_TESTS
        PREPEND ${TEST_CXX_SRC_ROOT}/tests/
        OUTPUT_VARIABLE ASY_TEST_SOURCES
)
