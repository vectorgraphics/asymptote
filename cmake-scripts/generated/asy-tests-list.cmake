
# This file is automatically generated. Do not modify manually.
# This file is checked in as part of the repo. It is not meant to be ignored.
#
# If more tests are added, run scan-asy-tests-cmake.py to re-generate
# the test list.


add_asy_tests(
    TEST_DIR arith
    TESTS integer pair random real roots transform triple


)


add_asy_tests(
    TEST_DIR array
    TESTS array delete determinant fields slice solve sort transpose


)



add_asy_tests(
    TEST_DIR frames
    TESTS loop stat stat2


)

if (ENABLE_GC)

add_asy_tests(
    TEST_DIR gc
    TESTS array file funcall guide label path pen shipout string struct transform
    TEST_ARTIFACTS .eps
    TEST_NOT_PART_OF_CHECK_TEST true
)
endif()


add_asy_tests(
    TEST_DIR gs
    TESTS ghostscript

    TEST_NOT_PART_OF_CHECK_TEST true
)

if (ENABLE_GSL)

add_asy_tests(
    TEST_DIR gsl
    TESTS random


)
endif()


add_asy_tests(
    TEST_DIR imp
    TESTS unravel


)


add_asy_tests(
    TEST_DIR io
    TESTS csv read


)


add_asy_tests(
    TEST_DIR output
    TESTS circle line
    TEST_ARTIFACTS circle.eps line.eps

)


add_asy_tests(
    TEST_DIR pic
    TESTS trans


)


add_asy_tests(
    TEST_DIR string
    TESTS erase find insert length rfind substr


)


add_asy_tests(
    TEST_DIR template
    TESTS functionTest initTest mapArrayTest multiImport nestedImport singletype sortedsetTest splaytreeTest structTest


)


add_asy_tests(
    TEST_DIR types
    TESTS autounravel builtinOps cast constructor ecast guide init keyword order overrideEquals resolve shadow spec var


)
