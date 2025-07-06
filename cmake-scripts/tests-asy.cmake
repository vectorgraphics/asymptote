# ---- asy tests ------

set(ASY_ASYLANG_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(ASY_ASYLANG_TEST_SCRATCH_DIR ${ASY_ASYLANG_TEST_ROOT}/out/)

function(add_individual_asy_tests)
    set(fn_oneval_args DIR FILE ADDR_ASY_ARGS)
    cmake_parse_arguments(
            ASY_TEST "${fn_oneval_args}" ${ARGN}
    )

    set(TEST_PATH ${ASY_ASYLANG_TEST_ROOT}/${ASY_TEST_DIR}/${ASY_TEST_FILE}.asy)
    set(TEST_NAME "asy.${ASY_TEST_DIR}.${ASY_TEST_FILE}")
    add_test(
            NAME ${TEST_NAME}
            COMMAND asy
                -dir ${ASY_BUILD_BASE_DIR} ${TEST_PATH}
                -noV
                -o out
                -globalwrite
                ${ASY_TEST_ADDR_ASY_ARGS}
            WORKING_DIRECTORY ${ASY_ASYLANG_TEST_ROOT}
    )

    if (ASY_TEST_TEST_LABELS)
        set_property(
                TEST ${TEST_NAME}
                PROPERTY LABELS ${ASY_TEST_TEST_LABELS}
        )
    endif()

endfunction()

macro(add_asy_tests)
    set(macro_oneval_args TEST_DIR ADDR_ASY_ARGS)
    cmake_parse_arguments(
            ASY_TESTING "${macro_oneval_args}" ${ARGN}
    )
    foreach(testfile ${ASY_TESTING_TESTS})
      set(TEST_LABEL asy-check-tests)

        add_individual_asy_tests(
                DIR ${ASY_TESTING_TEST_DIR}
                FILE ${testfile}
                ADDR_ASY_ARGS ${ASY_TESTING_ADDR_ASY_ARGS}
                TEST_LABELS ${TEST_LABEL}
        )
    endforeach()

endmacro()

# ------ tests ----------

include(${CMAKE_CURRENT_LIST_DIR}/generated/asy-tests-list.cmake)
