# ---- asy tests ------

set(ASY_ASYLANG_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(ASY_ASYLANG_TEST_SCRATCH_DIR ${ASY_ASYLANG_TEST_ROOT}/out/)

function(add_individual_asy_tests)
    set(fn_opts)
    set(fn_oneval_args DIR FILE ADDR_ASY_ARGS)
    set(fn_multival_args TEST_LABELS)
    cmake_parse_arguments(
            ASY_TEST "${fn_opts}" "${fn_oneval_args}" "${fn_multival_args}" ${ARGN}
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
    set(macro_opts TEST_NOT_PART_OF_CHECK_TEST)
    set(macro_oneval_args TEST_DIR ADDR_ASY_ARGS)
    set(macro_multival_args TESTS TEST_ARTIFACTS)
    cmake_parse_arguments(
            ASY_TESTING "${macro_opts}" "${macro_oneval_args}" "${macro_multival_args}" ${ARGN}
    )
    foreach(testfile ${ASY_TESTING_TESTS})

        if (ASY_TESTING_TEST_NOT_PART_OF_CHECK_TEST)
            set(TEST_LABEL asy-extended-tests)
        else()
            set(TEST_LABEL asy-check-tests)
        endif()


        add_individual_asy_tests(
                DIR ${ASY_TESTING_TEST_DIR}
                FILE ${testfile}
                ADDR_ASY_ARGS ${ASY_TESTING_ADDR_ASY_ARGS}
                TEST_LABELS ${TEST_LABEL}
        )
    endforeach()

    foreach(artifact ${ASY_TESTING_TEST_ARTIFACTS})
        set_property(
                TARGET asy
                APPEND
                PROPERTY ADDITIONAL_CLEAN_FILES ${ASY_ASYLANG_TEST_SCRATCH_DIR}/${artifact}
        )
    endforeach()
endmacro()

# ------ tests ----------


include(${CMAKE_CURRENT_LIST_DIR}/generated/asy-tests-list.cmake OPTIONAL RESULT_VARIABLE ASY_TESTS_IMPORTED)

if (RESULT_VARIABLE STREQUAL NOTFOUND)
    message(WARNING "\
Asymptote test list not found. Asymptote tests will not be available. Please run the python script
<asymptote-root>/scan-asy-tests-cmake.py
to generate the test list file and then reload the cmake project.
"
    )
endif()
