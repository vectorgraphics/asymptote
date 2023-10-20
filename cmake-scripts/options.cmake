# Perl

option(PERL_INTERPRETER "Perl interpreter")

if(NOT PERL_INTERPRETER)
    message(STATUS "No Perl interpreter specified, attempting to find perl")
    find_program(
            PERL_INTERPRETER_FOUND
            perl
            REQUIRED
    )
    message(STATUS "Found perl at ${PERL_INTERPRETER_FOUND}")
    set(PERL_INTERPRETER ${PERL_INTERPRETER_FOUND})
endif()

execute_process(COMMAND ${PERL_INTERPRETER} -e "print \"$]\"" OUTPUT_VARIABLE PERL_VERSION)
message(STATUS "Perl version: ${PERL_VERSION}")


# Python

option(PY3_INTERPRETER "Python 3 interpreter")

function(verify_py3_interpreter_is_py3 validator_result_var py_interpreter)
    execute_process(
            COMMAND ${py_interpreter} -c "import sys; print(int(sys.version[0])>=3,end='')"
            OUTPUT_VARIABLE PY3_INTERPRETER_VERSION_RESULT)
    if (NOT PY3_INTERPRETER_VERSION_RESULT STREQUAL "True")
        set(${validator_result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

if(NOT PY3_INTERPRETER)
    message(STATUS "No Python3 interpreter specified, attempting to find python")
    find_program(
            PY3_INTERPRETER_FOUND
            NAMES python3 python
            VALIDATOR verify_py3_interpreter_is_py3
            REQUIRED
    )
    message(STATUS "Found python3 at ${PY3_INTERPRETER_FOUND}")
    set(PY3_INTERPRETER ${PY3_INTERPRETER_FOUND})
else()
    set(PY_INTERPRETER_IS_PY3 TRUE)
    set(VARIABLE_RESULT_VAR PY_INTERPRETER_IS_PY3)
    verify_py3_interpreter_is_py3(VARIABLE_RESULT_VAR ${PY3_INTERPRETER})
endif()

execute_process(COMMAND ${PY3_INTERPRETER} --version OUTPUT_VARIABLE PY3_VERSION)
message(STATUS "Version: ${PY3_VERSION}")
