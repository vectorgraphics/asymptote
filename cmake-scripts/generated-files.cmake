
set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY ${GENERATED_INCLUDE_DIR})

# opsymbols.h
add_custom_command(
        OUTPUT ${GENERATED_INCLUDE_DIR}/opsymbols.h
        COMMAND ${PERL_INTERPRETER}
            ${ASY_SCRIPTS_DIR}/opsymbols.pl
            --campfile ${ASY_RESOURCE_DIR}/camp.l
            --output ${GENERATED_INCLUDE_DIR}/opsymbols.h
        MAIN_DEPENDENCY ${ASY_RESOURCE_DIR}/camp.l
        DEPENDS ${ASY_SCRIPTS_DIR}/opsymbols.pl
)

list(APPEND ASYMPTOTE_INCLUDES ${GENERATED_INCLUDE_DIR})
list(APPEND ASYMPTOTE_GENERATED_HEADERS ${GENERATED_INCLUDE_DIR}/opsymbols.h)

# generated sources
set(GENERATED_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/src")
file(MAKE_DIRECTORY ${GENERATED_SRC_DIR})


# run-* files

function(_int_add_runtime_file runtime_file)
    set(RUNTIME_FILE_IN_BASE ${ASY_SRC_TEMPLATES_DIR}/${runtime_file})
    set(RUNTIME_FILES_OUT ${GENERATED_SRC_DIR}/${runtime_file}.cc ${GENERATED_INCLUDE_DIR}/${runtime_file}.h)
    set(RUNTIME_SCRIPT ${ASY_SCRIPTS_DIR}/runtime.pl)
    set(OPSYM_FILE ${GENERATED_INCLUDE_DIR}/opsymbols.h)
    set(RUNTIME_BASE_FILE ${ASY_SRC_TEMPLATES_DIR}/runtimebase.in)

    add_custom_command(
            OUTPUT ${RUNTIME_FILES_OUT}
            COMMAND ${PERL_INTERPRETER} ${RUNTIME_SCRIPT}
            --opsym-file ${OPSYM_FILE}
            --runtime-base-file ${RUNTIME_BASE_FILE}
            --src-template-dir ${ASY_SRC_TEMPLATES_DIR}
            --prefix ${runtime_file}
            --header-out-dir ${GENERATED_INCLUDE_DIR}
            --src-out-dir ${GENERATED_SRC_DIR}
            MAIN_DEPENDENCY ${RUNTIME_FILE_IN_BASE}.in
            DEPENDS ${RUNTIME_SCRIPT} ${OPSYM_FILE} ${RUNTIME_BASE_FILE}
    )
endfunction()

macro(add_runtime_file runtime_file)
    _int_add_runtime_file(${runtime_file})
    list(APPEND ASY_GENERATED_BUILD_SOURCES
            ${GENERATED_SRC_DIR}/${runtime_file}.cc
            ${GENERATED_INCLUDE_DIR}/${runtime_file}.h
    )
endmacro()

set(RUNTIME_FILES
        runtime runbacktrace runpicture runlabel runhistory runarray
        runfile runsystem runpair runtriple runpath runpath3d runstring runmath
)

foreach(RUNTIME_FILE ${RUNTIME_FILES})
    add_runtime_file(${RUNTIME_FILE})
endforeach()
