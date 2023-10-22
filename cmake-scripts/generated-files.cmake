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
    )
    list(APPEND ASYMPTOTE_GENERATED_HEADERS
            ${GENERATED_INCLUDE_DIR}/${runtime_file}.h)
endmacro()

foreach(RUNTIME_FILE ${RUNTIME_BUILD_FILES})
    add_runtime_file(${RUNTIME_FILE})
endforeach()


# allsymbols.h

macro(create_preprocess_gcc_options out_var_name)
    set(macro_flags ${ASY_MACROS})
    list(TRANSFORM macro_flags PREPEND -D)

    set(addr_flags -DNOSYM -E)

    set(${out_var_name}
            "$<LIST:TRANSFORM,$<TARGET_PROPERTY:asy,INCLUDE_DIRECTORIES>,PREPEND,-I>" ${macro_flags} ${addr_flags}
    )

    message(STATUS "asy flag is ${${out_var_name}}")
endmacro()

macro(create_preprocess_msvc_options out_var_name)
    TODO_NOTIMPL("Implement for msvc!")
endmacro()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    create_preprocess_gcc_options(asy_cc_compile_flags)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    create_preprocess_msvc_options(asy_cc_compile_flags)
endif()

# directory for auxilliary files
set(GENERATED_AUX_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/aux")
file(MAKE_DIRECTORY ${GENERATED_AUX_DIR})

# generating preprocessed files

function(symfile_preprocess src_dir symfile symfile_raw_output_varname)
    set(symfile_raw_output_var ${symfile_raw_output_varname})

    set(output_var ${GENERATED_AUX_DIR}/${symfile}.raw.i)
    set(${symfile_raw_output_var} ${output_var} PARENT_SCOPE)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(OUTPUT_ARGS -o ${output_var})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        TODO_NOTIMPL("Implement for msvc!")
    endif()

    add_custom_command(
            OUTPUT ${output_var}
            COMMAND ${CMAKE_CXX_COMPILER} ${asy_cc_compile_flags} ${OUTPUT_ARGS} ${src_dir}/${symfile}.cc
            MAIN_DEPENDENCY ${src_dir}/${symfile}.cc
            COMMAND_EXPAND_LISTS
            VERBATIM
    )

endfunction()

# preprocess each individual symbol files

foreach(SYM_FILE ${SYMBOL_STATIC_BUILD_FILES})
    symfile_preprocess(${ASY_SRC_DIR} ${SYM_FILE} SYMFILE_OUT)
    list(APPEND SYMFILE_OUT_LIST ${SYMFILE_OUT})
endforeach()

foreach(SYM_FILE ${RUNTIME_BUILD_FILES})
    symfile_preprocess(${GENERATED_SRC_DIR} ${SYM_FILE} SYMFILE_OUT)
    list(APPEND SYMFILE_OUT_LIST ${SYMFILE_OUT})
endforeach ()

# combine all files into allsymbols.h
set(FINDSYM_FILE ${ASY_SCRIPTS_DIR}/findsym.pl)

add_custom_command(
        OUTPUT ${GENERATED_INCLUDE_DIR}/allsymbols.h
        COMMAND ${PERL_INTERPRETER} ${FINDSYM_FILE}
            ${GENERATED_INCLUDE_DIR}/allsymbols.h
            ${SYMFILE_OUT_LIST}
        DEPENDS ${FINDSYM_FILE} ${SYMFILE_OUT_LIST}
)

list(APPEND ASYMPTOTE_GENERATED_HEADERS ${GENERATED_INCLUDE_DIR}/allsymbols.h)

# keywords.h

add_custom_command(
        OUTPUT ${GENERATED_INCLUDE_DIR}/keywords.h
        COMMAND ${PERL_INTERPRETER} ${ASY_SCRIPTS_DIR}/keywords.pl
            --camplfile ${ASY_RESOURCE_DIR}/camp.l
            --output ${GENERATED_INCLUDE_DIR}/keywords.h
            --process-file ${ASY_SRC_DIR}/process.cc
        MAIN_DEPENDENCY ${ASY_RESOURCE_DIR}/camp.l
        DEPENDS ${ASY_SCRIPTS_DIR}/keywords.pl ${ASY_SRC_DIR}/process.cc
)

list(APPEND ASYMPTOTE_GENERATED_HEADERS ${GENERATED_INCLUDE_DIR}/keywords.h)

add_custom_target(asy_gen_headers
        DEPENDS ${ASYMPTOTE_GENERATED_HEADERS}
)
