macro(build_files_to_src in_var out_var)
    list(TRANSFORM ${in_var} PREPEND ${ASY_SRC_DIR}/ OUTPUT_VARIABLE ${out_var})
    list(TRANSFORM ${out_var} APPEND .cc)
endmacro()
