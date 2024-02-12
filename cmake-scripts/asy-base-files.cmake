set(ASY_STATIC_BASE_FILES
        animate animation annotate babel bezulate binarytree bsp CAD colormap
        contour3 contour drawtree embed external feynman flowchart fontsize
        geometry graph3 graph graph_settings graph_splinetype grid3 interpolate
        labelpath3 labelpath lmfit map markers math metapost obj ode palette patterns
        plain_arcs plain_arrows plain plain_bounds plain_boxes plain_constants plain_debugger
        plain_filldraw plain_Label plain_margins plain_markers plain_paths plain_pens
        plain_picture plain_prethree plain_scaling plain_shipout plain_strings pstoedit rational rationalSimplex
        roundedpath simplex size10 size11 slide slopefield smoothcontour3 solids stats syzygy
        texcolors three_arrows three three_light three_margins three_surface three_tube tree
        trembling tube v3d x11colors
)

set(ASY_STATIC_SHADER_FILES
        blend compress count fragment screen sum1 sum2
        sum3 vertex fxaa.cs
)

# base dir
set(ASY_SOURCE_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/base)
set(ASY_BUILD_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/base)

file(MAKE_DIRECTORY ${ASY_BUILD_BASE_DIR})

# version.asy
configure_file(${ASY_RESOURCE_DIR}/versionTemplate.asy.in ${ASY_BUILD_BASE_DIR}/version.asy)
list(APPEND ASY_OUTPUT_BASE_FILES ${ASY_BUILD_BASE_DIR}/version.asy)

# copy base files to build dir
macro (copy_base_file base_file_name)
    add_custom_command(
            COMMAND ${CMAKE_COMMAND} -E copy
            ${ASY_SOURCE_BASE_DIR}/${base_file_name}
            ${ASY_BUILD_BASE_DIR}/${base_file_name}
            OUTPUT ${ASY_BUILD_BASE_DIR}/${base_file_name}
            MAIN_DEPENDENCY ${ASY_SOURCE_BASE_DIR}/${base_file_name}
    )

    list(APPEND ASY_OUTPUT_BASE_FILES ${ASY_BUILD_BASE_DIR}/${base_file_name})
endmacro()

foreach(ASY_STATIC_BASE_FILE ${ASY_STATIC_BASE_FILES})
    copy_base_file(${ASY_STATIC_BASE_FILE}.asy)
endforeach ()

file(MAKE_DIRECTORY ${ASY_BUILD_BASE_DIR}/shaders)
foreach(ASY_STATIC_SHADER_FILE ${ASY_STATIC_SHADER_FILES})
    copy_base_file(shaders/${ASY_STATIC_SHADER_FILE}.glsl)
endforeach ()

# generated csv files
foreach(csv_enum_file ${ASY_CSV_ENUM_FILES})
    add_custom_command(
            OUTPUT ${ASY_BUILD_BASE_DIR}/${csv_enum_file}.asy
            COMMAND ${PY3_INTERPRETER} ${ASY_SCRIPTS_DIR}/generate_enums.py
            --language asy
            --name ${csv_enum_file}
            --input ${ASY_RESOURCE_DIR}/${csv_enum_file}.csv
            --output ${ASY_BUILD_BASE_DIR}/${csv_enum_file}.asy
            MAIN_DEPENDENCY ${ASY_RESOURCE_DIR}/${csv_enum_file}.csv
    )

    list(APPEND ASY_OUTPUT_BASE_FILES ${ASY_BUILD_BASE_DIR}/${csv_enum_file}.asy)
endforeach ()
