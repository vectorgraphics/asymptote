
set(ASY_SOURCE_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/base)
set(ASY_SOURCE_BASE_SHADER_DIR ${ASY_SOURCE_BASE_DIR}/shaders)

file(GLOB ASY_STATIC_BASE_FILES RELATIVE ${ASY_SOURCE_BASE_DIR} CONFIGURE_DEPENDS ${ASY_SOURCE_BASE_DIR}/*.asy)
file(GLOB ASY_STATIC_SHADER_FILES RELATIVE ${ASY_SOURCE_BASE_DIR} CONFIGURE_DEPENDS ${ASY_SOURCE_BASE_SHADER_DIR}/*.glsl)

set(ASY_STATIC_BASE_COLLECTIONS_FILES
    btree btreegeneral btreemap enumerate genericpair hashmap hashset iter map
    queue set sortedset splaytree wraparray wrapper zip zip2
)

set(OTHER_STATIC_BASE_FILES nopapersize.ps)

# base dir
set(ASY_BUILD_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/base)

file(MAKE_DIRECTORY ${ASY_BUILD_BASE_DIR})

# version.asy
configure_file(${ASY_RESOURCE_DIR}/versionTemplate.asy.in ${ASY_BUILD_BASE_DIR}/version.asy)
list(APPEND ASY_OUTPUT_BASE_FILES ${ASY_BUILD_BASE_DIR}/version.asy)

# copy base files to build dir
macro (copy_base_file_with_custom_output_name base_file_name output_file_name)
    add_custom_command(
            COMMAND ${CMAKE_COMMAND} -E copy
            ${ASY_SOURCE_BASE_DIR}/${base_file_name}
            ${ASY_BUILD_BASE_DIR}/${output_file_name}
            OUTPUT ${ASY_BUILD_BASE_DIR}/${output_file_name}
            MAIN_DEPENDENCY ${ASY_SOURCE_BASE_DIR}/${base_file_name}
    )

    list(APPEND ASY_OUTPUT_BASE_FILES ${ASY_BUILD_BASE_DIR}/${output_file_name})
endmacro()

macro (copy_base_file base_file_name)
    copy_base_file_with_custom_output_name(${base_file_name} ${base_file_name})
endmacro()

foreach(ASY_STATIC_BASE_FILE ${ASY_STATIC_BASE_FILES})
    copy_base_file(${ASY_STATIC_BASE_FILE})
endforeach ()

file(MAKE_DIRECTORY ${ASY_BUILD_BASE_DIR}/collections)
foreach (ASY_COLLECTION_BASE_FILE ${ASY_STATIC_BASE_COLLECTIONS_FILES})
    copy_base_file(collections/${ASY_COLLECTION_BASE_FILE}.asy)
endforeach()

foreach(OTHER_STATIC_BASE_FILE ${OTHER_STATIC_BASE_FILES})
    copy_base_file(${OTHER_STATIC_BASE_FILE})
endforeach ()

file(MAKE_DIRECTORY ${ASY_BUILD_BASE_DIR}/shaders)
foreach(ASY_STATIC_SHADER_FILE ${ASY_STATIC_SHADER_FILES})
    copy_base_file(${ASY_STATIC_SHADER_FILE})
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

# asygl
file(MAKE_DIRECTORY ${ASY_BUILD_BASE_DIR}/webgl)
copy_base_file_with_custom_output_name(webgl/asygl-${ASY_GL_VERSION}.js webgl/asygl.js)
