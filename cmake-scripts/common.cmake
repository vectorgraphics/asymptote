macro(TODO_NOTIMPL message)
    message(FATAL_ERROR "TODO: ${message}")
endmacro()

set(cmake_release_build_types Release RelWithDebInfo MinSizeRel)
set(cmake_debug_build_types Debug)
