set(ASYMPTOTE_INCLUDES ${ASY_INCLUDE_DIR})
# Note: settings.cc is intentionally NOT in this list. It bakes in
# ASYMPTOTE_SYSDIR, which differs per binary (normal vs CTAN), so it is compiled
# as a per-executable object library (settings_obj / settings_obj_ctan) rather
# than into the shared asycore. Keeping it out of asycore also lets asycore be
# whole-archived into the executables without colliding with those objects.
set(CAMP_BUILD_FILES
        camperror path drawpath drawlabel picture psfile texfile util
        guide flatguide knot drawfill path3 drawpath3 drawsurface
        beziercurve bezierpatch pen pipestream
)

set(RUNTIME_BUILD_FILES
        runtime runbacktrace runpicture runlabel runhistory runarray
        runfile runsystem runpair runtriple runpath runpath3d runstring
        runmath
)

set(SYMBOL_STATIC_BUILD_FILES types builtin gsl)
set(SYMBOL_BUILD_FILES ${RUNTIME_BUILD_FILES} ${SYMBOL_STATIC_BUILD_FILES})

set(GENERATED_SOURCE_BUILD_FILES
        ${SYMBOL_BUILD_FILES} camp.tab lex.yy
)

set(CORE_BUILD_FILES
        ${CAMP_BUILD_FILES} ${SYMBOL_STATIC_BUILD_FILES}
        env genv stm dec errormsg
        callable name symbol entry exp newexp stack exithandlers
        access virtualfieldaccess absyn record interact fileio hashing random
        fftw++asy parallel simpson coder coenv impdatum locate asyparser program application
        varinit fundec refaccess envcompleter asyprocess constructor array memory
        Delaunay predicates jsfile v3dfile EXRFiles
        lspserv symbolmaps win32helpers win32pipestream
        win32xdr xstream
        glfw renderBase rendererloader norender
        lspdec lspexp lspfundec lspstm
)

# Vulkan renderer sources. On a Unix Vulkan build these are compiled
# exclusively into the dlopened libasyvulkan.so (see
# cmake-scripts/renderer-shared-libs.cmake), so they must NOT also go into
# asycore: doing so pulls glslang into the main binary, whose bison parser
# collides with asymptote's own (`multiple definition of yydebug`) once asycore
# is whole-archived into asy.
#
# Everywhere else they belong in the core build: on Windows (including Cygwin)
# Vulkan is linked directly into the binary, and when Vulkan is disabled there
# is no separate # renderer library to hold them.
if (WIN32 OR NOT UNIX OR NOT ENABLE_VULKAN)
    list(APPEND CORE_BUILD_FILES vkrender vkutils vkdispatchstorage)
endif()

set(ASY_CSV_ENUM_FILES v3dtypes v3dheadertypes)
