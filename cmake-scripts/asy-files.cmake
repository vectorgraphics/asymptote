set(ASYMPTOTE_INCLUDES ${ASY_INCLUDE_DIR})
set(CAMP_BUILD_FILES
        camperror path drawpath drawlabel picture psfile texfile util settings
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
        callable name symbol entry exp newexp stack
        access virtualfieldaccess absyn record interact fileio
        fftw++asy parallel simpson coder coenv impdatum locate parser program application
        varinit fundec refaccess envcompleter process constructor array
        Delaunay predicates glrender tr shaders jsfile v3dfile
        EXRFiles GLTextures lspserv symbolmaps win32helpers win32pipestream
)

set(ASY_CSV_ENUM_FILES v3dtypes v3dheadertypes)
