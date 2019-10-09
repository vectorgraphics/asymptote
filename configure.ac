#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[2.58git],[http://sourceforge.net/projects/asymptote])
VERSION=$PACKAGE_VERSION
AC_SUBST(VERSION)
m4_include([ax_pthread.m4])

test "$CXXFLAGS" || CXXFLAGS="-std=c++11"

test "$CFLAGS" || CFLAGS="-g -O3"

AC_C_BIGENDIAN
AC_CANONICAL_HOST

test "$prefix" = NONE && prefix=/usr/local
Datadir=$datadir
test "$Datadir" = '${datarootdir}' && Datadir=$datarootdir
test "$Datadir" = '${prefix}/share' && Datadir=$prefix/share
AC_SUBST(Datadir)

AC_ARG_WITH(latex,
        [AS_HELP_STRING(--with-latex=PATH,
                        specify path to LaTeX installation)],
                     [if test "x$withval" != "x" ; then
                        latexdir=$withval
                      fi
                     ],[
AC_CHECK_PROG(kpsewhich,kpsewhich,true)
if test "x$kpsewhich" = "xtrue"; then
  latexdir=`kpsewhich -expand-var='$TEXMFLOCAL'/tex/latex`
else
  latexdir=$prefix/share/texmf/tex/latex
  AC_CHECK_FILE($latexdir/base/latex.ltx,,
  [latexdir=/usr/share/texmf/tex/latex
  AC_CHECK_FILE($latexdir/base/latex.ltx,,)])
fi
])

AC_ARG_WITH(context,
        [AS_HELP_STRING(--with-context=PATH,
                        specify path to ConTeXt installation)],
                     [if test "x$withval" != "x" ; then
                        contextdir=$withval
                      fi
                     ],[
AC_CHECK_PROG(kpsewhich,kpsewhich,true)
if test "x$kpsewhich" = "xtrue"; then
  contextdir=`kpsewhich -expand-var='$TEXMFLOCAL'/tex/context/third`
else
  contextdir=$prefix/share/texmf/tex/context/third
fi
])

AC_CHECK_PROGS(TEXI2DVI,[texi2dvi],
[@echo \*\*\*\*\*\*\*\*\*\* Please install texi2dvi or put http://asymptote.sourceforge.net/asymptote.pdf in the doc directory: cannot execute texi2dvi])

AC_SUBST(TEXI2DVI)

latexdir=$latexdir/asymptote
contextdir=$contextdir/asymptote

AC_MSG_NOTICE([Using $latexdir for LaTeX style file])
AC_MSG_NOTICE([Using $contextdir for ConTeXT style file])

AC_SUBST(latexdir)
AC_SUBST(contextdir)

docdir=$Datadir/doc/asymptote

AC_ARG_WITH(docdir,
        [AS_HELP_STRING(--with-docdir=PATH,
alternate documentation installation directory)],
                     [if test "x$withval" != "x" ; then
                        docdir=$withval
                      fi
                     ])
AC_SUBST(docdir)

sysdir=$Datadir/asymptote

AC_ARG_ENABLE(texlive-build,
[AS_HELP_STRING(--enable-texlive-build,
automatically determine sysdir from kpsewhich)],
[ if test "x$enableval" = "xyes" ; then
    sysdir=""
  fi
])

AC_DEFINE_UNQUOTED(ASYMPTOTE_SYSDIR,"$sysdir",
        [System directory for global .asy files])
AC_DEFINE_UNQUOTED(ASYMPTOTE_DOCDIR,"$docdir",
        [Directory for documentation])

AC_CONFIG_SRCDIR([absyn.cc])

AC_LANG([C++])

# Checks for programs.
AC_PROG_LEX
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_MAKE_SET
AC_PROG_YACC

if test "$GXX" = yes ; then
   ac_gcc_version=`echo __GNUC__ | $CC -E - | grep -v ^\#`
   ac_clang=`echo __clang__ | $CC -E - | grep -v ^\#`
   if test "$ac_gcc_version" -lt 4; then
       CFLAGS=$CFLAGS" -finline-limit=400"
   else
     if test "$ac_clang" != 1; then
         CFLAGS=$CFLAGS" -fno-var-tracking"
     fi
   fi
fi

AC_DEFUN([DEFINE],[
Define to 1 if you have $1.
])

AC_DEFUN([DEFINE_LIB],[
Define to 1 if you have the `$1' library (-l$1).
])

AC_CHECK_HEADER(tr1/unordered_map)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
    [#include <tr1/unordered_map>]
    [std::tr1::unordered_map<int,int> map;]
  )],
  AC_DEFINE(HAVE_TR1_UNORDERED_MAP,1,
  DEFINE([<tr1/unordered_map>])),
  [
  AC_CHECK_HEADER(unordered_map,AC_DEFINE(HAVE_UNORDERED_MAP,1,
                   DEFINE([<unordered_map>])),
  [AC_CHECK_HEADER(ext/hash_map,,OPTIONS=$OPTIONS"-DNOHASH ")])])

ASYGLVERSION=1.00

GCVERSION=8.0.4
ATOMICVERSION=7.6.10

GCFILE=gc-$GCVERSION
ac_cv_use_gc="system"

AC_CHECK_FILE($GCFILE.tar.gz,
ac_cv_use_gc=$GCVERSION)
AC_ARG_ENABLE(gc,
        [AS_HELP_STRING(--enable-gc[[[=system]]],
enable system Boehm garbage collector)]
        [     [[=VERSION]]   enable local VERSION of Boehm garbage collector]
        [     [[=PREFIX]]    use Boehm garbage collector installed in PREFIX],
                        [
                        if test "x$enableval" != "xyes" ; then
                           ac_cv_use_gc=$enableval
                        fi
                        ])

OPTIONS="-D_FILE_OFFSET_BITS=64 "
GCLIB=
GCPPLIB=
INCL="-I."
GCNAME="Boehm Garbage Collector"
GCOPTIONS="--disable-shared"
if test "x$ac_cv_use_gc" != "xno" ; then
   OPTIONS=$OPTIONS"-DUSEGC "
   case _$ac_cv_use_gc in
   _|_system|_*[[\\/]]*)
      if test "x$ac_cv_use_gc" = "xsystem" ; then
         INCL="-I. -I$prefix/include/gc -I/usr/include/gc"
         LIBS=$LIBS"-L$prefix/lib "
      else
         INCL="-I$ac_cv_use_gc/include/gc"
         LIBS=$LIBS"-L$ac_cv_use_gc/lib "
      fi
      CPPFLAGS_SAVE=$CPPFLAGS
      CPPFLAGS=$CPPFLAGS" $INCL"
      AC_CHECK_HEADER(gc.h,
        AC_CHECK_LIB([gc],[GC_malloc],[
          LIBS=$LIBS"-lgc "
          AC_MSG_NOTICE([enabling system $GCNAME])],[
        GCDIR=$GCFILE
        INCL="-I. -I\$(GC)/include"
        GCLIB="\$(GC)/.libs/libgc.a"
        AC_MSG_NOTICE($GCNAME library not found)]),
      GCDIR=$GCFILE
      GCLIB="\$(GC)/.libs/libgc.a"
      INCL="-I. -I\$(GC)/include"
      AC_MSG_NOTICE($GCNAME header file not found))
      CPPFLAGS=$CPPFLAGS_SAVE
      ;;
   *)
      GCVERSION=$ac_cv_use_gc
      GCFILE=gc-$GCVERSION
      GCDIR=$GCFILE
      AC_MSG_NOTICE([enabling local $GCNAME $GCDIR])
      GCLIB="\$(GC)/.libs/libgc.a"
      INCL="-I. -I\$(GC)/include"
      ;;
   esac
else
   AC_MSG_NOTICE([disabling the $GCNAME])
fi

AC_ARG_ENABLE(gc-debug,
[AS_HELP_STRING(--enable-gc-debug,enable (slow) garbage collector debugging)],
[ if test "x$ac_cv_use_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
       OPTIONS=$OPTIONS"-DGC_DEBUG "
       AC_MSG_NOTICE([*** Enabling GC debugging: remember to make clean ***])
       AC_MSG_NOTICE([*** Set the environment variable GC_FIND_LEAK at runtime ***])
    fi
 fi
])

AC_ARG_ENABLE(gc-full-debug,
[AS_HELP_STRING(--enable-gc-full-debug,enable (very slow) garbage collector backtrace)],
[ if test "x$ac_cv_use_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
       OPTIONS=$OPTIONS"-DGC_DEBUG -DGC_BACKTRACE "
       GCOPTIONS=$GCOPTIONS"--enable-gc-debug "
       AC_MSG_NOTICE([*** Enabling GC backtrace debugging; remember to make gc-clean ***])
    fi
 fi
])

INCL=$INCL" -I/usr/include/tirpc"

if test "$OSTYPE" = "msdos"; then
CPPFLAGS=$CPPFLAGS" -D__MSDOS__ -I/usr/include/w32api -I/usr/include -DCALLBACK=__stdcall $INCL"
else
CPPFLAGS=$CPPFLAGS" $INCL"
fi

AC_CHECK_FUNC(getopt_long_only, AC_DEFINE(HAVE_GNU_GETOPT_H, 1,
                           DEFINE([GNU <getopt.h>])),
                           getopt="getopt getopt1",)
AC_SUBST(getopt)
AC_SUBST(GCVERSION)
AC_SUBST(ASYGLVERSION)
AC_SUBST(ATOMICVERSION)
AC_SUBST(GCOPTIONS)
AC_SUBST(GCLIB)
AC_SUBST(GCPPLIB)
AC_SUBST(INCL)
AC_SUBST(OPTIONS)
AC_SUBST(GLEW)

# Checks for libraries.
AC_SEARCH_LIBS([lgamma],[m c],,
AC_MSG_ERROR([*** Please install libm on your system ***]))
AC_CHECK_LIB([z], [deflate],,
AC_MSG_ERROR([*** Please install libz or zlib-devel on your system ***]))
AX_PTHREAD

AC_ARG_ENABLE(sigsegv,
[AS_HELP_STRING(--enable-sigsegv[[[=yes]]],enable GNU Stack Overflow Handler)])

if test "x$enable_sigsegv" != "xno"; then
  AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])
fi

AC_CHECK_LIB([rt], [sched_yield])

AC_ARG_ENABLE(readline,
[AS_HELP_STRING(--enable-readline[[[=yes]]],enable GNU Readline Library)])

AC_ARG_ENABLE(static,
[AS_HELP_STRING(--enable-static[[[=no]]],link against static libraries)])

LDSTATIC=""
STATIC=""
DYNAMIC=""
if test "x$enable_static" = "xyes"; then
  LDSTATIC="-static "
  STATIC="-Wl,-Bstatic "
  DYNAMIC="-Wl,-Bdynamic "
fi

AC_DEFUN([READLINE],[
AC_MSG_NOTICE([*** Could not find GNU readline 4.3 or later: will compile without readline support ***])
])

AC_DEFUN([CHECK_FOUND_STATIC],[
AC_DEFINE($1,1,DEFINE_LIB($2))
LIBS=$LIBS$STATIC"-l$2 "$DYNAMIC
])

AC_DEFUN([AC_CHECK_LIB_STATIC],[
LDFLAGS0=$LDFLAGS
LDFLAGS=$LDFLAGS$LDSTATIC
AC_CHECK_LIB($1,$2,CHECK_FOUND_STATIC($3,$1),
if test "x$enable_static" = "xyes"; then
[
AS_UNSET([ac_cv_lib_$1_$2])
LDFLAGS=$LDFLAGS0
AC_CHECK_LIB($1,$2,,$4)
]
else
$4
fi
)
LDFLAGS=$LDFLAGS0
])

if test "x$enable_readline" != "xno"; then
AC_COMPILE_IFELSE([
AC_LANG_PROGRAM([
#include <stdio.h>
#include <unistd.h>
#include <readline/readline.h>
],[
#ifndef RL_READLINE_VERSION
abort
#endif
])],
AC_CHECK_LIB_STATIC(readline,history_list,HAVE_LIBREADLINE,READLINE),
READLINE)

AC_CHECK_LIB_STATIC(tinfo,tgetent,HAVE_LIBTINFO,AC_MSG_NOTICE([perhaps tgetent is in -lncurses]))

AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])
AC_CHECK_LIB_STATIC([ncurses],[setupterm],HAVE_LIBCURSES,
                    AC_CHECK_LIB([curses],[setupterm]))
fi

AC_ARG_ENABLE(fftw,
[AS_HELP_STRING(--enable-fftw[[[=yes]]],enable FFTW Library)])

if test "x$enable_fftw" != "xno"; then

AC_CHECK_HEADER(fftw3.h,
AC_CHECK_LIB_STATIC([fftw3],[fftw_execute],HAVE_LIBFFTW3,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))
fi

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([fenv.h stddef.h libintl.h])
AC_CHECK_HEADERS(fpu_control.h)
AC_CHECK_FUNCS([feenableexcept])


AC_COMPILE_IFELSE([AC_LANG_PROGRAM([#include "xstream.h"])],
                   [AC_SEARCH_LIBS([xdrstdio_create],[nsl tirpc])
                    AC_DEFINE(HAVE_RPC_RPC_H,1,
                   DEFINE([a working <tirpc/rpc/rpc.h> header]))],
   AC_MSG_WARN([*** Broken rpc headers; XDR support disabled ***]))

AC_ARG_ENABLE(gsl,
[AS_HELP_STRING(--enable-gsl[[[=yes]]],enable GNU Scientific Library)])

if test "x$enable_gsl" != "xno"; then
  AC_CHECK_HEADER(gsl/gsl_sf.h,
   AC_CHECK_LIB([gsl], gsl_sf_debye_6, [AC_DEFINE(HAVE_LIBGSL,1,
   DEFINE_LIB[gsl]) LIBS=$LIBS"-lgsl -lgslcblas "],
           AC_MSG_NOTICE([*** Could not find libgsl: will compile without optional special functions. ***]),[-lgslcblas]),
     AC_MSG_NOTICE([*** Header file gsl_sf.h not found: will compile without optional special functions. ***]))
fi

AC_ARG_ENABLE(gl,
[AS_HELP_STRING(--enable-gl[[[=yes]]],enable OpenGL Library)])

AC_ARG_ENABLE(offscreen,
[AS_HELP_STRING(--enable-offscreen[[[=no]]],enable experimental offscreen rendering using OSMesa library)])

AC_ARG_ENABLE(OpenImageIO,
[AS_HELP_STRING(--enable-openimageio[[[=no]]],enable experimental OpenImageIO Library)])

            AC_CHECK_HEADER(glm/glm.hpp,
                       [AC_DEFINE(HAVE_LIBGLM,1,
                    DEFINE([the <glm/glm.hpp> header]))],AC_MSG_NOTICE([*** Could not find glm header files: will compile without WebGL or OpenGL support ***]))

if test "x$enable_gl" != "xno"; then
                    AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])
if test "x$enable_openimageio" = "xyes"; then
            AC_CHECK_HEADER([OpenEXR/ImathVec.h],
                AC_CHECK_HEADER([OpenImageIO/imageio.h],
                        AC_CHECK_LIB([OpenImageIO],[open])))
fi

case "$OSTYPE" in
     msdos)
              AC_CHECK_LIB([gccpp],[GC_throw_bad_alloc])
              AC_CHECK_HEADER(GL/glut.h,
                                  [AC_DEFINE(HAVE_LIBGLUT,1,
                     DEFINE_LIB[freeglut])
                             LIBS=$LIBS"-lfreeglut "],
                                   AC_MSG_NOTICE([*** Could not find libfreeglut: will compile without OpenGL support ***]))
              AC_CHECK_HEADER(GL/gl.h,
                                  [AC_DEFINE(HAVE_LIBGL,1,
                     DEFINE_LIB[opengl32])
                             LIBS=$LIBS"-lopengl32 glew.o"],
                                   AC_MSG_NOTICE([*** Could not find libopengl32: will compile without OpenGL support ***]))
        ;;
     darwin*)
              AC_CHECK_LIB([gccpp],[GC_throw_bad_alloc])
              AC_CHECK_HEADER(OpenGL/gl.h,
                             [AC_DEFINE(HAVE_LIBGL,1,
                             DEFINE([<gl.h> header]))])
              AC_CHECK_HEADER(GLUT/glut.h, [AC_DEFINE(HAVE_LIBGLUT,1,
                   DEFINE_LIB[GLUT])
                           LIBS=$LIBS"-framework GLUT -framework OpenGL -framework Cocoa glew.o "],
                                 AC_MSG_NOTICE([*** Could not find GLUT: will compile without OpenGLLUT support ***]))
        ;;
     *)
         AC_CHECK_LIB([gccpp],[GC_throw_bad_alloc])
          AC_CHECK_LIB([glut], [glutMainLoop],,
                       AC_MSG_NOTICE([*** Could not find libglut: will compile without OpenGL support ***]))
         AC_CHECK_LIB([GL], [glDepthMask],
                                 [AC_DEFINE(HAVE_LIBGL,1,
                             DEFINE_LIB([GL]))
                             LIBS=$LIBS"-lGL "
                             GLEW="glew.o "
                             AC_CHECK_LIB([GLX],[glXGetProcAddressARB],
                             GLEW=$GLEW"-lGLX ")],
                                   AC_MSG_NOTICE([*** Could not find libGL: will compile without OpenGL support ***]))
esac
   if test "x$enable_offscreen" = "xyes"; then
           AC_CHECK_LIB([OSMesa],OSMesaCreateContext,,
                        AC_MSG_NOTICE([*** Could not find libOSMesa: will compile without offscreen rendering support ***]))
   fi
fi

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([ptrdiff_t])
AC_CHECK_TYPES([long long])
AC_CHECK_TYPES([long])
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIGNAL

AC_DEFUN([ac_FUNC_STRPTIME], [
    AC_CHECK_FUNCS(strptime)
])

# Checks for library functions.
AC_FUNC_FORK
AC_CHECK_FUNCS([dup2 floor memset strchr tgamma lgamma memrchr])
AC_FUNC_STRFTIME
ac_FUNC_STRPTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FSEEKO

AC_CONFIG_HEADERS(config.h)

AC_CONFIG_FILES([Makefile doc/Makefile doc/png/Makefile])
AC_OUTPUT

if test "x$GCDIR" != "x" ; then
   AC_CHECK_FILE($GCDIR.tar.gz,,[
   echo
   echo Please put the Boehm garbage collector tar.gz files in the asymptote directory.
   echo FOR EXAMPLE, USE THE COMMANDS:
   echo
   echo wget https://github.com/ivmai/bdwgc/releases/download/v$GCVERSION/$GCFILE.tar.gz
   echo wget http://www.ivmaisoft.com/_bin/atomic_ops/libatomic_ops-$ATOMICVERSION.tar.gz

   echo
   exit 1
   ])
fi
