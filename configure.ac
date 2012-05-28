#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[2.16svn],[http://sourceforge.net/projects/asymptote])
VERSION=$PACKAGE_VERSION
AC_SUBST(VERSION)
m4_include([ax_pthread.m4])

test "$CFLAGS" || CFLAGS="-g -O3 -ansi"

AC_C_BIGENDIAN

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

AC_CHECK_PROGS(TEXI2DVI,[texi2dvi texi2dvi4a2ps],
[echo texi2dvi is missing! Please put http://asymptote.sourceforge.net/asymptote.pdf in the doc directory and touch doc/asymptote.pdf; exit 1;])

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
   if test "$ac_gcc_version" -lt 4; then
       CFLAGS=$CFLAGS" -finline-limit=400"
   fi
fi

AC_CHECK_HEADER(tr1/unordered_map,,
		[AC_CHECK_HEADER(ext/hash_map,,OPTIONS=$OPTIONS"-DNOHASH ")])

GCVERSION=gc-7.2b

AC_CHECK_FILE($GCVERSION.tar.gz,
ac_cv_use_gc=$GCVERSION,
[GC_SYSTEM="$GCVERSION.tar.gz not found: using system Boehm garbage collector"
ac_cv_use_gc="system"])
AC_ARG_ENABLE(gc,
	[AS_HELP_STRING(--enable-gc[[[=system]]],
enable system Boehm garbage collector)]
	[     [[=VERSION]]   enable local VERSION of Boehm garbage collector]
	[     [[=PREFIX]]    use Boehm garbage collector installed in PREFIX],
			[
			if test "x$enableval" != "xyes"; then
			   ac_cv_use_gc=$enableval
			   GC_SYSTEM=""
			fi
			])

OPTIONS="-D_FILE_OFFSET_BITS=64 "
GCLIB=
GCPPLIB=
GCNAME="Boehm Garbage Collector"
GCOPTIONS="--disable-shared --enable-large-config"
if test "x$ac_cv_use_gc" != "xno" ; then
   OPTIONS=$OPTIONS"-DUSEGC "
   case $ac_cv_use_gc in
   system|*[[\\/]]*)
      if test "x$ac_cv_use_gc" = "xsystem" ; then
         INCL="-I$prefix/include/gc -I/usr/include/gc"
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
        GCDIR=$GCVERSION
        INCL="-I\$(GC)/include"
	GCLIB="\$(GC)/.libs/libgc.a"
	AC_MSG_NOTICE($GCNAME library not found)]),
      GCDIR=$GCVERSION
      GCLIB="\$(GC)/.libs/libgc.a"
      INCL="-I\$(GC)/include"
      AC_MSG_NOTICE($GCNAME header file not found))
      CPPFLAGS=$CPPFLAGS_SAVE
      ;;
   *)
      GCDIR=$ac_cv_use_gc
      AC_MSG_NOTICE([enabling local $GCNAME $GCDIR])
      GCLIB="\$(GC)/.libs/libgc.a"
      INCL="-I\$(GC)/include"
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

if test "$OSTYPE" = "cygwin"; then
INCL=$INCL" -I/usr/include/tirpc"
CPPFLAGS=$CPPFLAGS" $INCL"
fi

AC_CHECK_FUNC(getopt_long_only, AC_DEFINE(HAVE_GNU_GETOPT_H, 1,
			   [Define if getopt.h is the GNU version]),
			   getopt="getopt getopt1",)

AC_SUBST(getopt)
AC_SUBST(GCDIR)
AC_SUBST(GCOPTIONS)
AC_SUBST(GCLIB)
AC_SUBST(GCPPLIB)
AC_SUBST(INCL)
AC_SUBST(OPTIONS)

# Checks for libraries.
AC_CHECK_LIB([ncurses], [setupterm], [AC_DEFINE(HAVE_LIBCURSES)
			 LIBS=$LIBS"-lncurses "],
			 AC_CHECK_LIB([curses], [setupterm]))
AC_CHECK_LIB([m], [sqrt],,
AC_MSG_ERROR([*** Please install libm on your system ***]))
AC_CHECK_LIB([z], [deflate],,
AC_MSG_ERROR([*** Please install libz on your system ***]))
AX_PTHREAD
AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])
AC_CHECK_LIB([rt], [sched_yield])

AC_ARG_ENABLE(readline,
[AS_HELP_STRING(--enable-readline[[[=yes]]],enable GNU Readline Library)])

if test "x$enable_readline" != "xno"; then
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
#include <stdio.h>
#include <unistd.h>
#include <readline/readline.h>
],[
#ifndef RL_READLINE_VERSION
abort
#endif
])],AC_CHECK_LIB([readline], [rl_completion_matches],,
AC_MSG_NOTICE(*** Could not find GNU readline 4.3 or later: will compile without readline support ***)))
fi

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([fenv.h stddef.h libintl.h])
AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h], [break])
AC_CHECK_HEADERS(fpu_control.h)

AC_COMPILE_IFELSE([AC_LANG_PROGRAM([#include "xstream.h"])],
                   [AC_SEARCH_LIBS([xdrstdio_create],[nsl rpc])
		    AC_DEFINE(HAVE_RPC_RPC_H, 1,
		   [Define if you have a working <rpc/rpc.h> header file])],
   AC_MSG_WARN([*** Broken rpc headers; XDR support disabled ***]))

AC_ARG_ENABLE(gsl,
[AS_HELP_STRING(--enable-gsl[[[=yes]]],enable GNU Scientific Library)])

if test "x$enable_gsl" != "xno"; then
  AC_CHECK_HEADER(gsl/gsl_sf.h,
   AC_CHECK_LIB([gsl], gsl_sf_debye_6, [AC_DEFINE(HAVE_LIBGSL, 1,
   [Define to 1 if you have the `gsl' library (-lgsl).]) LIBS=$LIBS"-lgsl -lgslcblas "],
           AC_MSG_NOTICE([*** Could not find libgsl: will compile without optional special functions. ***]),[-lgslcblas]),
     AC_MSG_NOTICE([*** Header file gsl_sf.h not found: will compile without optional special functions. ***]))
fi

AC_ARG_ENABLE(fftw,
[AS_HELP_STRING(--enable-fftw[[[=yes]]],enable FFTW Library)])

if test "x$enable_fftw" != "xno"; then
  AC_CHECK_HEADER(fftw3.h,
    AC_CHECK_LIB([fftw3], fftw_execute,,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))
fi

AC_ARG_ENABLE(gl,
[AS_HELP_STRING(--enable-gl[[[=yes]]],enable OpenGL Library)])

AC_ARG_ENABLE(offscreen,
[AS_HELP_STRING(--enable-offscreen[[[=no]]],enable offscreen rendering using OSMesa library)])

# First check if GL is enabled. If so, then:
# onscreen rendering requires GL, glut, and GLU libraries
# offscreen rendering require OSMesa and GLU libraries 

if test "x$enable_gl" != "xno"; then
case "$OSTYPE" in
     cygwin)
              if test "x$enable_offscreen" != "xyes"; then
                AC_CHECK_HEADER(GL/gl.h,
				  [AC_DEFINE(HAVE_LIBGL, 1,
		     [Define if you have the `opengl32' library (-lopengl32).])
			     LIBS=$LIBS"-lopengl32 "],
				   AC_MSG_NOTICE([*** Could not find libopengl32: will compile without OpenGL support ***]))
              fi
	    AC_CHECK_LIB([GLU], [gluNewNurbsRenderer],[AC_DEFINE(HAVE_LIBGLU, 1,
		              [Define to 1 if you have the `glu32' library (-lglu32).]) LIBS=$LIBS"-lglu32 "],
	    AC_MSG_NOTICE([*** Could not find libglu32: will compile without OpenGL support ***]))
	       	       AC_CHECK_HEADER(GL/glut.h,
				  [AC_DEFINE(HAVE_LIBGLUT, 1,
		     [Define if you have the `freeglut' library (-lfreeglut).])
			     LIBS=$LIBS"-lfreeglut "],
				   AC_MSG_NOTICE([*** Could not find libfreeglut: will compile without GLUT support ***]))
        ;;                                 
     darwin*) 
              if test "x$enable_offscreen" != "xyes"; then
                AC_CHECK_HEADER(OpenGL/gl.h,
                               [AC_DEFINE(HAVE_LIBGL, 1,
  		               [Define if you have the `OpenGL' library.])])
              fi                             
              AC_CHECK_HEADER(OpenGL/glu.h,
                              [AC_DEFINE(HAVE_LIBGLU, 1,
			      [Define if you have the `GLU' library.])])
              AC_CHECK_HEADER(GLUT/glut.h, [AC_DEFINE(HAVE_LIBGLUT, 1,
		   [Define if you have the `GLUT' library.])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#ifndef GLU_TESS_CALLBACK_TRIPLEDOT
typedef GLvoid (* _GLUfuncptr)(...);
void f(void) {
  gluNurbsCallback(gluNewNurbsRenderer(),GLU_NURBS_BEGIN,(_GLUfuncptr) glBegin);
}
#endif
])],[AC_DEFINE(GLU_TESS_CALLBACK_TRIPLEDOT, 1, [Define if gluNurbsCallback expects a variadic function.])])
			   LIBS=$LIBS"-framework GLUT -framework OpenGL -framework Cocoa"],
				 AC_MSG_NOTICE([*** Could not find glut: will compile without GLUT support ***]))
        ;;                                 
     *)    if test "x$enable_offscreen" != "xyes"; then
             AC_CHECK_LIB([GL], [glDepthMask])
           fi
        AC_CHECK_LIB([GLU], [gluNewNurbsRenderer],,
	             AC_MSG_NOTICE([*** Could not find libGLU: will compile without OpenGL support ***]))
	  AC_CHECK_LIB([glut], [glutMainLoop],,
		       AC_MSG_NOTICE([*** Could not find libglut: will compile without GLUT support ***]))
esac
   if test "x$enable_offscreen" == "xyes"; then
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
AC_CHECK_FUNCS([dup2 floor memset pow sqrt strchr tgamma memrchr])
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
   echo Please put the Boehm garbage collector tar.gz file in the current directory.
   echo FOR EXAMPLE, USE THE COMMAND:
   echo
   echo wget http://www.hpl.hp.com/personal/Hans_Boehm/gc/gc_source/$GCDIR.tar.gz
   exit 1
   ])
fi
