#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[1.35svn],[http://sourceforge.net/projects/asymptote])
VERSION=$PACKAGE_VERSION
AC_SUBST(VERSION)

test "$prefix" = NONE && prefix=/usr/local
Datadir=$datadir
test "$Datadir" = '${datarootdir}' && Datadir=$datarootdir
test "$Datadir" = '${prefix}/share' && Datadir=$prefix/share
AC_SUBST(Datadir)

AC_ARG_WITH(latex,
	[  --with-latex=PATH       specify path to LaTeX installation ],
                     [if test "x$withval" != "x" ; then
			latexdir=$withval
		      fi
		     ],[
AC_CHECK_PROG(kpsewhich,kpsewhich,true)
if test "x$kpsewhich" = "xtrue"; then
  latexdir=`kpsewhich base/latex.ltx`
  latexdir=`echo $latexdir | sed -e 's;/base/latex.ltx;;'`
else
  latexdir=$prefix/share/texmf/tex/latex
  AC_CHECK_FILE($latexdir/base/latex.ltx,,
  [latexdir=/usr/share/texmf/tex/latex
  AC_CHECK_FILE($latexdir/base/latex.ltx,,)])
fi
])

AC_CHECK_PROGS(TEXI2DVI,[texi2dvi texi2dvi4a2ps])
AC_SUBST(TEXI2DVI)

latexdir=$latexdir/asymptote

AC_MSG_NOTICE([Using $latexdir for LaTeX style file])

AC_SUBST(latexdir)

docdir=$Datadir/doc/asymptote

AC_ARG_WITH(docdir,
	[  --with-docdir=PATH      alternate documentation installation directory],
                     [if test "x$withval" != "x" ; then
			docdir=$withval
		      fi
		     ])
AC_SUBST(docdir)

AC_DEFINE_UNQUOTED(ASYMPTOTE_SYSDIR,"$Datadir/asymptote",
	[System directory for global .asy files])
AC_DEFINE_UNQUOTED(ASYMPTOTE_DOCDIR,"$docdir",
	[Directory for documentation])

AC_CONFIG_SRCDIR([absyn.cc])

AC_LANG([C++])

test "$CFLAGS" || CFLAGS="-g -O3"

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

AC_CHECK_HEADER(fftw3.h,
   AC_CHECK_LIB([fftw3], fftw_execute,,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))

GCVERSION=gc-7.0

AC_ARG_ENABLE(gc,
	[  --enable-gc[[=VERSION]]   use single-threaded Boehm garbage collector (default)]
	[     [[=PREFIX]]    use (typically slower) system Boehm garbage collector]
	[                  installed in PREFIX ("system" uses $prefix/include/gc)],
			[
			if test "x$enableval" = "xyes"; then
			   ac_cv_use_gc=$GCVERSION
			else 
			   ac_cv_use_gc=$enableval
			fi
			],[ac_cv_use_gc=$GCVERSION])

OPTIONS="-D_FILE_OFFSET_BITS=64 "
GCLIB=
GCPPLIB=
GCNAME="Boehm Garbage Collector"
GCOPTIONS="--disable-threads --disable-shared "
if test "x$ac_cv_use_gc" != "xno" ; then
   OPTIONS=$OPTIONS"-DUSEGC "
   case $ac_cv_use_gc in
   system|*[[\\/]]*)
      if test "x$ac_cv_use_gc" = "xsystem"
      then INCL="-I$prefix/include/gc" 
      else INCL="-I$ac_cv_use_gc/include/gc"
      fi
      AC_CHECK_HEADER(gc.h,
	AC_CHECK_LIB([gc],[GC_malloc],[
	  LIBS=$LIBS"-lgc "
#	  LIBS=$LIBS"-lgccpp -lgc "
	  AC_MSG_NOTICE([enabling system $GCNAME])],[
	AC_MSG_ERROR($GCNAME library not found)]),
      AC_MSG_ERROR($GCNAME header file not found))
      ;;
   *)
      GCDIR=$ac_cv_use_gc
      AC_MSG_NOTICE([enabling the $GCNAME $GCDIR])
      GCLIB="\$(GC)/.libs/libgc.a"
#      GCPPLIB="\$(GC)/.libs/libgccpp.a"
      INCL="-I\$(GC)/include"
      ;;
   esac
else
   GCDIR=
   AC_MSG_NOTICE([disabling the $GCNAME])
fi

AC_ARG_ENABLE(gc-debug,
[  --enable-gc-debug       enable (slow) garbage collector debugging],
[ if test "x$ac_cv_use_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
       OPTIONS=$OPTIONS"-DGC_DEBUG "
       AC_MSG_NOTICE([*** Enabling GC debugging: remember to make clean ***])
       AC_MSG_NOTICE([*** Set the environment variable GC_FIND_LEAK at runtime ***])
    fi
 fi
])

AC_ARG_ENABLE(gc-full-debug,
[  --enable-gc-full-debug  enable (very slow) garbage collector backtrace],
[ if test "x$ac_cv_use_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
       OPTIONS=$OPTIONS"-DGC_DEBUG -DGC_BACKTRACE "
       GCOPTIONS=$GCOPTIONS"--enable-gc-debug "
       AC_MSG_NOTICE([*** Enabling GC backtrace debugging; remember to make gc-clean ***])
    fi
 fi
])

AC_CHECK_FUNC(getopt_long_only, AC_DEFINE(HAVE_GNU_GETOPT_H, 1,
			   [Define if getopt.h is the GNU version]),
			   getopt="getopt getopt1",)
AC_CHECK_HEADER(ext/hash_map,,OPTIONS=$OPTIONS"-DNOHASH ")

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
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])

AC_CHECK_LIB([readline], [rl_completion_matches],,
AC_MSG_NOTICE(*** Could not find GNU readline 4.2 or later: will compile without readline support ***))

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([fenv.h stddef.h libintl.h])
AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h], [break])

AC_COMPILE_IFELSE(AC_LANG_PROGRAM([#include "xstream.h"]),
                   [AC_SEARCH_LIBS([xdrstdio_create],[nsl rpc])
		    AC_DEFINE(HAVE_RPC_RPC_H, 1,
		   [Define if you have a working <rpc/rpc.h> header file])],
   AC_MSG_WARN([*** Broken rpc headers; XDR support disabled ***]))

AC_CHECK_HEADER(gsl/gsl_sf.h,
   AC_CHECK_LIB([gsl], gsl_sf_Si, [AC_DEFINE(HAVE_LIBGSL, 1,
   [Define to 1 if you have the 'gsl' library (-lgsl). ]) LIBS=$LIBS"-lgsl -lgslcblas "],
           AC_MSG_NOTICE([*** Could not find libgsl: will compile without optional special functions. ***]),[-lgslcblas]),
     AC_MSG_NOTICE([*** Header file gsl_sf.h not found: will compile without optional special functions. ***]))

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([ptrdiff_t])
AC_CHECK_TYPES([long long])
AC_CHECK_TYPES([long])
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIGNAL

AC_DEFUN(AC_FUNC_STRPTIME, [
    AC_CHECK_FUNCS(strptime)
    AC_MSG_CHECKING(for strptime declaration in time.h)
    AC_EGREP_HEADER(strptime, time.h, [
        AC_DEFINE([HAVE_STRPTIME_DECL],,[Define if the function strptime is declared in <time.h>])
    AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)
    ])
])

# Checks for library functions.
AC_FUNC_FORK
AC_CHECK_FUNCS([dup2 floor memset pow sqrt strchr tgamma])
AC_FUNC_STRFTIME
AC_FUNC_STRPTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FSEEKO

AC_CONFIG_HEADERS(config.h)

AC_CONFIG_FILES([Makefile doc/Makefile doc/version.texi base/version.asy])
AC_OUTPUT

if test "x$GCDIR" != "x" ; then
   AC_CHECK_FILE($GCDIR.tar.gz,,[
   echo
   echo Please put the Boehm garbage collector tar.gz file in the current directory.
   echo FOR EXAMPLE, USE THE COMMAND:
   echo
   echo wget http://www.hpl.hp.com/personal/Hans_Boehm/gc/gc_source/$GCDIR.tar.gz
   echo
   echo Or, to use a slower system version of gc:
   echo ./configure --enable-gc=system
   exit 1
   ])
fi
