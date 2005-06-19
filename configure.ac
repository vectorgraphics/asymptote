#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[0.81cvs],[http://sourceforge.net/projects/asymptote])
VERSION=$PACKAGE_VERSION
AC_SUBST(VERSION)

test "$prefix" = NONE && prefix=/usr/local
Datadir=$datadir
test "$datadir" = '${prefix}/share' && Datadir=$prefix/share

latexdir=/usr/share/texmf/tex/latex
AC_ARG_WITH(latex,
	[  --with-latex=PATH       specify path to LaTeX installation ],
                     [if test "x$withval" != "x" ; then
			latexdir=$withval
		      fi
		     ])
latexdir=$latexdir/asymptote
AC_SUBST(latexdir)
AC_DEFINE_UNQUOTED(ASYMPTOTE_SYSDIR,"$Datadir/asymptote",
	[System directory for global .asy files])

AC_CONFIG_SRCDIR([absyn.cc])

AC_LANG([C++])

# Checks for programs.
AC_PROG_LEX
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_MAKE_SET
AC_PROG_YACC

AC_CHECK_HEADER(fftw3.h,
   AC_CHECK_LIB([fftw3], fftw_execute,,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))

GCVERSION=6.5

AC_ARG_ENABLE(gc,
	[  --enable-gc[[=VERSION]]   Use single-threaded Boehm garbage collector (default)]
	[     [[=system]]    Use system Boehm garbage collector],
			[
			if test "x$enableval" = "xyes"; then
			   ac_cv_use_gc=$GCVERSION
			else 
			   ac_cv_use_gc=$enableval
			fi
			],[ac_cv_use_gc=$GCVERSION])

USEGC=
GCLIBS=
GCNAME="Boehm Garbage Collector"
if test "x$ac_cv_use_gc" != "xno" ; then
   USEGC="-DUSEGC"
   if test "x$ac_cv_use_gc" = "xsystem"; then
      AC_CHECK_HEADER(gc.h,
	AC_CHECK_LIB([gc],[GC_malloc],[LIBS=$LIBS"-lgc"
	AC_MSG_NOTICE([enabling system $GCNAME])],[
	AC_MSG_ERROR($GCNAME library not found)
	USEGC=]),AC_MSG_ERROR($GCNAME header file not found))
   else
      GCDIR=gc$ac_cv_use_gc
      AC_MSG_NOTICE([enabling the $GCNAME $GCDIR])
      GCLIBS="\$(GC)/.libs/libgc.a"
      INCL="-I \$(GC)/include"
   fi
else
   GCDIR=
   AC_MSG_NOTICE([disabling the $GCNAME])
fi

CFLAGS="-g -O3"

AC_CHECK_FUNC(getopt_long_only, AC_DEFINE(HAVE_GNU_GETOPT_H, 1,
			   [Define if getopt.h is the GNU version]),
			   getopt="getopt getopt1",)
AC_SUBST(getopt)
AC_SUBST(GCDIR)
AC_SUBST(GCLIBS)
AC_SUBST(INCL)
AC_SUBST(USEGC)

# Checks for libraries.
AC_CHECK_LIB([curses], [tparm])
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_LIB([readline], [readline])
AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_HEADER_STDBOOL
AC_CHECK_HEADERS([fenv.h stddef.h libintl.h])

AC_COMPILE_IFELSE(AC_LANG_PROGRAM([#include "xstream.h"]),
                   [AC_SEARCH_LIBS([xdrstdio_create],[nsl])
		    AC_DEFINE(HAVE_RPC_RPC_H, 1,
		   [Define if you have a working <rpc/rpc.h> header file])],
   AC_MSG_WARN([*** Broken rpc headers; XDR support disabled ***]))

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([ptrdiff_t])
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIGNAL

# Checks for library functions.
AC_FUNC_FORK
AC_CHECK_FUNCS([dup2 floor memset pow sqrt strchr])
AC_FUNC_STRFTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_REALLOC

AC_CONFIG_HEADERS(config.h)

AC_CONFIG_FILES([Makefile doc/version.texi])
AC_OUTPUT

if test "x$GCDIR" != "x" ; then
   AC_CHECK_FILE($GCDIR.tar.gz,,[
   echo
   echo You need to obtain the Boehm garbage collector source code.
   echo For example, use the command:
   echo
   echo wget http://www.hpl.hp.com/personal/Hans_Boehm/gc/gc_source/$GCDIR.tar.gz
   echo
   echo Or ./configure --enable-gc=system to use a slower system version of gc
   exit 1
   ])
fi
