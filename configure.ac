#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[0.62cvs],[http://sourceforge.net/projects/asymptote])
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
BISON=bison
AC_CHECK_PROG(BISON,bison)
AC_PROG_LEX
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_MAKE_SET

CFLAGS="-g -O3"

AC_CHECK_FUNC(getopt_long_only, AC_DEFINE(HAVE_GNU_GETOPT_H, 1,
			   [Define if getopt.h is the GNU version]),
			   getopt="getopt getopt1",)
AC_SUBST(getopt)

AC_CHECK_HEADER(fftw3.h,
   AC_CHECK_LIB([fftw3], fftw_execute,,
           AC_MSG_WARN([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_WARN([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))

# Currently only the boost headers are used
AC_CHECK_HEADER([boost/shared_ptr.hpp],,
    AC_CHECK_FILE(["boost/shared_ptr.hpp"],,
    AC_MSG_ERROR([*** could not find the boost headers ***
*** Please install boost_1_32_0 or later from http://www.boost.org ***
*** (it is sufficient to move or link the boost_1_32_0/boost directory into .) ***])))

# Checks for libraries.
AC_CHECK_LIB([curses], [tparm])
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_LIB([readline], [readline])
AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_HEADER_STDBOOL
AC_CHECK_HEADERS([fenv.h])
AC_CHECK_HEADERS([stddef.h])

AC_COMPILE_IFELSE(AC_LANG_PROGRAM([#include "xstream.h"]),
                   AC_DEFINE(HAVE_RPC_RPC_H, 1,
		   [Define if you have a working <rpc/rpc.h> header file]),
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
AC_CHECK_FUNCS([dup2 floor memset pow sqrt])
AC_FUNC_STRFTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_FUNC_REALLOC

AC_CONFIG_HEADERS(config.h)

AC_CONFIG_FILES([Makefile doc/version.texi])
AC_OUTPUT
