#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[0.72cvs],[http://sourceforge.net/projects/asymptote])
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
AC_PROG_YACC

CFLAGS="-g -O3"

AC_CHECK_FUNC(getopt_long_only, AC_DEFINE(HAVE_GNU_GETOPT_H, 1,
			   [Define if getopt.h is the GNU version]),
			   getopt="getopt getopt1",)
AC_SUBST(getopt)

AC_CHECK_HEADER(fftw3.h,
   AC_CHECK_LIB([fftw3], fftw_execute,,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))

AC_MSG_CHECKING([for working boost::filesystem library])

AC_ARG_WITH([boost-filesystem],[  --with-boost-filesystem specify the boost filesystem library or suffix to use],
  [if test "x$with_boost_filesystem" != "xno"; then
    ax_filesystem_lib=$with_boost_filesystem
    ax_boost_filesystem_lib=boost_filesystem-$with_boost_filesystem
  fi])
 
LIBS_SAVED=$LIBS
BOOST_FILE_SYSTEM_LIB=
for LIBS in -l$ax_filesystem_lib -l$ax_boost_filesystem_lib -lboost_filesystem-$CC -lboost_filesystem -lboost_filesystem-gcc;
do 
   if test "$LIBS" != "-l"; then
      AC_LINK_IFELSE(AC_LANG_PROGRAM([[#include <boost/filesystem/path.hpp>]],
          [[boost::filesystem::path p("");]]),
	  [AC_MSG_RESULT([$LIBS])
	  BOOST_FILESYSTEM_LIB=$LIBS break],)
   fi
done

if test "$BOOST_FILESYSTEM_LIB" == ""; then
AC_MSG_RESULT([no])
CXXFLAGS="-I ."
AC_COMPILE_IFELSE(AC_LANG_PROGRAM([[#include "libboost/operations_posix_windows.cc"]],),
boostlib="libboost/path_posix_windows libboost/operations_posix_windows libboost/exception libboost/convenience"
,
AC_MSG_ERROR([*** could not find a working boost library ***
*** Please install boost_1_32_0 or later from http://www.boost.org ***
*** (it is sufficient to move or link the boost_1_32_0/boost directory into .) ***]))
fi

AC_SUBST(boostlib)
LIBS="$LIBS_SAVED$BOOST_FILESYSTEM_LIB "

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
