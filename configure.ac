#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ(2)
AC_INIT([Asymptote],[0.53cvs],[asymptote-discuss@lists.sourceforge.net])
test "$prefix" = NONE && prefix=/usr/local
test "$datadir" = '${prefix}/share' && datadir=$prefix/share
asydir=$datadir/asymptote
AC_SUBST(asydir)
AC_DEFINE_UNQUOTED(ASYMPTOTE_SYSDIR,"$asydir",
	[System directory for global .asy files])
AC_CONFIG_SRCDIR([absyn.cc])
AC_LANG([C++])

# Checks for programs.
AC_PROG_YACC
AC_PROG_LEX
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_CC

AC_CHECK_FUNC(getopt_long, AC_DEFINE(HAVE_GETOPT_H, 1,
			   [Define if there is a GNU getopt.h header;
			   otherwise look for gnugetopt.h]),)
AC_SEARCH_LIBS([getopt_long],[gnugetopt])

AC_CHECK_HEADER(fftw3.h,
   AC_CHECK_LIB([fftw3], fftw_execute,,
           AC_MSG_WARN([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_WARN([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***])
     , dnl Don't use standard includes.
     )

# The boost library
# Currently only headers are used
AC_CHECK_HEADER([boost/shared_ptr.hpp],,
    AC_CHECK_FILE(["boost/shared_ptr.hpp"],AC_SUBST(ACFLAGS,"-I- -I boost -I ."),
    AC_MSG_ERROR([*** could not find the boost headers ***
*** Please install boost_1_32_0 or later from http://www.boost.org ***
*** (it is sufficient to move the boost_1_32_0/boost directory into .) ***])))

# Checks for libraries.
AC_CHECK_LIB([curses], [tparm])
AC_CHECK_LIB([m], [sqrt])
AC_CHECK_LIB([readline], [readline])

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_HEADER_STDBOOL
AC_CHECK_HEADERS([fenv.h])
AC_CHECK_HEADERS([stddef.h])

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

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
