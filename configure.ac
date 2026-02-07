#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ([2.71])
AC_INIT([Asymptote],[3.07git],[https://github.com/vectorgraphics/asymptote/issues])
VERSION=$PACKAGE_VERSION
AC_SUBST(VERSION)
m4_include([ax_pthread.m4])
m4_include([pkg.m4])

AH_TOP([#pragma once])

AC_CANONICAL_HOST
AC_CANONICAL_TARGET

test "$prefix" = NONE && prefix=/usr/local
Datadir=$datadir
test "$Datadir" = '${datarootdir}' && Datadir=$datarootdir
test "$Datadir" = '${prefix}/share' && Datadir=$prefix/share
AC_SUBST(Datadir)

AC_ARG_ENABLE(lsp,
[AS_HELP_STRING(--enable-lsp[[[=yes]]],enable Language Server Protocol)])
cxxstd="c++17"

CXXFLAGS=$CXXFLAGS" -std=$cxxstd"

test "$CFLAGS" || CFLAGS="-g -O3"

AC_C_BIGENDIAN(
   [AC_DEFINE(WORDS_BIGENDIAN,1,[Build is big endian])],
   [],
   [],
   [AC_DEFINE(UNIVERSAL_ENDIAN_BUILD,1,[Endianness is determined at compile time])]
)

echo Compiling with $CXXFLAGS $CFLAGS

# --------------- begin vcpkg option/check ---------------------
# result:
# with_vcpkg is yes/no, depending if user wants to use vcpkg or not
# vcpkg_exec is the executable of vcpkg. this is only given if use_vcpkg is yes.

AC_ARG_WITH(vcpkg,
       [AS_HELP_STRING(--with-vcpkg=[[[no]]],
      [Use vcpkg. If VCPKG_ROOT environment variable is set, will use that vcpkg. Otherwise searches in PATH.]
      )
],[
   if test "x$VCPKG_ROOT" = "x"; then  # vcpkg root not given
      echo "VCPKG_ROOT is not given; searching vcpkg in path"
      AC_CHECK_PROG(has_vcpkg,vcpkg,true)
      if test "x$has_vcpkg" = "xtrue"; then
         vcpkg_exec=vcpkg
      fi
   else
      echo "VCPKG_ROOT is given; is using VCPKG_ROOT"
      AC_CHECK_FILE($VCPKG_ROOT/vcpkg,[
         vcpkg_exec=$VCPKG_ROOT/vcpkg
      ])
   fi

   # ensure vcpkg is specified
   if test "x$vcpkg_exec" = "x"; then
      AC_MSG_ERROR([Please ensure vcpkg is available through VCPKG_ROOT or in PATH])
   fi
   echo "Using vcpkg executable $vcpkg_exec"
],[
with_vcpkg=no
])

#----------------- end vcpkg option check --------------------

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
[@echo \*\*\*\*\*\*\*\*\*\* Please install texi2dvi or put https://asymptote.sourceforge.io/asymptote.pdf in the doc directory: cannot execute texi2dvi])

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
AC_PROG_LEX(noyywrap)
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_CC
AC_PROG_MAKE_SET

AC_ARG_VAR([BISON], [bison command])
AC_CHECK_PROG([BISON], [bison -y], [yes], [no])
AS_IF([test "x$BISON" = "xno"], [AC_MSG_ERROR([bison not found])])

AC_ARG_VAR([FLEX], [flex command])
AC_CHECK_PROG([FLEX], [flex -y], [yes], [no])
AS_IF([test "x$FLEX" = "xno"], [AC_MSG_ERROR([flex not found])])

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
Define to 1 if you have `$1`.
])

AC_DEFUN([DEFINE_LIB],[
Define to 1 if you have the `$1` library (-l$1).
])

ASYGLVERSION=1.03

DEFS=""
OPTIONS=$OPTIONS"-D_FILE_OFFSET_BITS=64 "
INCL=""
OPTIONS=""
LIBS=""
CPPFLAGS=$CPPFLAGS" -I. "

# ------------- vcpkg initialization ---------------
# vcpkg_triplet

AC_ARG_WITH(vcpkg-target-triplet,
   AS_HELP_STRING(--with-vcpkg-target-triplet[[[=<target-triplet>]]],
      [
         specify custom vcpkg triplet. This option is only relevant if vcpkg is used.
         If left blank, will try to determine the triplet
      ]
   ), [  # given
   if test "x$with_vcpkg" != "xno" -a "x$withval" != "xno"; then
      vcpkg_triplet=$withval
   fi
   ],[  # not given
   if test "x$with_vcpkg" != "xno"; then
      case $target_cpu in
         x86_64)
         triplet_arch=x64
         ;;
         aarch64)
         triplet_arch=arm64
         ;;
         i*86)
         triplet_arch=x86
         ;;
         *)
         AC_MSG_NOTICE([Cannot determine architecture, will leave triplet blank])
         ;;
      esac

      case "$OSTYPE" in
         darwin*)
         triplet_os=osx
         ;;
         *)
         triplet_os=linux
         ;;
      esac

      if test "x$triplet_arch" != "x" -a "x$triplet_os" != "x"; then
         vcpkg_triplet="${triplet_arch}-${triplet_os}"
         AC_MSG_NOTICE([Determined vcpkg triplet as $vcpkg_triplet])
      else
         AC_MSG_ERROR([Cannot determine vcpkg triplet for usage])
      fi
   fi
   ]
)

if test "x$with_vcpkg" != "xno"; then
   VCPKG_INSTALL_ARGUMENTS="--triplet=$vcpkg_triplet "
   VCPKG_PKG_CONFIG_LOC="vcpkg_installed/$vcpkg_triplet/lib/pkgconfig"
   INCL=$INCL"-Ivcpkg_installed/$vcpkg_triplet/include "
   LIBS=$LIBS"-Lvcpkg_installed/$vcpkg_triplet/lib "

   VCPKG_LIBS_TO_USE_PKG_CONFIG=""
   AC_DEFUN([ADD_VCPKG_LIB_FOR_PKGCONFIG],[VCPKG_LIBS_TO_USE_PKG_CONFIG=$VCPKG_LIBS_TO_USE_PKG_CONFIG"$1 "])
   AC_DEFUN([ADD_VCPKG_FEATURE],[VCPKG_INSTALL_ARGUMENTS=$VCPKG_INSTALL_ARGUMENTS"--x-feature=$1 "])
fi


AC_ARG_WITH(vcpkg-host-triplet,
   AS_HELP_STRING(--with-vcpkg-host-triplet[[[=<host-triplet>]]],
      [
         specify custom vcpkg host triplet. This option is only relevant if vcpkg is ued.
         If left blank, will try to determine the triplet
      ]
   ), [  # given
   if test "x$with_vcpkg" != "xno" -a "x$withval" != "xno"; then
      VCPKG_INSTALL_ARGUMENTS=$VCPKG_INSTALL_ARGUMENTS"--host-triplet=$withval "
   fi
   ],
)



# Checks for libraries.
#AC_SEARCH_LIBS([lgamma],[m c],,
#AC_MSG_ERROR([*** Please install libm on your system ***]))

# ----------------- zlib ----------------------------
if test "x$with_vcpkg" != "xno"; then
   ADD_VCPKG_LIB_FOR_PKGCONFIG(zlib)
else
   AC_CHECK_LIB(
      [z],
      [deflate],
      ,
      AC_MSG_ERROR([*** Please install libz or zlib-devel on your system ***])
   )
fi
AC_DEFINE(HAVE_ZLIB,1,[ZLib library present])

# -------------- threads -------------------


GCOPTIONS="--disable-shared --disable-dependency-tracking --enable-large-config"
AC_ARG_ENABLE(
   threads,
   [AS_HELP_STRING(--enable-threads[[[=yes]]],enable POSIX threads)]
)

if test "x$enable_threads" != "xno"; then
   if test "x$with_vcpkg" != "xno"; then
      ADD_VCPKG_FEATURE(threading)
      AC_DEFINE(HAVE_PTHREAD,1)
      OPTIONS=$OPTIONS" -pthread"
   else
      AX_PTHREAD
   fi
else
   GCOPTIONS=$GCOPTIONS"--disable-threads "
fi

# --------------- sigsegv ------------------------------

if test "x$with_vcpkg" = "xno"; then
   AC_ARG_ENABLE(sigsegv,
   [AS_HELP_STRING(--enable-sigsegv[[[=yes]]],enable GNU Stack Overflow Handler)])

   if test "x$enable_sigsegv" != "xno"; then
   AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])
   fi

   AC_CHECK_LIB([rt], [sched_yield])
fi

# ----------------- lsp -------------------------------

LSP_ROOT=LspCpp
AC_SUBST(LSP_ROOT)

LSPLIBS=
LSPLIB=
if test "x$enable_lsp" != "xno" -a "x$enable_threads" != "xno"; then
   LSPLIB=$LSP_ROOT/liblspcpp.a
   LSPLIBS=$LSPLIBS" -L$LSP_ROOT -L$LSP_ROOT/third_party/uri/src -llspcpp -lnetwork-uri"
   LSP_CMAKE_OPTIONS="-DLSPCPP_USE_CPP17=ON "

   if test "x$enable_gc" != "xno"; then
      LSP_CMAKE_OPTIONS=$LSP_CMAKE_OPTIONS"-DLSPCPP_SUPPORT_BOEHM_GC=ON "
      if test "x$with_vcpkg" = "xno"; then
          LSP_CMAKE_OPTIONS=$LSP_CMAKE_OPTIONS"-DLSPCPP_GC_DOWNLOADED_ROOT=../\$(GC) "
      fi
   else
      LSP_CMAKE_OPTIONS=$LSP_CMAKE_OPTIONS"-DLSPCPP_SUPPORT_BOEHM_GC=OFF "
   fi

   AC_DEFUN([ENABLE_LSP_MACRO],
      AC_SUBST(LSP_CXX_BUILD_FLAGS)
      AC_SUBST(LSP_CMAKE_OPTIONS)
      AC_DEFINE(HAVE_LSP,1,DEFINE([Language server protocol]))
   )
   if test "x$with_vcpkg" != "xno"; then
      ADD_VCPKG_FEATURE(lsp)
      RELATIVE_LSP_VCPKG_ROOT=../vcpkg_installed/$vcpkg_triplet
      LSP_CXX_BUILD_FLAGS=$LSP_BUILD_FLAGS"-I$RELATIVE_LSP_VCPKG_ROOT/include -L$RELATIVE_LSP_VCPKG_ROOT/lib "
      LSP_CMAKE_OPTIONS=$LSP_CMAKE_OPTIONS"-DUSE_SYSTEM_RAPIDJSON=True -DCMAKE_PREFIX_PATH=\"$RELATIVE_LSP_VCPKG_ROOT\" "
   fi
   ENABLE_LSP_MACRO
fi

AC_ARG_ENABLE(readline,
[AS_HELP_STRING(--enable-readline[[[=yes]]],enable GNU Readline Library)])

AC_ARG_ENABLE(static,
[AS_HELP_STRING(--enable-static[[[=no]]],link against static libraries)])

AC_DEFUN([PKG_CONFIG],[
ifdef(
   [PKG_CHECK_MODULES],
   $1="$2 "$(pkg-config --silence-errors $3 $4 || echo),
   $1="$2 "
)
])

STATIC=""
DYNAMIC=""
if test "x$enable_static" = "xyes"; then
  STATIC="-Wl,-Bstatic "
  DYNAMIC="-Wl,-Bdynamic "
fi

AC_DEFUN([CHECK_LIB_STATIC],[
   PKG_CONFIG(PKG_FLAGS,,--libs,$1)
   if test "x$PKG_FLAGS" == "x "; then
      PKG_FLAGS=-l"$2"
      echo will try linking with $PKG_FLAGS
   fi
   LDFLAGS_SAVE=$LDFLAGS
   LDFLAGS="$LDFLAGS "$STATIC"$PKG_FLAGS "$DYNAMIC
   AC_MSG_NOTICE([Checking $2 for function $3... PKG_FLAGS:$PKG_FLAGS])
   AC_CHECK_LIB(
      $2,$3,
      AC_DEFINE($4,1,DEFINE_LIB($2)),
      if test "x$enable_static" = "xyes"; then
      [
         echo "Static library not found; will check for dynamic version"
         LDFLAGS="$LDFLAGS_SAVE ""$PKG_FLAGS "
         AS_UNSET([ac_cv_lib_$2_$3])

         AC_CHECK_LIB(
            $2,$3,
            AC_DEFINE($4,1,DEFINE_LIB($2)),
            [
               LDFLAGS=$LDFLAGS_SAVE
               $5
            ],
            $PKG_FLAGS
         )
      ]
      else
      [
         LDFLAGS=$LDFLAGS_SAVE
         $5
      ]
      fi,
      $PKG_FLAGS
   )
])

if test "x$enable_readline" != "xno"; then
if test "x$with_vcpkg" != "xno"; then  # vcpkg
   ADD_VCPKG_FEATURE(readline)
   AC_DEFINE(HAVE_NCURSES_CURSES_H)
   AC_DEFINE(HAVE_LIBCURSES)
   AC_DEFINE(HAVE_LIBREADLINE)
   ADD_VCPKG_LIB_FOR_PKGCONFIG(readline)
   ADD_VCPKG_LIB_FOR_PKGCONFIG(ncurses)
else  # managed by the system
   PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,readline)

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
   CHECK_LIB_STATIC(readline,readline,history_list,HAVE_LIBREADLINE,
   readline="no"),readline="no")

   if test "x$readline" == "xno"; then
     AC_MSG_NOTICE([*** Could not find GNU readline 4.3 or later: will compile without readline support ***])
     AC_CHECK_LIB([edit],[readline])
     AC_CHECK_HEADERS(editline/readline.h)
   fi

   PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,tinfo)
   CHECK_LIB_STATIC(tinfo,tinfo,tgetent,HAVE_LIBTINFO,AC_MSG_NOTICE([perhaps tgetent is in -lncurses]))

   PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,ncurses)
   AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])
   CHECK_LIB_STATIC(ncurses,ncurses,setupterm,HAVE_LIBCURSES,
                     AC_CHECK_LIB(curses,setupterm))
fi
fi

# --------------- gc -------------------------------

GCNAME="Boehm Garbage Collector"

AC_ARG_ENABLE(gc,
[AS_HELP_STRING(--enable-gc[[[=yes]]],enable local $GCNAME)])

if test "x$enable_gc" != "xno" ; then
  if test "x$with_vcpkg" == "xno"; then
    AC_DEFINE(USEGC,1,[GC Enabled])
    GCLIB="\$(GC)/.libs/libgc.a"
    INCL=$INCL"-I\$(GC)/include "
    AC_MSG_NOTICE([$GCNAME is enabled])
  fi
else
  AC_MSG_NOTICE([*** $GCNAME disabled by configure flag: will compile without garbage collection. ***])
fi

AC_ARG_ENABLE(gc-debug,
[AS_HELP_STRING(--enable-gc-debug,enable (slow) garbage collector debugging)],
[ if test "x$enable_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
      AC_DEFINE(GC_DEBUG,1,[GC Debug is enabled])
      AC_MSG_NOTICE([*** Enabling GC debugging: remember to make clean ***])
      AC_MSG_NOTICE([*** Set the environment variable GC_FIND_LEAK at runtime ***])
    fi
 fi
])

AC_ARG_ENABLE(gc-full-debug,
[AS_HELP_STRING(--enable-gc-full-debug,enable (very slow) garbage collector backtrace)],
[ if test "x$enable_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
      AC_DEFINE(GC_DEBUG,1,[GC Debug is enabled])
      AC_DEFINE(GC_BACKTRACE,1,[GC backtrace is enabled])
      GCOPTIONS=$GCOPTIONS"--enable-gc-debug "
      AC_MSG_NOTICE([*** Enabling GC backtrace debugging; remember to make gc-clean ***])
    fi
 fi
])

# ---------------- curl ------------------------

AC_ARG_ENABLE(curl,
[AS_HELP_STRING(--enable-curl[[[=yes]]],enable libcurl and compile with optional URL support)])
if test "x$enable_curl" != "xno"; then
if test "x$with_vcpkg" != "xno"; then
   ADD_VCPKG_FEATURE(curl)
   AC_DEFINE(HAVE_LIBCURL)
   ADD_VCPKG_LIB_FOR_PKGCONFIG(libcurl)
else
  PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,curl)
  AC_CHECK_HEADER(curl/curl.h,
  CHECK_LIB_STATIC(libcurl,curl,curl_easy_init,HAVE_LIBCURL,
  AC_MSG_NOTICE([*** Could not find libcurl: will compile without optional URL support. ***])),
     AC_MSG_NOTICE([*** Header file curl.h not found: will compile without optional URL support. ***]))
fi
else
  AC_MSG_NOTICE([*** libcurl support disabled by configure flag: will compile without optional URL support. ***])
fi

# --------------------- fftw -------------------------

AC_ARG_ENABLE(fftw,
[AS_HELP_STRING(--enable-fftw[[[=yes]]],enable FFTW Library)])

if test "x$enable_fftw" != "xno"; then
if test "x$with_vcpkg" != "xno"; then  # vcpkg
ADD_VCPKG_FEATURE(fftw3)
AC_DEFINE(HAVE_LIBFFTW3)
ADD_VCPKG_LIB_FOR_PKGCONFIG(fftw3)
else  # system managed
PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,fftw3)
AC_CHECK_HEADER(fftw3.h,
CHECK_LIB_STATIC(fftw3,fftw3,fftw_execute,HAVE_LIBFFTW3,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))
fi
fi

# ------------------- eigen -------------------------

AC_ARG_ENABLE(eigen,
[AS_HELP_STRING(--enable-eigen[[[=yes]]],enable Eigen Library)])

if test "x$enable_eigen" != "xno"; then
if test "x$with_vcpkg" != "xno"; then
ADD_VCPKG_FEATURE(eigen3)
AC_DEFINE(HAVE_EIGEN_DENSE)
ADD_VCPKG_LIB_FOR_PKGCONFIG(eigen3)
else
PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,eigen3)
AC_CHECK_HEADERS(Eigen/Dense)
fi
fi

# Checks for header files.
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([fenv.h stddef.h libintl.h])
AC_CHECK_HEADERS(fpu_control.h)
AC_CHECK_FUNCS([feenableexcept])

# ------------------- GSL --------------------

AC_ARG_ENABLE(gsl,
[AS_HELP_STRING(--enable-gsl[[[=yes]]],enable GNU Scientific Library)])

if test "x$enable_gsl" != "xno"; then
if test "x$with_vcpkg" != "xno"; then  # vcpkg
   ADD_VCPKG_FEATURE(gsl)
   AC_DEFINE(HAVE_LIBGSL)
   LIBS=$LIBS"-lgsl "  # gsl's pkg-config file is incomplete
   ADD_VCPKG_LIB_FOR_PKGCONFIG(gsl)
else
  PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,gsl)
  PKG_CONFIG(LDFLAGS,$LDFLAGS,--libs,gsl)
  AC_CHECK_HEADER(gsl/gsl_sf.h,
  CHECK_LIB_STATIC(gsl,gsl,gsl_sf_debye_6,HAVE_LIBGSL,
           AC_MSG_NOTICE([*** Could not find libgsl: will compile without optional special functions. ***])),
     AC_MSG_NOTICE([*** Header file gsl_sf.h not found: will compile without optional special functions. ***]))
fi
fi

# ----------- glm
if test "x$with_vcpkg" != "xno"; then
   AC_DEFINE(HAVE_LIBGLM,1)
   LIBS=$LIBS"-lglm "
else
   # glm is header only and is already on the include path
   AC_CHECK_HEADER(
      glm/glm.hpp,
      [AC_DEFINE(
         HAVE_LIBGLM,1,
         DEFINE([the <glm/glm.hpp> header]))
      ],
      AC_MSG_NOTICE([*** Could not find glm header files: will compile without WebGL or Vulkan support ***])
   )
fi

# ----------- tinyexr
TINYEXR_ROOT=tinyexr
INCL=$INCL"-I$TINYEXR_ROOT "
AC_SUBST(TINYEXR_ROOT)

# ----------- vulkan
AC_ARG_ENABLE(vulkan,
[AS_HELP_STRING(--enable-vulkan[[[=yes]]],enable Vulkan Library)])

VMA_REPO_ROOT=VulkanMemoryAllocator
VMA_IMPL_FILES=""
if test "x$enable_vulkan" != "xno"; then
INCL="-I$VMA_REPO_ROOT/include "$INCL"-Ithirdparty_impl/vk-mem-allocator_impl/include "
VMA_IMPL_FILES=$VMA_IMPL_FILES" vma_cxx vma_impl"

if test "x$with_vcpkg" != "xno"; then  # managed by vcpkg
   ADD_VCPKG_FEATURE(vulkan)
   AC_DEFINE(HAVE_LIBVULKAN,1,[Vulkan is enabled])
   ADD_VCPKG_LIB_FOR_PKGCONFIG(glfw3)
   LIBS=$LIBS"-lvulkan -lMachineIndependent -lOSDependent -lGenericCodeGen -lSPIRV -lglslang "
else  # managed by the system
   AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])

case "$OSTYPE" in
     *)
         AC_CHECK_LIB([gccpp],[GC_throw_bad_alloc])
          AC_CHECK_LIB([glfw], [glfwShowWindow],
          [
          AC_CHECK_LIB([vulkan], [vkCreateInstance],
          [
          LIBS_SAVE=$LIBS
          case "$OSTYPE" in
               darwin*)
                 VULKAN_LIBS=""
               ;;
               *)
                 VULKAN_LIBS="-lMachineIndependent -lOSDependent -lGenericCodeGen "
               ;;
          esac
          LIBS=$LIBS"-lglfw -lvulkan "$VULKAN_LIBS"-lSPIRV "
          AC_CHECK_LIB([glslang],[glslang_initialize_process],
          [
          AC_DEFINE(HAVE_LIBVULKAN,1,
          DEFINE_LIB([vulkan]))
          LIBS=$LIBS"-lglslang -lSPIRV-Tools-opt -lSPIRV-Tools "
          ],

          [
          LIBS=$LIBS"-lOGLCompiler "
          AC_CHECK_LIB([glslang],[glslang_finalize_process],
          [
          AC_DEFINE(HAVE_LIBVULKAN,1,
          DEFINE_LIB([vulkan]))
          LIBS=$LIBS"-lglslang "
          ],
          [
          AC_MSG_NOTICE([*** Could not find libglslang: will compile without Vulkan support ***])
          LIBS=$LIBS_SAVE
          ]
          )
          ]
          )
          ],
          AC_MSG_NOTICE([*** Could not find libvulkan: will compile without Vulkan support ***]))
          ],
          AC_MSG_NOTICE([*** Could not find libglfw: will compile without Vulkan support ***]))
esac
fi
fi


# ------------------- VCPKG INSTALL

if test "x$with_vcpkg" != "xno"; then
AC_MSG_NOTICE([Running vcpkg install])
$vcpkg_exec install $VCPKG_INSTALL_ARGUMENTS

LIBS=$LIBS"$(PKG_CONFIG_PATH=$VCPKG_PKG_CONFIG_LOC pkg-config --libs $VCPKG_LIBS_TO_USE_PKG_CONFIG) "
CFLAGS=$CFLAGS" $(PKG_CONFIG_PATH=$VCPKG_PKG_CONFIG_LOC pkg-config --cflags-only-other $VCPKG_LIBS_TO_USE_PKG_CONFIG)"
INCL=$INCL"$(PKG_CONFIG_PATH=$VCPKG_PKG_CONFIG_LOC pkg-config --cflags-only-I $VCPKG_LIBS_TO_USE_PKG_CONFIG) "
fi

# ----------------- xdr ------------------------
# managed by the system here, unlike other packages

AC_ARG_ENABLE(xdr,
[AS_HELP_STRING(--enable-xdr[[[=yes]]],enable XDR/V3D support)])
if test "x$enable_xdr" != "xno"; then
case "$OSTYPE" in
   darwin* | FreeBSD)
         AC_DEFINE(HAVE_LIBTIRPC,1,DEFINE_LIB([tirpc]))
   ;;
   *)
   PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,libtirpc)
   AC_COMPILE_IFELSE(
      [AC_LANG_PROGRAM([#include "xstream.h"])],
      CHECK_LIB_STATIC(libtirpc,tirpc,xdrstdio_create,
      HAVE_LIBTIRPC,
      AC_MSG_NOTICE([*** Could not find libtirpc; XDR/V3D support disabled ***])
      ),
      AC_MSG_NOTICE([*** Broken rpc headers; XDR/V3D support disabled ***])
   )
   ;;
esac
fi

CPPFLAGS=$CPPFLAGS" $INCL"
CXX_STANDARD=$cxxstd

AC_SUBST(getopt)
AC_SUBST(ASYGLVERSION)
AC_SUBST(GCOPTIONS)
AC_SUBST(GCLIB)
AC_SUBST(GCPPLIB)
AC_SUBST(LSPLIB)
AC_SUBST(LSPLIBS)
AC_SUBST(INCL)
AC_SUBST(CXX_STANDARD)
AC_SUBST(DEFS)
AC_SUBST(OPTIONS)
AC_SUBST(GLEW)
AC_SUBST(VMA_IMPL_FILES)
# we are on C++ >= 11, so unordered map is a guarantee
AC_DEFINE(HAVE_UNORDERED_MAP,1,[Unordered map is present])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([ptrdiff_t])
AC_CHECK_TYPES([long long])
AC_CHECK_TYPES([long])
AC_C_CONST
AC_C_INLINE

# Checks for library functions.
AC_FUNC_FORK
AC_CHECK_FUNCS([dup2 floor memset strchr tgamma lgamma memrchr popcount])
AC_FUNC_STRFTIME
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FSEEKO

AC_CHECK_FUNCS(strptime)
AC_CHECK_FUNCS(strnlen)

AC_CONFIG_FILES([Makefile doc/Makefile doc/png/Makefile])
AC_CONFIG_HEADERS([config.h])
AC_OUTPUT
