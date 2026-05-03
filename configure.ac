#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ([2.71])
AC_INIT([Asymptote],[3.11git],[https://github.com/vectorgraphics/asymptote/issues])
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

# Checks for libraries.
#AC_SEARCH_LIBS([lgamma],[m c],,
#AC_MSG_ERROR([*** Please install libm on your system ***]))

# ----------------- zlib ----------------------------
AC_CHECK_LIB(
   [z],
   [deflate],
   ,
   AC_MSG_ERROR([*** Please install libz or zlib-devel on your system ***])
)
AC_DEFINE(HAVE_ZLIB,1,[ZLib library present])

# -------------- threads -------------------


GCOPTIONS="--disable-shared --disable-dependency-tracking --enable-large-config"
AC_ARG_ENABLE(
   threads,
   [AS_HELP_STRING(--enable-threads[[[=yes]]],enable POSIX threads)]
)

if test "x$enable_threads" != "xno"; then
   AX_PTHREAD
else
   GCOPTIONS=$GCOPTIONS"--disable-threads "
fi

# --------------- sigsegv ------------------------------

AC_ARG_ENABLE(sigsegv,
[AS_HELP_STRING(--enable-sigsegv[[[=yes]]],enable GNU Stack Overflow Handler)])

if test "x$enable_sigsegv" != "xno"; then
AC_CHECK_LIB([sigsegv], [stackoverflow_install_handler])
fi

AC_CHECK_LIB([rt], [sched_yield])

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
      LSP_CMAKE_OPTIONS=$LSP_CMAKE_OPTIONS"-DLSPCPP_GC_DOWNLOADED_ROOT=../\$(GC) "
   else
      LSP_CMAKE_OPTIONS=$LSP_CMAKE_OPTIONS"-DLSPCPP_SUPPORT_BOEHM_GC=OFF "
   fi

   AC_DEFUN([ENABLE_LSP_MACRO],
      AC_SUBST(LSP_CXX_BUILD_FLAGS)
      AC_SUBST(LSP_CMAKE_OPTIONS)
      AC_DEFINE(HAVE_LSP,1,DEFINE([Language server protocol]))
   )
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
   AC_CHECK_HEADERS([ncursesw/curses.h ncurses/curses.h ncurses.h curses.h],[break])
   CHECK_LIB_STATIC(ncurses,ncurses,setupterm,HAVE_LIBCURSES,
                     AC_CHECK_LIB(curses,setupterm))
fi

# --------------- gc -------------------------------

GCNAME="Boehm Garbage Collector"

AC_ARG_ENABLE(gc,
[AS_HELP_STRING(--enable-gc[[[=yes]]],enable local $GCNAME)])

if test "x$enable_gc" != "xno" ; then
  AC_DEFINE(USEGC,1,[GC Enabled])
  GCLIB="\$(GC)/.libs/libgc.a"
  INCL=$INCL"-I\$(GC)/include "
  AC_MSG_NOTICE([$GCNAME is enabled])
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
  PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,curl)
  AC_CHECK_HEADER(curl/curl.h,
  CHECK_LIB_STATIC(libcurl,curl,curl_easy_init,HAVE_LIBCURL,
  AC_MSG_NOTICE([*** Could not find libcurl: will compile without optional URL support. ***])),
     AC_MSG_NOTICE([*** Header file curl.h not found: will compile without optional URL support. ***]))
else
  AC_MSG_NOTICE([*** libcurl support disabled by configure flag: will compile without optional URL support. ***])
fi

# --------------------- fftw -------------------------

AC_ARG_ENABLE(fftw,
[AS_HELP_STRING(--enable-fftw[[[=yes]]],enable FFTW Library)])

if test "x$enable_fftw" != "xno"; then
PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,fftw3)
AC_CHECK_HEADER(fftw3.h,
CHECK_LIB_STATIC(fftw3,fftw3,fftw_execute,HAVE_LIBFFTW3,
           AC_MSG_NOTICE([*** Could not find libfftw3: will compile without optional fast Fourier transforms. ***])),
     AC_MSG_NOTICE([*** Header file fftw3.h not found: will compile without optional fast Fourier transforms. ***]))
fi

# ------------------- eigen -------------------------

AC_ARG_ENABLE(eigen,
[AS_HELP_STRING(--enable-eigen[[[=yes]]],enable Eigen Library)])

if test "x$enable_eigen" != "xno"; then
PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,eigen3)
AC_CHECK_HEADERS(Eigen/Dense)
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
  PKG_CONFIG(CPPFLAGS,$CPPFLAGS,--cflags,gsl)
  AC_CHECK_HEADER(gsl/gsl_sf.h,
  CHECK_LIB_STATIC(gsl,gsl,gsl_sf_debye_6,HAVE_LIBGSL,
           AC_MSG_NOTICE([*** Could not find libgsl: will compile without optional special functions. ***])),
     AC_MSG_NOTICE([*** Header file gsl_sf.h not found: will compile without optional special functions. ***]))
fi

# ----------- glm
# glm is header only and is already on the include path
AC_CHECK_HEADER(
   glm/glm.hpp,
   [AC_DEFINE(
      HAVE_LIBGLM,1,
      DEFINE([the <glm/glm.hpp> header]))
   ],
   AC_MSG_NOTICE([*** Could not find glm header files: will compile without WebGL or OpenGL/Vulkan support ***])
)

TINYEXR_ROOT=tinyexr
INCL=$INCL" -I$TINYEXR_ROOT"
AC_SUBST(TINYEXR_ROOT)

# ------------------- GLFW -----------------

if test "x$enable_vulkan" != "xno" -o "x$enable_gl" != "xno"; then
   AC_CHECK_LIB([glfw], [glfwShowWindow],
                [AC_DEFINE(HAVE_LIBGLFW,1,DEFINE_LIB([GLFW]))
                 LIBS=$LIBS"-lglfw "],
                AC_MSG_NOTICE([*** Could not find libglfw: will compile without Vulkan or OpenGL support ***]))
fi

# ------------------- Vulkan -----------------

AC_ARG_ENABLE(vulkan,
[AS_HELP_STRING(--enable-vulkan[[[=yes]]],enable Vulkan Library)])

VMA_REPO_ROOT=VulkanMemoryAllocator
VMA_IMPL_FILES=""
if test "x$enable_vulkan" != "xno"; then
INCL=" -I$VMA_REPO_ROOT/include "$INCL" -Ithirdparty_impl/vk-mem-allocator_impl/include "
VMA_IMPL_FILES=$VMA_IMPL_FILES" vma_cxx vma_impl"

ASY_VULKAN_LIBS=""
AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])

case "$OSTYPE" in
     *)
          AC_CHECK_LIB([gccpp],[GC_throw_bad_alloc])
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
          ASY_VULKAN_LIBS="-lvulkan $VULKAN_LIBS-lSPIRV"
          AC_CHECK_LIB([glslang],[glslang_initialize_process],
          [
          AC_DEFINE(HAVE_LIBVULKAN,1,
          DEFINE_LIB([vulkan]))
          ASY_VULKAN_LIBS="$ASY_VULKAN_LIBS -lglslang -lSPIRV-Tools-opt -lSPIRV-Tools"
          ],

          [
          ASY_VULKAN_LIBS="$ASY_VULKAN_LIBS -lOGLCompiler"
          AC_CHECK_LIB([glslang],[glslang_finalize_process],
          [
          AC_DEFINE(HAVE_LIBVULKAN,1,
          DEFINE_LIB([vulkan]))
          ASY_VULKAN_LIBS="$ASY_VULKAN_LIBS -lglslang"
          ],
          [
          AC_MSG_NOTICE([*** Could not find libglslang: will compile without Vulkan support ***])
          ASY_VULKAN_LIBS=""
          LIBS=$LIBS_SAVE
          ]
          )
          ]
          )
          ],
          AC_MSG_NOTICE([*** Could not find libvulkan: will compile without Vulkan support ***]))
esac
fi

# ------------------- OpenGL (fallback) -----------------

AC_ARG_ENABLE(gl,
[AS_HELP_STRING(--enable-gl[[[=yes]]],enable OpenGL Library)])

AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
   [#include <GL/gl.h>]
   [#if !defined(GL_VERSION_4_3) && !defined(GL_ARB_shader_storage_buffer_object)]
   [#error]
   [#endif ]
   )],
   AC_DEFINE(HAVE_SSBO,1,DEFINE([GLSL shader storage buffer objects])))

AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
   [#include <GL/gl.h>]
   [#if !defined(GL_VERSION_4_3) && !defined(GL_ARB_compute_shader)]
   [#error]
   [#endif ]
   )],
   AC_DEFINE(HAVE_COMPUTE_SHADER,1,DEFINE([GLSL compute shaders])))

# -------------- offscreen rendering

AC_ARG_ENABLE(offscreen,
[AS_HELP_STRING(--enable-offscreen[[[=no]]],use OSMesa library to implement offscreen rendering)])

ASY_GL_LIBS=""

if test "x$enable_gl" != "xno"; then
   AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])

case "$OSTYPE" in
     darwin*)
              AC_CHECK_LIB([gccpp],[GC_throw_bad_alloc])
              AC_CHECK_HEADER(OpenGL/gl.h,
                             [AC_DEFINE(HAVE_LIBGL,1,
                             DEFINE([<gl.h> header]))
                              ASY_GL_LIBS="-framework OpenGL -framework Cocoa "
                              INCL=$INCL" -Ibackports/glew/include"
                              GLEW="glew.o "],
                                 AC_MSG_NOTICE([*** Could not find OpenGL: will compile without OpenGL support ***]))
        ;;
     *)
         # Use AC_LINK_IFELSE instead of AC_CHECK_LIB to avoid adding -lGL to LIBS.
         # The OpenGL library will only be linked into libasyopengl.so, not the main binary.
         LIBS_SAVE=$LIBS
         LIBS="$LIBS -lGL"
         AC_LINK_IFELSE([AC_LANG_CALL([], [glDepthMask])],
                                 [AC_DEFINE(HAVE_LIBGL,1,
                             DEFINE_LIB([GL]))
                             ASY_GL_LIBS="-lGL "
                             GLEW="glew.o "
                             INCL=$INCL" -Ibackports/glew/include"],
                                   AC_MSG_NOTICE([*** Could not find libGL: will compile without OpenGL support ***]))
         LIBS=$LIBS_SAVE
esac
   if test "x$enable_offscreen" = "xyes"; then
           AC_CHECK_LIB([OSMesa],OSMesaCreateContext,,
                        AC_MSG_NOTICE([*** Could not find libOSMesa: will compile without offscreen rendering support ***]))
   fi
fi

# ----------------- xdr ------------------------

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
AC_SUBST(ASY_VULKAN_LIBS)
AC_SUBST(ASY_GL_LIBS)

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