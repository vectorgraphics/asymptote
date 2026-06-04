#                                               -*- Autoconf -*-
# Run autoheader and autoconf to produce a header and configure script from
# this file.

AC_PREREQ([2.71])
AC_INIT([Asymptote],[3.13git],[https://github.com/vectorgraphics/asymptote/issues])
VERSION=$PACKAGE_VERSION
AC_SUBST(VERSION)
m4_include([ax_pthread.m4])
m4_include([pkg.m4])

AH_TOP([#pragma once])

AC_CANONICAL_HOST
AC_CANONICAL_TARGET

MACOS_UNIVERSAL=no
MACOS_BUNDLING=no
MACOS_MIN_VERSION=
IS_DARWIN=no
case "$host_os" in
    darwin*)
      IS_DARWIN=yes
      MACOS_MIN_VERSION=12.0
    ;;
esac

AC_ARG_ENABLE(macos-universal,
[AS_HELP_STRING(--enable-macos-universal[[[=no]]],build universal (x86_64+arm64) binaries on macOS)],
[ if test "x$IS_DARWIN" = "xyes"; then
      if test "x$enableval" = "xno" ; then
         MACOS_UNIVERSAL=no
      else
         MACOS_UNIVERSAL=yes
      fi
   fi
])

AC_ARG_WITH(macos-min-version,
[AS_HELP_STRING(--with-macos-min-version=VERSION,minimum macOS deployment target (default: 12.0 on macOS))],
[ if test "x$IS_DARWIN" = "xyes"; then
      MACOS_MIN_VERSION=$withval
   fi
])

AC_ARG_ENABLE(macos-bundling,
[AS_HELP_STRING(--enable-macos-bundling[[[=no]]],bundle macOS-specific libraries (e.g., Vulkan) for distribution)],
[ if test "x$IS_DARWIN" = "xyes"; then
      MACOS_BUNDLING=$enableval
   fi
])

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

# macOS portability flags are applied only during the actual build, NOT during
# configure feature-detection tests. Injecting -arch into CFLAGS at configure
# time would cause AC_CHECK_LIB tests to require universal libraries, which
# breaks detection of arm64-only or x86_64-only SDK installations.
#
# Instead we emit MACOS_ARCH_FLAGS and MACOS_DEPLOYMENT_FLAGS as standalone
# Makefile variables that are appended to OPTS and the linker invocations.
MACOS_ARCH_FLAGS=
MACOS_DEPLOYMENT_FLAGS=
if test "x$IS_DARWIN" = "xyes"; then
   if test "x$MACOS_UNIVERSAL" = "xyes"; then
      MACOS_ARCH_FLAGS="-arch x86_64 -arch arm64"
      AC_MSG_NOTICE([macOS universal build enabled (x86_64 and arm64)])
   fi

   if test "x$MACOS_MIN_VERSION" != "x"; then
      MACOS_DEPLOYMENT_FLAGS="-mmacosx-version-min=$MACOS_MIN_VERSION"
      AC_MSG_NOTICE([macOS deployment target set to $MACOS_MIN_VERSION])
   fi
fi

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

licensedir=$docdir/licenses

AC_ARG_WITH(licensedir,
        [AS_HELP_STRING(--with-licensedir=PATH,
alternate license files installation directory)],
                     [if test "x$withval" != "x" ; then
                        licensedir=$withval
                      fi
                     ])
AC_SUBST(licensedir)

sysdir=$Datadir/asymptote

DEFS=""
OPTIONS="-D_FILE_OFFSET_BITS=64 "
INCL=""
LIBS=""
CPPFLAGS=$CPPFLAGS" -I. "

AC_ARG_ENABLE(texlive-build,
[AS_HELP_STRING(--enable-texlive-build[[[=no]]],
automatically determine sysdir from kpsewhich)],
[ if test "x$enableval" = "xyes" ; then
    sysdir=""
  fi
])

AC_ARG_ENABLE(relocatable,
[AS_HELP_STRING(--enable-relocatable[[[=no]]],
leave ASYMPTOTE_SYSDIR empty so the binary finds its data directory
relative to its own location at runtime (for portable/staged macOS installs))],
[ if test "x$enableval" = "xyes" ; then
    OPTIONS=$OPTIONS"-DIS_RELOCATABLE "
  fi
])

AC_DEFINE_UNQUOTED(ASYMPTOTE_SYSDIR,"$sysdir",
        [System directory for global .asy files])
AC_DEFINE_UNQUOTED(ASYMPTOTE_DOCDIR,"$docdir",
        [Directory for documentation])
AC_DEFINE_UNQUOTED(ASYMPTOTE_LICENSEDIR,"$licensedir",
        [Directory for license files])

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
   $1="$2 "$(pkg-config --silence-errors $3 $4 2>/dev/null || echo),
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
[AS_HELP_STRING(--enable-gc-debug[[[=no]]],enable (slow) garbage collector debugging)],
[ if test "x$enable_gc" != "xno" ; then
    if test "x$enableval" = "xyes" ; then
      AC_DEFINE(GC_DEBUG,1,[GC Debug is enabled])
      AC_MSG_NOTICE([*** Enabling GC debugging: remember to make clean ***])
      AC_MSG_NOTICE([*** Set the environment variable GC_FIND_LEAK at runtime ***])
    fi
 fi
])

AC_ARG_ENABLE(gc-full-debug,
[AS_HELP_STRING(--enable-gc-full-debug[[[=no]]],enable (very slow) garbage collector backtrace)],
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

dnl Check for vulkan/vulkan.hpp and VK_API_VERSION_1_4 using a C++ compile test.
dnl On some distros (e.g., Ubuntu with LunarG libvulkan-dev) the C++ header
dnl may be missing, or the headers may be too old to support Vulkan 1.4.
AC_LANG_PUSH([C++])
AC_COMPILE_IFELSE(
  [AC_LANG_PROGRAM([[#include <vulkan/vulkan.hpp>]],
                   [[(void)(uintptr_t)VK_API_VERSION_1_4;]])],
  [have_vulkan_hpp=yes],
  [have_vulkan_hpp=no]
)
AC_LANG_POP([C++])

case "$OSTYPE" in
     *)
          if test "x$have_vulkan_hpp" != "xyes"; then
            AC_MSG_NOTICE([*** Could not find vulkan/vulkan.hpp with VK_API_VERSION_1_4: will compile without Vulkan support ***])
          else
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
          dnl On some distros (e.g., Ubuntu) libglslang.a is a thin C wrapper and
          dnl the actual C++ implementation is in companion libraries. Try -lglslang
          dnl alone first; if that fails, try with companions.
          ASY_GLSLANG_LIBS=""
          LIBS_SAVE=$LIBS
          LIBS="$LIBS -lglslang"
          AC_LINK_IFELSE(
            [AC_LANG_CALL([], [glslang_initialize_process])],
            [ASY_GLSLANG_LIBS="-lglslang"],
            []
          )
          dnl If -lglslang alone didn't work, try with companion libraries
          if test "x$ASY_GLSLANG_LIBS" = "x"; then
            LIBS="$LIBS_SAVE -lglslang $VULKAN_LIBS -lOGLCompiler"
            AC_LINK_IFELSE(
              [AC_LANG_CALL([], [glslang_initialize_process])],
              [ASY_GLSLANG_LIBS="-lglslang $VULKAN_LIBS -lOGLCompiler"],
              []
            )
          fi
          LIBS=$LIBS_SAVE
          if test "x$ASY_GLSLANG_LIBS" != "x"; then
            AC_DEFINE(HAVE_LIBVULKAN,1,
            DEFINE_LIB([vulkan]))
            ASY_VULKAN_LIBS="$ASY_VULKAN_LIBS $ASY_GLSLANG_LIBS -lSPIRV-Tools-opt -lSPIRV-Tools"
          else
            AC_MSG_NOTICE([*** Could not find libglslang: will compile without Vulkan support ***])
            ASY_VULKAN_LIBS=""
            LIBS=$LIBS_SAVE
          fi
          ],
          AC_MSG_NOTICE([*** Could not find libvulkan: will compile without Vulkan support ***]))
          fi
esac
fi

# Check if Vulkan libraries should be bundled (macOS only).
# Bundling requires --enable-macos-bundling AND Vulkan libraries to be present.
# Use $host_os (from AC_CANONICAL_HOST) rather than $OSTYPE, which is a
# bash-ism that may be empty when configure runs as /bin/sh.
# Check ASY_VULKAN_LIBS rather than $LIBS: CHECK_LIB_STATIC appends -lvulkan
# to $LDFLAGS, not $LIBS, so $LIBS never contains -lvulkan.
BUNDLE_VULKAN=no
case "$host_os" in
    darwin*)
        if test "x$MACOS_BUNDLING" = "xyes" -a "x$ASY_VULKAN_LIBS" != "x"; then
            BUNDLE_VULKAN=yes
            AC_MSG_NOTICE([Will bundle Vulkan libraries for macOS distribution])
        fi
    ;;
esac
AC_SUBST(BUNDLE_VULKAN)

# Stage the vendored license texts for the dylibs bundled by
# --enable-macos-bundling.  Licenses must ship exactly when (and only when) the
# dylibs do, so this defaults to "auto" = follow BUNDLE_VULKAN;
# --disable-bundled-dylibs-licenses is an escape hatch.  When enabled, compile
# with -DASY_BUNDLED_DYLIBS so the runtime --licenses output gains the bundled
# section (see settings.cc and build-scripts/bundled-dylib-licenses.tsv).
AC_ARG_ENABLE(bundled-dylibs-licenses,
[AS_HELP_STRING(--enable-bundled-dylibs-licenses[[[=auto]]],
stage license files for the dylibs bundled by --enable-macos-bundling
(default: auto = follow whether Vulkan libraries are bundled))],
[], [enable_bundled_dylibs_licenses=auto])
AS_IF([test "x$enable_bundled_dylibs_licenses" = "xauto"],
  [enable_bundled_dylibs_licenses=$BUNDLE_VULKAN])
AS_IF([test "x$enable_bundled_dylibs_licenses" = "xyes"],
  [OPTIONS=$OPTIONS"-DASY_BUNDLED_DYLIBS "])
AC_SUBST([BUNDLED_DYLIBS_LICENSES], [$enable_bundled_dylibs_licenses])

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

ASY_GL_LIBS=""
SHIM_LDFLAGS=""

if test "x$enable_gl" != "xno"; then
   AC_CHECK_HEADERS([ncurses/curses.h ncurses.h curses.h],[break])

case "$OSTYPE" in
     darwin*)
              # On macOS 11+, -undefined dynamic_lookup is incompatible with
              # chained fixups. Add -no_fixup_chains to use traditional fixups
              # which work correctly with dynamic_lookup for dlopen'd libraries.
              SHIM_LDFLAGS="-undefined dynamic_lookup -Wl,-no_fixup_chains"
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
AC_SUBST(SHIM_LDFLAGS)
AC_SUBST(MACOS_UNIVERSAL)
AC_SUBST(MACOS_MIN_VERSION)
AC_SUBST(MACOS_ARCH_FLAGS)
AC_SUBST(MACOS_DEPLOYMENT_FLAGS)

# Determine which shim shared libraries (loaded via dlopen at runtime) to
# build and link the asy binary against. Empty when both renderers are
# disabled so the main binary does not depend on libglfw, libGL, or libvulkan.
SHIMLIBS=""
if test "x$enable_vulkan" != "xno"; then
    SHIMLIBS="$SHIMLIBS libasyvulkan.so"
fi
if test "x$enable_gl" != "xno"; then
    SHIMLIBS="$SHIMLIBS libasyopengl.so"
fi
AC_SUBST(SHIMLIBS)

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
