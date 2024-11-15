// this file is not used by cmake build. An equivalent handling of libosmesa is done in
// backports/glew/CMakeLists.txt for cmake builds.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef GLEW_INCLUDE
#include <GL/glew.h>
#else
#include GLEW_INCLUDE
#endif

#ifdef HAVE_LIBGL
#ifdef HAVE_LIBOSMESA
#define GLEW_OSMESA
#define APIENTRY
#endif

#include "backports/glew/src/glew.c"
#endif /* HAVE_LIBGL */
