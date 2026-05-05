// this file is not used by cmake build.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_LIBGL

#ifndef GLEW_INCLUDE
#include "GL/glew.h"
#else
#include GLEW_INCLUDE
#endif

#ifdef HAVE_LIBOSMESA
#define GLEW_OSMESA
#define APIENTRY
#endif

#include "backports/glew/src/glew.c"

#endif /* HAVE_LIBGL */
