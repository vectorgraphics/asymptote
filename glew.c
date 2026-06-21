// this file is not used by cmake build.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_LIBGL

#include "backports/glew/include/GL/glew.h"
#ifndef __APPLE__
#include "backports/glew/include/GL/glxew.h"
#endif

#ifdef HAVE_LIBOSMESA
#define GLEW_OSMESA
#define APIENTRY
#endif

#include "backports/glew/src/glew.c"

#endif /* HAVE_LIBGL */
